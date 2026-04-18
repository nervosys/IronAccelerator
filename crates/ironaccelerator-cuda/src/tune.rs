//! Autotuner — benchmarks candidate implementations and caches the winner
//! per problem key. Timing uses [`Timer`](crate::events::Timer) so the
//! measurement is GPU-accurate (wall-clock via `cuEventElapsedTime`).
//!
//! Typical use: pass a set of `Candidate`s (tile sizes, kernel variants),
//! run `tune(...)`, and subsequent calls with the same `Key` return the
//! previous winner from the cache without re-benchmarking.

use crate::events::Timer;
use crate::Session;
use ironaccelerator_core::Result;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

/// One candidate implementation. `label` is for logging/metrics; `run`
/// executes exactly the kernel we want to time (it must enqueue on the
/// session's stream and return as soon as the work is submitted).
pub struct Candidate<'a> {
    pub label: &'static str,
    pub run: Box<dyn Fn(&Session) -> Result<()> + Send + Sync + 'a>,
}

#[derive(Debug, Clone)]
pub struct TuneResult {
    pub winner: &'static str,
    pub winner_ms: f32,
    pub samples: Vec<(&'static str, f32)>,
}

pub struct Autotuner<K: Eq + Hash + Clone> {
    cache: RwLock<HashMap<K, TuneResult>>,
    /// How many iterations to time per candidate (inside a single event pair).
    iters: u32,
    /// Warm-up iterations (untimed).
    warmup: u32,
}

impl<K: Eq + Hash + Clone> Default for Autotuner<K> {
    fn default() -> Self {
        Self { cache: RwLock::new(HashMap::new()), iters: 5, warmup: 1 }
    }
}

impl<K: Eq + Hash + Clone> Autotuner<K> {
    pub fn new(warmup: u32, iters: u32) -> Self {
        Self { cache: RwLock::new(HashMap::new()), iters, warmup }
    }

    pub fn len(&self) -> usize { self.cache.read().len() }
    pub fn is_empty(&self) -> bool { self.cache.read().is_empty() }

    /// Look up a cached decision.
    pub fn lookup(&self, key: &K) -> Option<TuneResult> { self.cache.read().get(key).cloned() }

    /// Benchmark each candidate and cache the winner. Returns the result
    /// even if a previous entry existed (overwrites).
    pub fn tune<'a>(&self, session: &Session, key: K, candidates: &[Candidate<'a>])
        -> Result<TuneResult>
    {
        assert!(!candidates.is_empty(), "autotuner needs at least one candidate");

        let mut samples: Vec<(&'static str, f32)> = Vec::with_capacity(candidates.len());
        for c in candidates {
            // Warm up.
            for _ in 0..self.warmup { (c.run)(session)?; }
            session.synchronize()?;

            // Timed window.
            let stream = session.stream().clone();
            let t = Timer::new(&stream)?;
            t.begin(&stream)?;
            for _ in 0..self.iters { (c.run)(session)?; }
            t.end(&stream)?;
            let total = t.elapsed_ms()?;
            let avg = total / self.iters as f32;
            samples.push((c.label, avg));
        }

        let (winner, winner_ms) = samples.iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .copied().unwrap();

        let res = TuneResult { winner, winner_ms, samples };
        self.cache.write().insert(key, res.clone());
        Ok(res)
    }

    /// Get-or-benchmark: returns cached result immediately if present, else
    /// benchmarks.
    pub fn get_or_tune<'a>(
        &self, session: &Session, key: K, candidates: &[Candidate<'a>],
    ) -> Result<TuneResult> {
        if let Some(r) = self.lookup(&key) { return Ok(r); }
        self.tune(session, key, candidates)
    }

    pub fn clear(&self) { self.cache.write().clear(); }
}

/// Simple shape key: `(M, N, K, dtype_discriminant)`. Callers can also
/// supply their own `K` type — the tuner is generic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmKey {
    pub m: u32, pub n: u32, pub k: u32, pub dtype: u8,
}

// Keep compile-time sanity that our public bounds are Send so tuners can be
// stashed in global state.
const _: fn() = || {
    fn is_send<T: Send + Sync>() {}
    is_send::<Autotuner<GemmKey>>();
    is_send::<Arc<Autotuner<GemmKey>>>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_key_is_hashable() {
        use std::collections::HashSet;
        let mut s = HashSet::new();
        s.insert(GemmKey { m: 4096, n: 4096, k: 4096, dtype: 0 });
        s.insert(GemmKey { m: 4096, n: 4096, k: 4096, dtype: 1 });
        s.insert(GemmKey { m: 4096, n: 4096, k: 4096, dtype: 0 });
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn empty_cache_returns_none() {
        let t = Autotuner::<GemmKey>::default();
        assert!(t.lookup(&GemmKey { m: 1, n: 1, k: 1, dtype: 0 }).is_none());
    }
}
