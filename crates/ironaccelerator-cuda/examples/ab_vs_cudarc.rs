//! Paired A/B comparison against cudarc, robust to a contended machine.
//!
//! Criterion measures each implementation in its own contiguous block, so any
//! drift in machine state during a run lands entirely on one side and shows up
//! as a difference that is not in the code. On a shared desktop that effect is
//! larger than the difference being measured — it previously produced a
//! reported 1.98× at 16 MiB where the truth was ~1.0×.
//!
//! The fix is pairing. Each sample measures IronAccelerator and cudarc
//! back-to-back, microseconds apart, and the statistic is the *per-pair ratio*.
//! Contention that spans a pair scales both sides and cancels in their ratio,
//! so the median ratio stays meaningful even while absolute timings swing by an
//! order of magnitude. Pair order alternates so that being measured first is
//! not systematically an advantage.
//!
//! A verdict is only issued when the bootstrap 95% confidence interval of the
//! median ratio excludes 1.0 — that is what makes a claim defensible rather
//! than an artefact of whatever else the machine was doing.
//!
//! ```text
//! cargo run --release -p ironaccelerator-cuda --example ab_vs_cudarc
//! CUDA_VISIBLE_DEVICES=1 cargo run --release -p ironaccelerator-cuda --example ab_vs_cudarc
//! ```

use ironaccelerator_cuda::cudarc_compat as iron;
use ironaccelerator_cuda::cudarc_compat::CudaStreamExt;
use std::time::{Duration, Instant};

const BOOTSTRAP: usize = 4000;

/// Deterministic xorshift — the bootstrap must be reproducible run to run.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn median_of(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Percentile bootstrap CI for the median of `xs`.
fn median_ci(xs: &[f64]) -> (f64, f64) {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut meds = Vec::with_capacity(BOOTSTRAP);
    let mut buf = vec![0.0; xs.len()];
    for _ in 0..BOOTSTRAP {
        for slot in buf.iter_mut() {
            *slot = xs[rng.below(xs.len())];
        }
        meds.push(median_of(&mut buf));
    }
    meds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (
        meds[(BOOTSTRAP as f64 * 0.025) as usize],
        meds[(BOOTSTRAP as f64 * 0.975) as usize],
    )
}

fn timed(inner: usize, mut f: impl FnMut()) -> Duration {
    let t = Instant::now();
    for _ in 0..inner {
        f();
    }
    t.elapsed() / inner as u32
}

fn physical_index() -> String {
    std::env::var("CUDA_VISIBLE_DEVICES")
        .ok()
        .and_then(|v| v.split(',').next().map(str::to_owned))
        .unwrap_or_else(|| "0".into())
}

/// Device conditions worth knowing about. These no longer invalidate the
/// result — pairing handles contention — but they explain the absolute numbers.
fn conditions(idx: &str) -> String {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "-i",
            idx,
            "--query-gpu=utilization.gpu,pstate,clocks.sm,clocks.max.sm,\
             pcie.link.gen.current,pcie.link.width.current,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(out) = out else {
        return "unknown (nvidia-smi unavailable)".into();
    };
    let line = String::from_utf8_lossy(&out.stdout);
    let f: Vec<&str> = line.trim().split(',').map(str::trim).collect();
    if f.len() < 7 {
        return "unknown".into();
    }
    format!(
        "{}% busy, {}, {} of {} MHz, PCIe gen{} x{}, {} MiB resident",
        f[0], f[1], f[2], f[3], f[4], f[5], f[6]
    )
}

/// One measured comparison: `pairs` back-to-back (IA, cudarc) samples.
fn compare(
    pairs: usize,
    inner: usize,
    mut ia_op: impl FnMut(),
    mut cu_op: impl FnMut(),
) -> (Duration, Duration, Vec<f64>) {
    // Warm both sides so first-touch costs land outside the samples.
    for _ in 0..3 {
        ia_op();
        cu_op();
    }
    let mut ia_s = Vec::with_capacity(pairs);
    let mut cu_s = Vec::with_capacity(pairs);
    let mut ratios = Vec::with_capacity(pairs);
    for k in 0..pairs {
        // Alternate which side goes first: whichever runs second can inherit
        // cache or power state from the first, and that bias must not land on
        // one implementation.
        let (a, b) = if k % 2 == 0 {
            let a = timed(inner, &mut ia_op);
            let b = timed(inner, &mut cu_op);
            (a, b)
        } else {
            let b = timed(inner, &mut cu_op);
            let a = timed(inner, &mut ia_op);
            (a, b)
        };
        ia_s.push(a);
        cu_s.push(b);
        ratios.push(b.as_secs_f64() / a.as_secs_f64().max(f64::MIN_POSITIVE));
    }
    let mut ia_f: Vec<f64> = ia_s.iter().map(|d| d.as_secs_f64()).collect();
    let mut cu_f: Vec<f64> = cu_s.iter().map(|d| d.as_secs_f64()).collect();
    (
        Duration::from_secs_f64(median_of(&mut ia_f)),
        Duration::from_secs_f64(median_of(&mut cu_f)),
        ratios,
    )
}

fn report(op: &str, size: &str, ia: Duration, cu: Duration, ratios: &mut [f64]) -> bool {
    let wins = ratios.iter().filter(|r| **r > 1.0).count();
    let (lo, hi) = median_ci(ratios);
    let med = median_of(&mut ratios.to_vec());
    // Definitive only when the whole interval sits on one side of parity.
    let verdict = if lo > 1.0 {
        "IA faster"
    } else if hi < 1.0 {
        "IA SLOWER"
    } else {
        "inconclusive"
    };
    println!(
        "{:<5} {:>6} {:>11?} {:>11?} {:>7.2}x  [{:>5.2},{:>5.2}] {:>5.0}%  {}",
        op,
        size,
        ia,
        cu,
        med,
        lo,
        hi,
        100.0 * wins as f64 / ratios.len() as f64,
        verdict
    );
    lo > 1.0
}

fn main() {
    let Ok(iron_dev) = iron::CudaDevice::new(0) else {
        eprintln!("no CUDA device");
        return;
    };
    let iron_stream = iron_dev.default_stream();
    let Ok(cudarc_ctx) = cudarc::driver::CudaContext::new(0) else {
        eprintln!("cudarc init failed");
        return;
    };
    let cudarc_stream = cudarc_ctx.default_stream();

    let idx = physical_index();
    println!(
        "device : {} (physical {idx})\nstate  : {}\nmethod : paired back-to-back samples, \
         median per-pair ratio, bootstrap 95% CI\n",
        iron_dev.name().unwrap_or_else(|_| "?".into()),
        conditions(&idx)
    );
    println!(
        "{:<5} {:>6} {:>11} {:>11} {:>8}  {:>13} {:>5}  verdict",
        "op", "size", "ironaccel", "cudarc", "ratio", "95% CI", "win"
    );
    println!("{}", "-".repeat(84));

    // (bytes, label, inner reps per sample, pairs). Swept from 256 B to 64 MiB
    // so the table shows the whole transfer curve — the tiny end where
    // per-call wrapper overhead dominates, the mid-range PCIe-bound regime, and
    // the large end where the pinned-staging path pulls ahead.
    let sizes: &[(usize, &str, usize, usize)] = &[
        (256, "256B", 200, 151),
        (1 << 10, "1KB", 100, 151),
        (64 << 10, "64KB", 50, 151),
        (256 << 10, "256KB", 30, 151),
        (1 << 20, "1MB", 10, 151),
        (4 << 20, "4MB", 5, 121),
        (16 << 20, "16MB", 2, 101),
        (64 << 20, "64MB", 1, 61),
    ];

    let mut all_definitive = true;
    for &(n, label, inner, pairs) in sizes {
        let host = vec![0u8; n];

        let (ia, cu, mut r) = compare(
            pairs,
            inner,
            || {
                let b = iron_stream.htod_sync_copy(&host).unwrap();
                std::hint::black_box(&b);
            },
            || {
                let b = cudarc_stream.clone_htod(&host).unwrap();
                cudarc_stream.synchronize().unwrap();
                std::hint::black_box(&b);
            },
        );
        all_definitive &= report("h2d", label, ia, cu, &mut r);

        let ia_dev = iron_stream.htod_sync_copy(&host).unwrap();
        let cu_dev = cudarc_stream.clone_htod(&host).unwrap();
        cudarc_stream.synchronize().unwrap();
        let (ia, cu, mut r) = compare(
            pairs,
            inner,
            || {
                let v: Vec<u8> = iron_stream.dtoh_sync_copy(&ia_dev).unwrap();
                std::hint::black_box(v);
            },
            || {
                let v = cudarc_stream.clone_dtoh(&cu_dev).unwrap();
                std::hint::black_box(v);
            },
        );
        all_definitive &= report("d2h", label, ia, cu, &mut r);

        // Read into a caller-owned buffer. This is what a serving loop actually
        // does — output buffers are reused, not allocated per token — so the
        // per-call `Vec` allocation stops being a constant shared by both sides.
        let mut ia_dst = vec![0u8; n];
        let mut cu_dst = vec![0u8; n];
        let (ia, cu, mut r) = compare(
            pairs,
            inner,
            || {
                iron_stream
                    .dtoh_sync_copy_into(&ia_dev, &mut ia_dst)
                    .unwrap();
                std::hint::black_box(ia_dst[0]);
            },
            || {
                cudarc_stream.memcpy_dtoh(&cu_dev, &mut cu_dst).unwrap();
                std::hint::black_box(cu_dst[0]);
            },
        );
        all_definitive &= report("d2h→buf", label, ia, cu, &mut r);
    }

    println!(
        "\n{}",
        if all_definitive {
            "RESULT: IronAccelerator is faster on every row, CI-confirmed."
        } else {
            "RESULT: not yet faster on every row — see rows above."
        }
    );
}
