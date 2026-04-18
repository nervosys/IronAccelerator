//! Process-wide runtime: registry of available backends + a planner that
//! routes a [`Workload`] to a `(backend, device, strategy)` tuple.

use ironaccelerator_core::{
    Backend, BackendKind, BackendRegistry, DeviceDescriptor, Error, Result, Strategy,
    StrategyHint, Workload,
};

pub struct Runtime {
    registry: BackendRegistry,
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub backend: BackendKind,
    pub device: u32,
    pub strategy: Strategy,
    pub score: f32,
}

impl Runtime {
    pub fn new() -> Self {
        let mut registry = BackendRegistry::new();

        #[cfg(feature = "cuda")]
        ironaccelerator_cuda::register(&mut registry);
        #[cfg(feature = "rocm")]
        ironaccelerator_rocm::register(&mut registry);
        #[cfg(feature = "metal")]
        ironaccelerator_metal::register(&mut registry);
        #[cfg(feature = "qnn")]
        ironaccelerator_qnn::register(&mut registry);

        Self { registry }
    }

    pub fn registry(&self) -> &BackendRegistry { &self.registry }

    /// Enumerate every visible device across every available backend.
    pub fn devices(&self) -> Vec<DeviceDescriptor> {
        self.registry
            .available()
            .flat_map(|b| b.enumerate().unwrap_or_default())
            .collect()
    }

    /// Plan a workload with no caller hints — picks the highest-scoring
    /// (backend, device) pair.
    pub fn plan(&self, workload: &Workload) -> Result<Plan> {
        self.plan_with(workload, &StrategyHint::default())
    }

    pub fn plan_with(&self, workload: &Workload, hint: &StrategyHint) -> Result<Plan> {
        let mut best: Option<Plan> = None;

        let backends: Vec<&'static dyn Backend> = if hint.prefer_backends.is_empty() {
            self.registry.available().collect()
        } else {
            hint.prefer_backends.iter()
                .filter_map(|k| self.registry.get(*k).filter(|b| b.is_available()))
                .collect()
        };

        for b in backends {
            let devs = b.enumerate().unwrap_or_default();
            for d in devs {
                let score = b.score(d.id.ordinal, workload);
                if score <= 0.0 { continue; }
                let strategy = b.plan(d.id.ordinal, workload)?;
                let candidate = Plan {
                    backend: b.kind(),
                    device: d.id.ordinal,
                    strategy,
                    score,
                };
                if best.as_ref().map_or(true, |b| candidate.score > b.score) {
                    best = Some(candidate);
                }
            }
        }

        best.ok_or(Error::BackendUnavailable("no backend produced a plan"))
    }
}

impl Default for Runtime {
    fn default() -> Self { Self::new() }
}

/// Convenience: build a runtime with every compiled-in backend registered.
pub fn init() -> Runtime { Runtime::new() }
