//! Process-wide runtime: a registry of the backends compiled into this build,
//! plus device discovery across all of them.
//!
//! This is a *survey*, not a planner. It answers "what hardware can I reach
//! from this process, and what can each part do?" — deciding what to run where
//! is the consumer's job.

use ironaccelerator_core::{
    BackendKind, BackendRegistry, CapabilityFlags, DeviceDescriptor, Result,
};

pub struct Runtime {
    registry: BackendRegistry,
}

impl Runtime {
    pub fn new() -> Self {
        let mut registry = BackendRegistry::new();

        // NOTE: ironaccelerator-cuda is intentionally NOT registered here.
        // The CUDA crate is a low-level hardware-agnostic interface (a cudarc
        // drop-in replacement) and has no `Backend` impl. Use the CUDA crate
        // directly via `ironaccelerator_cuda::drv` / `cudarc_compat`.
        #[cfg(feature = "rocm")]
        ironaccelerator_rocm::register(&mut registry);
        #[cfg(feature = "metal")]
        ironaccelerator_metal::register(&mut registry);
        #[cfg(feature = "qnn")]
        ironaccelerator_qnn::register(&mut registry);
        #[cfg(feature = "vulkan")]
        ironaccelerator_vulkan::register(&mut registry);
        #[cfg(feature = "opengl")]
        ironaccelerator_opengl::register(&mut registry);
        #[cfg(feature = "dx12")]
        ironaccelerator_dx12::register(&mut registry);
        #[cfg(feature = "webgpu")]
        ironaccelerator_webgpu::register(&mut registry);
        #[cfg(feature = "tpu")]
        ironaccelerator_tpu::register(&mut registry);
        #[cfg(feature = "levelzero")]
        ironaccelerator_levelzero::register(&mut registry);
        #[cfg(feature = "neuron")]
        ironaccelerator_neuron::register(&mut registry);

        Self { registry }
    }

    pub fn registry(&self) -> &BackendRegistry {
        &self.registry
    }

    /// Every backend that located its runtime libraries on this host.
    pub fn available_backends(&self) -> Vec<BackendKind> {
        self.registry.available().map(|b| b.kind()).collect()
    }

    /// Enumerate every visible device across every available backend. A
    /// backend that fails to enumerate contributes nothing rather than
    /// masking the rest — iterate [`Runtime::registry`] directly if you need
    /// per-backend error visibility.
    pub fn devices(&self) -> Vec<DeviceDescriptor> {
        self.registry
            .available()
            .flat_map(|b| b.enumerate().unwrap_or_default())
            .collect()
    }

    /// Devices whose capability bits are a superset of `required`. Pure
    /// hardware filtering — no workload knowledge involved.
    pub fn devices_with(&self, required: CapabilityFlags) -> Vec<DeviceDescriptor> {
        self.devices()
            .into_iter()
            .filter(|d| d.capability.flags.contains(required))
            .collect()
    }

    /// Capability bits for one device on one backend, queried live from the
    /// backend rather than read off a cached descriptor.
    pub fn capabilities(
        &self,
        backend: BackendKind,
        device: u32,
    ) -> Option<Result<CapabilityFlags>> {
        self.registry
            .get(backend)
            .filter(|b| b.is_available())
            .map(|b| b.capabilities(device))
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience: build a runtime with every compiled-in backend registered.
pub fn init() -> Runtime {
    Runtime::new()
}
