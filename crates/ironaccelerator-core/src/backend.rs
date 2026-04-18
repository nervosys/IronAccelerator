//! [`Backend`] is the single trait every accelerator backend implements.
//!
//! The trait is intentionally small and object-safe so the runtime can hold a
//! `&'static dyn Backend` table and dispatch without monomorphisation overhead
//! for the rare cross-backend code paths. Hot paths (kernel launches, memcpys)
//! live on the concrete backend type and are inlined.

use crate::{
    capability::CapabilityFlags, device::DeviceDescriptor, error::Result, strategy::Strategy,
    workload::Workload,
};

#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// The set of accelerator families IronAccelerator can target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum BackendKind {
    /// NVIDIA CUDA (Toolkit 13.2+ targeted).
    Cuda,
    /// AMD ROCm / HIP.
    Rocm,
    /// Apple Metal (Performance Shaders + MLX kernels).
    Metal,
    /// Qualcomm AI Engine — Hexagon NPU via QNN SDK.
    QualcommNpu,
    /// CPU SIMD reference path (used as a fallback / oracle).
    Cpu,
}

impl BackendKind {
    pub const ALL: &'static [BackendKind] = &[
        BackendKind::Cuda,
        BackendKind::Rocm,
        BackendKind::Metal,
        BackendKind::QualcommNpu,
        BackendKind::Cpu,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            BackendKind::Cuda => "cuda",
            BackendKind::Rocm => "rocm",
            BackendKind::Metal => "metal",
            BackendKind::QualcommNpu => "qnn",
            BackendKind::Cpu => "cpu",
        }
    }
}

/// The minimum capability surface a backend must expose.
///
/// Object-safe; concrete backends typically also expose a non-trait API with
/// inlined fast paths.
pub trait Backend: Send + Sync + 'static {
    /// Static identity.
    fn kind(&self) -> BackendKind;

    /// Whether the backend's runtime libraries were located on this host.
    fn is_available(&self) -> bool;

    /// Enumerate every visible device.
    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>>;

    /// Coarse capability bits used by [`Strategy`] selection.
    fn capabilities(&self, device: u32) -> Result<CapabilityFlags>;

    /// Score a workload on the given device. Higher is better. The default
    /// returns `0.0`; backends override with vendor-tuned heuristics.
    fn score(&self, _device: u32, _workload: &Workload) -> f32 {
        0.0
    }

    /// Pick the best execution strategy for a workload on a device.
    fn plan(&self, device: u32, workload: &Workload) -> Result<Strategy>;
}

/// Process-wide registry of compiled-in backends. Backends register themselves
/// from their crate's `init()` (called from [`ironaccelerator::init`]).
pub struct BackendRegistry {
    entries: Vec<&'static dyn Backend>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn register(&mut self, backend: &'static dyn Backend) {
        if !self.entries.iter().any(|b| b.kind() == backend.kind()) {
            self.entries.push(backend);
        }
    }

    pub fn get(&self, kind: BackendKind) -> Option<&'static dyn Backend> {
        self.entries.iter().copied().find(|b| b.kind() == kind)
    }

    pub fn iter(&self) -> impl Iterator<Item = &'static dyn Backend> + '_ {
        self.entries.iter().copied()
    }

    pub fn available(&self) -> impl Iterator<Item = &'static dyn Backend> + '_ {
        self.iter().filter(|b| b.is_available())
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}
