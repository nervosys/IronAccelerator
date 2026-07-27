//! [`Backend`] is the single trait every accelerator backend implements.
//!
//! The trait is intentionally small and object-safe so the runtime can hold a
//! `&'static dyn Backend` table and dispatch without monomorphisation overhead
//! for the rare cross-backend code paths. Hot paths (kernel launches, memcpys)
//! live on the concrete backend type and are inlined.

use crate::{capability::CapabilityFlags, device::DeviceDescriptor, error::Result};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

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
    /// Vulkan Compute — cross-vendor GPU compute.
    Vulkan,
    /// OpenGL 4.3+ compute shaders — legacy / embedded GPU fallback.
    OpenGl,
    /// Direct3D 12 — the Windows-native GPU API, across all vendors.
    Dx12,
    /// WebGPU in the browser — the WASM compute path.
    WebGpu,
    /// Google TPU (v4 / v5 / v6e) via the PJRT plugin interface.
    Tpu,
    /// Intel GPU + NPU via oneAPI Level Zero (`ze_loader`).
    LevelZero,
    /// AWS Trainium / Inferentia via the Neuron Runtime (`libnrt`).
    Neuron,
}

impl BackendKind {
    pub const ALL: &'static [BackendKind] = &[
        BackendKind::Cuda,
        BackendKind::Rocm,
        BackendKind::Metal,
        BackendKind::QualcommNpu,
        BackendKind::Cpu,
        BackendKind::Vulkan,
        BackendKind::OpenGl,
        BackendKind::Dx12,
        BackendKind::WebGpu,
        BackendKind::Tpu,
        BackendKind::LevelZero,
        BackendKind::Neuron,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            BackendKind::Cuda => "cuda",
            BackendKind::Rocm => "rocm",
            BackendKind::Metal => "metal",
            BackendKind::QualcommNpu => "qnn",
            BackendKind::Cpu => "cpu",
            BackendKind::Vulkan => "vulkan",
            BackendKind::OpenGl => "opengl",
            BackendKind::Dx12 => "dx12",
            BackendKind::WebGpu => "webgpu",
            BackendKind::Tpu => "tpu",
            BackendKind::LevelZero => "level-zero",
            BackendKind::Neuron => "neuron",
        }
    }
}

/// The minimum capability surface a backend must expose.
///
/// Object-safe; concrete backends typically also expose a non-trait API with
/// inlined fast paths.
///
/// The trait is deliberately limited to *discovery* — identity, availability,
/// devices, capability bits. Choosing what to run on a device is a planner
/// concern and lives in the consumer, not here.
pub trait Backend: Send + Sync + 'static {
    /// Static identity.
    fn kind(&self) -> BackendKind;

    /// Whether the backend's runtime libraries were located on this host.
    fn is_available(&self) -> bool;

    /// Enumerate every visible device.
    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>>;

    /// Coarse capability bits for a device, translated from the vendor's
    /// capability table into the common [`CapabilityFlags`] space.
    fn capabilities(&self, device: u32) -> Result<CapabilityFlags>;
}

/// Process-wide registry of compiled-in backends. Backends register themselves
/// from their crate's `init()` (called from `ironaccelerator::init`).
pub struct BackendRegistry {
    entries: Vec<&'static dyn Backend>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
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

    /// Enumerate every device across every *available* backend. Errors
    /// from any one backend are swallowed (we return an empty list for
    /// it) so a single broken backend can't mask the rest — callers
    /// that want error visibility should iterate manually.
    pub fn describe_all(&self) -> Vec<DeviceDescriptor> {
        self.available()
            .flat_map(|b| b.enumerate().unwrap_or_default())
            .collect()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}
