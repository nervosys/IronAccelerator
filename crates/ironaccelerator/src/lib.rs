//! # IronAccelerator
//!
//! Low-level, hardware-agnostic Rust interface over CUDA, ROCm, Metal, and
//! other accelerators. This facade crate re-exports each backend behind a
//! feature flag and exposes a `Runtime` that surveys available backends to
//! dispatch a `Workload`. Most CUDA users will pull in
//! [`ironaccelerator-cuda`](::ironaccelerator_cuda) directly — it doubles as a
//! [drop-in replacement for cudarc](::ironaccelerator_cuda::cudarc_compat)
//! that's measurably faster on every host-side hot path.
//!
//! ## Crate layout
//!
//! | crate                       | purpose                                                    |
//! |-----------------------------|------------------------------------------------------------|
//! | `ironaccelerator-core`      | shared types, errors, capability flags                     |
//! | `ironaccelerator-cuda-sys`  | clean-room CUDA 13.2 FFI + dynamic loader (no link deps)   |
//! | `ironaccelerator-cuda`      | safe CUDA driver wrappers + cudarc-shaped compatibility    |
//! | `ironaccelerator-rocm`      | safe ROCm/HIP driver wrappers (same fast-path pattern)     |
//! | `ironaccelerator-metal`     | Apple Metal / MPS scaffold                                 |
//! | `ironaccelerator-qnn`       | Qualcomm Hexagon NPU scaffold                              |
//! | `ironaccelerator-vulkan` …  | cross-vendor + niche backends                              |
//!
//! ## Direct CUDA usage (recommended for CUDA-only consumers)
//!
//! ```no_run
//! use ironaccelerator_cuda::cudarc_compat::*;
//!
//! let dev = CudaDevice::new(0)?;
//! let stream = dev.default_stream();
//! let xs = stream.htod_copy(vec![1.0f32, 2.0, 3.0])?;
//! let out = stream.dtoh_sync_copy(&xs)?;
//! # Ok::<(), DriverError>(())
//! ```
//!
//! ## Cross-backend dispatch (this facade)
//!
//! ```no_run
//! use ironaccelerator::prelude::*;
//!
//! let runtime = ironaccelerator::init();
//! let workload = Workload::gemm(8192, 8192, 8192, DType::F8E4M3);
//! let plan = runtime.plan(&workload)?;
//! println!("{plan:?}");
//! # Ok::<(), Error>(())
//! ```
//!
//! Note: the facade's `plan` covers backends that implement
//! [`ironaccelerator_core::Backend`]. The CUDA crate intentionally does **not**
//! — it ships only driver wrappers, not workload planners. Use the CUDA crate
//! directly when you want fine-grained driver control or the cudarc-shaped
//! API.

pub use ironaccelerator_core as core;
#[cfg(feature = "ontology")]
pub use ironaccelerator_ontology as ontology;

#[cfg(feature = "cuda")]
pub use ironaccelerator_cuda as cuda;
#[cfg(feature = "levelzero")]
pub use ironaccelerator_levelzero as levelzero;
#[cfg(feature = "metal")]
pub use ironaccelerator_metal as metal;
#[cfg(feature = "neuron")]
pub use ironaccelerator_neuron as neuron;
#[cfg(feature = "opengl")]
pub use ironaccelerator_opengl as opengl;
#[cfg(feature = "qnn")]
pub use ironaccelerator_qnn as qnn;
#[cfg(feature = "rocm")]
pub use ironaccelerator_rocm as rocm;
#[cfg(feature = "tpu")]
pub use ironaccelerator_tpu as tpu;
#[cfg(feature = "vulkan")]
pub use ironaccelerator_vulkan as vulkan;
#[cfg(feature = "webgpu")]
pub use ironaccelerator_webgpu as webgpu;

pub mod prelude {
    pub use ironaccelerator_core::{
        Backend, BackendKind, Capability, CapabilityFlags, ComputeTier, DType, Device,
        DeviceDescriptor, DeviceId, Error, Result, Strategy, StrategyHint, Vendor, Workload,
        WorkloadKind, WorkloadShape,
    };
}

mod runtime;
pub use runtime::{init, Runtime};
