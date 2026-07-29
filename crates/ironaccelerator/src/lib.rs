//! # IronAccelerator
//!
//! Low-level, hardware-agnostic Rust interface over CUDA, ROCm, Metal, and
//! other accelerators. This facade crate re-exports each backend behind a
//! feature flag and exposes a `Runtime` that surveys the backends and devices
//! reachable from this process. Most CUDA users will pull in
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
//! | `ironaccelerator-vulkan`    | Vulkan compute: enumerate, buffers, SPIR-V pipeline, dispatch |
//! | `ironaccelerator-dx12`      | Direct3D 12 compute: enumerate, buffers, DXIL pipeline, dispatch |
//! | `ironaccelerator-opengl`    | OpenGL 4.3+ compute-shader fallback (host-bound GL context) |
//! | `ironaccelerator-webgpu`    | browser/WASM compute path, host-bound                      |
//!
//! ## Native vs browser
//!
//! On native hosts the GPU backends talk to the platform driver directly:
//! CUDA and ROCm for the vendor stacks, Vulkan for cross-vendor compute,
//! Metal on Apple, D3D12 on Windows, OpenGL as the legacy fallback.
//! `ironaccelerator-webgpu` is the browser path only — on native it reaches
//! nothing the others do not, so it does not try.
//!
//! ## Scope
//!
//! IronAccelerator stops at the driver. Workload descriptors, execution-strategy
//! selection, tensor descriptors, quantization schemes, and the accelerator
//! ontology used to rank kernel strategies live in the inference engine on top
//! ([IronWorks](https://github.com/nervosys/ironworks)) — not here.
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
//! ## Cross-backend device survey (this facade)
//!
//! ```no_run
//! use ironaccelerator::prelude::*;
//!
//! let runtime = ironaccelerator::init();
//! for dev in runtime.devices_with(CapabilityFlags::FP8_E4M3) {
//!     println!("{} ({}) — {:?}", dev.name, dev.arch, dev.id.backend);
//! }
//! ```
//!
//! Note: the survey covers backends that implement
//! [`ironaccelerator_core::Backend`]. The CUDA crate intentionally does **not**
//! — it ships only driver wrappers. Use the CUDA crate directly when you want
//! fine-grained driver control or the cudarc-shaped API.

pub use ironaccelerator_core as core;

#[cfg(feature = "cuda")]
pub use ironaccelerator_cuda as cuda;
#[cfg(feature = "dx12")]
pub use ironaccelerator_dx12 as dx12;
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
        DeviceDescriptor, DeviceId, Error, LaunchDims, MemoryKind, Result, Vendor,
    };
}

mod runtime;
pub use runtime::{init, Runtime};
