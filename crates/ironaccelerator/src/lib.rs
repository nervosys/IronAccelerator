//! # IronAccelerator
//!
//! A high-performance, **agentic-first** Rust acceleration library spanning
//! NVIDIA CUDA, AMD ROCm, Apple Metal, and Qualcomm Hexagon NPUs.
//!
//! IronAccelerator's premise is that the right kernel for a given workload is
//! a function of *(workload class, hardware capability, system constraints)*.
//! Rather than picking that function by hand, we let an **agent** — be it an
//! LLM tool-use loop, a build-time script, or a runtime auto-tuner — query
//! the [`ontology`] graph and dispatch through the matching backend.
//!
//! ## Quickstart
//!
//! ```no_run
//! use ironaccelerator::prelude::*;
//!
//! let mut runtime = ironaccelerator::init();
//! let workload = Workload::gemm(8192, 8192, 8192, DType::F8E4M3);
//!
//! // Let the planner choose backend + strategy.
//! let plan = runtime.plan(&workload).unwrap();
//! println!("{plan:?}");
//! ```
//!
//! ## Crate layout
//!
//! | crate                       | purpose                                    |
//! |-----------------------------|--------------------------------------------|
//! | `ironaccelerator-core`      | traits, types, capability flags            |
//! | `ironaccelerator-ontology`  | machine-readable knowledge graph for agents|
//! | `ironaccelerator-cuda`      | CUDA 13.2 backend (atop `cudarc`)          |
//! | `ironaccelerator-rocm`      | ROCm/HIP backend                           |
//! | `ironaccelerator-metal`     | Apple Metal / MPS / MLX backend            |
//! | `ironaccelerator-qnn`       | Qualcomm Hexagon NPU backend               |
//!
//! ## Performance posture
//!
//! IronAccelerator prioritises **throughput over guard-rails**. Hot-path
//! launches are `#[inline(always)]`, allocations are stream-ordered, and
//! every safe call has a paired `_unchecked` sibling. See `docs/perf.md`.

pub use ironaccelerator_core as core;
#[cfg(feature = "ontology")]
pub use ironaccelerator_ontology as ontology;

#[cfg(feature = "cuda")]
pub use ironaccelerator_cuda as cuda;
#[cfg(feature = "rocm")]
pub use ironaccelerator_rocm as rocm;
#[cfg(feature = "metal")]
pub use ironaccelerator_metal as metal;
#[cfg(feature = "qnn")]
pub use ironaccelerator_qnn as qnn;
#[cfg(feature = "vulkan")]
pub use ironaccelerator_vulkan as vulkan;
#[cfg(feature = "opengl")]
pub use ironaccelerator_opengl as opengl;
#[cfg(feature = "webgpu")]
pub use ironaccelerator_webgpu as webgpu;
#[cfg(feature = "tpu")]
pub use ironaccelerator_tpu as tpu;
#[cfg(feature = "levelzero")]
pub use ironaccelerator_levelzero as levelzero;
#[cfg(feature = "neuron")]
pub use ironaccelerator_neuron as neuron;

pub mod prelude {
    pub use ironaccelerator_core::{
        Backend, BackendKind, Capability, CapabilityFlags, ComputeTier,
        DType, Device, DeviceDescriptor, DeviceId,
        Error, Result, Strategy, StrategyHint, Vendor,
        Workload, WorkloadKind, WorkloadShape,
    };
}

mod runtime;
pub use runtime::{init, Runtime};
