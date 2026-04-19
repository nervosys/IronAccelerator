//! # `ironaccelerator-cuda`
//!
//! CUDA backend for IronAccelerator. Targets the **CUDA Toolkit 13.2** API
//! surface (`cudarc` feature `cuda-13020`) with full coverage of the major
//! libraries: driver, runtime, NVRTC, cuBLAS / cuBLASLt, cuDNN, cuRAND,
//! cuSPARSE, cuSOLVER, cuFFT, cuTENSOR, NCCL, NVTX, CUPTI.
//!
//! The crate re-exports the raw FFI surface under [`sys`] so callers can
//! drop down to vendor primitives without losing IronAccelerator's planner
//! integration. New code should prefer the [`safe`] module which adds
//! IronAccelerator-specific fast paths:
//!
//! - [`alloc`] — stream-ordered slab allocator with optional pinned-host pool.
//! - [`kernel`] — process-wide NVRTC kernel cache keyed by source hash + arch.
//! - [`blas`] — cuBLASLt FP8 / BF16 / TF32 GEMM helpers with epilogue fusion.
//! - [`launch`] — `#[inline(always)]` launch helpers + `_unchecked` fast path.
//!
//! ## Performance posture
//!
//! - All wrappers are `#[inline(always)]`. Errors map to opaque codes.
//! - Default allocator is `cudaMallocAsync` (driver ≥ 11.2). Older drivers
//!   fall back to a per-stream slab pool.
//! - Every safe call has a paired `_unchecked` sibling.

#![allow(
    clippy::missing_safety_doc,
    clippy::too_many_arguments,    // intrinsic to FFI wrappers (cublasLtMatmul &c.)
    clippy::type_complexity,       // libloading::Symbol<fn(...)> types
    clippy::missing_transmute_annotations,
    clippy::field_reassign_with_default,
    clippy::if_same_then_else,
    clippy::doc_lazy_continuation,
)]

pub use iron_cuda_sys as sys;

pub mod drv;

pub mod alloc;
pub mod attention;
pub mod backend;
pub mod blas;
pub mod events;
pub mod fft;
pub mod fp8;
pub mod fp8_gemm;
pub mod cudnn;
pub mod flash_attention;
pub mod moe;
pub mod cusolver;
pub mod cusparse;
pub mod cutensor;
pub mod cupti;
pub mod graph;
pub mod kernel;
pub mod launch;
pub mod memcpy;
pub mod nccl;
pub mod observe;
pub mod peer;
pub mod pinned;
pub mod profile;
pub mod rng;
pub mod safe;
pub mod session;
pub mod streams;
pub mod tensor;
pub mod tune;

pub use backend::{CudaBackend, CUDA_BACKEND};
pub use session::Session;
pub use tensor::CudaTensor;

/// Register the CUDA backend with the global
/// `ironaccelerator_core::BackendRegistry`. Idempotent.
pub fn register(reg: &mut ironaccelerator_core::BackendRegistry) {
    let b: &'static backend::CudaBackend = &CUDA_BACKEND;
    reg.register(b);
}
