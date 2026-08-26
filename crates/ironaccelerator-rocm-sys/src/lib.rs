//! ROCm 6.x FFI + dynamic loader — mirror of `iron_cuda_sys` for AMD GPUs.
//!
//! Every library is dynamically loaded on first use. Nothing is linked at
//! build time, so the workspace compiles on any machine with or without
//! ROCm installed.
//!
//! Supported libraries:
//!
//! - [`hip`] — HIP runtime (driver + runtime merged, as ROCm ships it)
//! - [`hipblas`] — hipBLAS dense linear algebra (Level 1/2/3 + batched)
//! - [`hipblaslt`] — hipBLASLt matmul front-end (FP8 on CDNA3+)
//! - [`hiprtc`] — HIPRTC runtime kernel compile (the NVRTC analogue)
//! - [`rccl`] — AMD collective comms
//!
//! Everything else (rocFFT, rocSPARSE, rocSOLVER, MIOpen, rocRAND) will
//! be added as needed. Pattern is identical: add a module, implement
//! `load_fns`, stash the result in a `LazyLock<Result<Fns, LoadError>>`.

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(
    clippy::missing_safety_doc,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

pub mod loader;

pub mod hip;
pub mod hipblas;
pub mod hipblaslt;
pub mod hiprtc;
pub mod rccl;
