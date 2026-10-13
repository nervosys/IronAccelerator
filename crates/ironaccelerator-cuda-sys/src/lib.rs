//! # `ironaccelerator-cuda-sys`
//!
//! Clean-room, hand-written FFI for the **CUDA Toolkit 13.2** library surface
//! used by IronAccelerator. This crate is a drop-in replacement for
//! `cudarc`'s sys layer — it depends on nothing from that project.
//!
//! ## Design
//!
//! - **Fully dynamic loading.** No `extern "C"` link-time dependency on any
//!   CUDA library — everything is resolved lazily via [`libloading`] at
//!   first use. Consequence: the crate compiles on machines without CUDA
//!   installed, tests run, and CI stays green.
//!
//! - **One `Fns` struct per library.** Each loaded library exposes its
//!   function pointers as fields on a struct; a `LazyLock<Result<&Fns, _>>`
//!   amortises the `dlopen`+`dlsym` cost. Call sites look like
//!   `driver::fns()?.cuLaunchKernel(…)`.
//!
//! - **No opaque-pointer guesswork.** Handle types are `#[repr(transparent)]`
//!   newtypes over `*mut ()` with matching `Send`/`Sync` and `Default`
//!   impls. Status codes and other values the library *writes* (`CUresult`,
//!   `CublasStatus`, …) are `#[repr(transparent)]` `u32` newtypes with the
//!   known codes as associated constants, so any value a library returns is
//!   representable; enums are used only for values Rust passes *in*.
//!
//! - **Only what we use.** Every function, every type in this crate is
//!   exercised by the safe layer. No dead FFI.
//!
//! ## Modules
//!
//! | Module        | Library            | Purpose                                 |
//! |---------------|--------------------|-----------------------------------------|
//! | [`driver`]    | `libcuda` / `nvcuda` | Core driver API (context, stream, module, kernel, memcpy, graph, event, peer access) |
//! | [`nvrtc`]     | `libnvrtc`         | Runtime compilation of CUDA C++ → PTX   |
//! | [`cublas_lt`] | `libcublasLt`      | Matmul descriptor-based API (FP8 GEMM)  |
//! | [`cublas`]    | `libcublas`        | Legacy BLAS (fallback)                  |
//! | [`cudnn`]     | `libcudnn`         | Deep-learning primitives (MHA, conv)    |
//! | [`curand`]    | `libcurand`        | RNG                                     |
//! | [`cufft`]     | `libcufft`         | FFTs                                    |
//! | [`cusparse`]  | `libcusparse`      | Sparse BLAS (SpMM, SDDMM)               |
//! | [`cusolver`]  | `libcusolverDn`    | Dense linear solvers                    |
//! | [`cutensor`]  | `libcutensor`      | N-dim tensor contraction                |
//! | [`nccl`]      | `libnccl`          | Collectives                             |
//! | [`nvtx`]      | `libnvToolsExt`    | Profiler ranges/marks                   |
//! | [`cupti`]     | `libcupti`         | Profiling/tracing activity records      |
//!
//! ## Documentation policy
//!
//! Items here mirror the CUDA C headers one-to-one: a `Fns` field is named
//! exactly after the C symbol it resolves, and enum variants and struct
//! fields carry the C names minus their prefix. They are deliberately not
//! documented item by item — NVIDIA's reference for the same symbol is the
//! authority, and a paraphrase here could only drift from it. The safe crate
//! (`ironaccelerator-cuda`) is fully documented and enforces `missing_docs`.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]
#![allow(
    clippy::missing_safety_doc,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::missing_transmute_annotations
)]

#[macro_use]
mod code;

pub mod cublas;
pub mod cublas_lt;
pub mod cudnn;
pub mod cufft;
pub mod cupti;
pub mod curand;
pub mod cusolver;
pub mod cusparse;
pub mod cutensor;
pub mod driver;
pub mod loader;
pub mod nccl;
pub mod nvrtc;
pub mod nvtx;

pub use loader::{LoadError, LoaderResult};

/// CUDA Toolkit version this binding targets. Packed `MMmm` (13.20 = 1320).
pub const CUDA_VERSION: u32 = 13_20;
