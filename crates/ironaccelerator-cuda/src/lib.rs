//! # `ironaccelerator-cuda`
//!
//! Low-level, hardware-agnostic CUDA interface. Targets the **CUDA Toolkit
//! 13.2** API surface via [`iron_cuda_sys`], with safe wrappers around the
//! driver, NVRTC, and every major vendor library (cuBLAS / cuBLASLt / cuDNN
//! / cuRAND / cuSPARSE / cuSOLVER / cuFFT / cuTENSOR / NCCL / NVTX / CUPTI).
//!
//! **Scope.** This crate is a *driver substrate*. It does **not** ship
//! kernels, workload planners, FP8 recipes, attention/MoE implementations,
//! or autotuners — those belong to downstream libraries layered on top.
//! What's here:
//!
//! - [`drv`] — Device / Stream / Event / DeviceBuf / Module / Function
//! - [`kernel`] — NVRTC compile + in-memory and on-disk PTX cache
//! - [`blas`], [`cudnn`], [`fft`], [`cusparse`], [`cusolver`], [`cutensor`],
//!   [`rng`], [`nccl`] — per-library handle plumbing
//! - [`advanced`] — VMM, green contexts, multicast teams, conditional graph nodes
//! - [`graph`], [`launch`], [`peer`], [`pinned`], [`streams`],
//!   [`events`], [`alloc`] — driver-level primitives
//! - [`observe`], [`profile`], [`cupti`] — NVTX / profiler hooks
//! - [`cudarc_compat`] — drop-in compatibility surface for cudarc users
//! - [`sys`] — the raw FFI re-export for callers that need it
//!
//! ## If you want to…
//!
//! | Task                                       | Use                                                          |
//! |--------------------------------------------|--------------------------------------------------------------|
//! | port existing cudarc 0.19 code             | [`cudarc_compat`] (drop-in `use` swap)                       |
//! | open a device + create a stream            | [`drv::Device::open`] then [`drv::Stream::new`]              |
//! | allocate a typed device buffer             | [`drv::DeviceBuf::alloc`] / [`alloc::alloc`]                 |
//! | copy host ↔ device on a stream             | [`drv::DeviceBuf::copy_from_host`] / `copy_to_host`          |
//! | compile a CUDA C++ kernel at runtime       | [`kernel::get_or_compile`] (cached on disk per arch+source)  |
//! | launch a kernel with typed args            | [`drv::Function::launch`] or [`launch::launch_1d`] / `_2d`   |
//! | record + measure GPU time                  | [`events::Timer`] (or [`drv::TimingEvent`] directly)         |
//! | capture & replay a stream as a CUDA graph  | [`drv::Stream::begin_capture`] → [`drv::GraphExec`]          |
//! | call cuBLASLt / cuDNN / cuFFT / NCCL       | the matching `blas`/`cudnn`/`fft`/`nccl` module              |
//! | drop down to raw FFI                       | [`sys`] (re-exports [`iron_cuda_sys`])                       |
//!
//! ## Performance posture
//!
//! Every hot path is `#[inline]`. The driver `fns()` lookup uses an
//! `AtomicPtr<DriverFns>` fast-path cache, and each `Device`/`Stream`/`Event`/
//! `Module`/`Function` caches `&'static DriverFns` at construction — so a
//! wrapped op costs one struct-field load + one FFI entry. See the workspace
//! README for benchmarks against `cudarc` 0.19; alloc/free is **~2× faster**.

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

pub mod advanced;
pub mod alloc;
pub mod blas;
pub mod cudarc_compat;
pub mod cudnn;
pub mod cupti;
pub mod cusolver;
pub mod cusparse;
pub mod cutensor;
pub mod events;
pub mod fft;
pub mod graph;
pub mod kernel;
pub mod launch;
pub mod nccl;
pub mod observe;
pub mod peer;
pub mod pinned;
pub mod profile;
pub mod rng;
pub mod safe;
pub mod streams;
