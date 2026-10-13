//! # `ironaccelerator-rocm`
//!
//! ROCm / HIP backend. Targets the ROCm 6.x API surface and the AMD
//! datacenter accelerator stack:
//!
//! - **HIP** driver + runtime (analogous to CUDA driver/runtime)
//! - **hipBLAS / hipBLASLt** — dense matmul, FP8 on CDNA3+
//! - **MIOpen** — convolution + fused kernels (cuDNN analogue)
//! - **rocFFT, rocRAND, rocSPARSE, rocSOLVER**
//! - **RCCL** — collective comms (NCCL analogue)
//! - **Composable Kernel** — tunable templates (CUTLASS analogue)
//!
//! ## Status
//!
//! Safe HIP driver wrappers (`drv`), HIPRTC kernel compilation with a
//! race-free in-memory module cache (`kernel`), a size-class memory pool
//! (`pool`), and a hipBLASLt matmul surface (`blas`). Device paths compile
//! but are not yet validated on AMD hardware.

#![allow(
    clippy::missing_safety_doc,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

pub mod backend;
pub mod blas;
pub mod drv;
pub mod kernel;
pub mod pool;

pub use backend::{RocmBackend, ROCM_BACKEND};

pub fn register(reg: &mut ironaccelerator_core::BackendRegistry) {
    // Force the Lazy and hand the registry the inner `'static` reference.
    let b: &'static backend::RocmBackend = &ROCM_BACKEND;
    reg.register(b);
}
