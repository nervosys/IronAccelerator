//! # `ironaccelerator-metal`
//!
//! Apple Metal backend. Will dispatch through:
//!
//! - **Metal Performance Shaders (MPS)** — `MPSMatrixMultiplication`, etc.
//! - **MPSGraph** — JIT-compiled compute graphs, the "official" fast path.
//! - **MLX-style kernels** — fused attention / RMSNorm / sampling kernels
//!   ported from MLX (Apple's ML framework).
//! - **CoreML** — opportunistic offload to the Apple Neural Engine for
//!   subgraphs whose ops are ANE-supported.
//!
//! ## Status
//!
//! v0.1 scaffold: backend trait wired up. Apple-only FFI bindings (objc2 /
//! metal-rs) gated behind `cfg(target_vendor = "apple")` will land next.

#![allow(clippy::missing_safety_doc)]

pub mod backend;

#[cfg(target_vendor = "apple")]
pub mod blas;
#[cfg(target_vendor = "apple")]
pub mod drv;

pub use backend::{MetalBackend, METAL_BACKEND};

pub fn register(reg: &mut ironaccelerator_core::BackendRegistry) {
    reg.register(&METAL_BACKEND);
}
