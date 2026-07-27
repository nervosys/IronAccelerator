//! # `ironaccelerator-qnn`
//!
//! Qualcomm AI Engine backend, dispatching through the **QNN SDK**:
//!
//! - **HTP** (Hexagon Tensor Processor) — primary NPU path; INT8 / FP16
//!   with HMX matrix-engine acceleration on Hexagon v73+.
//! - **HVX** — fallback Hexagon vector path for ops not yet supported by HMX.
//! - **Adreno GPU** — alternative target for FP16 graphs.
//!
//! Scope is the same as every other backend here: probe the runtime, enumerate
//! targets, report capability bits. QNN graphs are built ahead of time by the
//! SDK, so graph construction and quantisation calibration belong to the
//! consumer, not to this crate.
//!
//! ## Status
//!
//! v0.1 scaffold. Library probe via libloading; FFI bindings for `QnnApi.h`,
//! `QnnContext.h`, `QnnGraph.h`, `QnnHtp.h` are tracked next.

#![allow(clippy::missing_safety_doc)]

pub mod backend;
pub mod drv;

pub use backend::{QnnBackend, QNN_BACKEND};
pub use iron_qnn_sys as sys;

pub fn register(reg: &mut ironaccelerator_core::BackendRegistry) {
    let b: &'static backend::QnnBackend = &QNN_BACKEND;
    reg.register(b);
}
