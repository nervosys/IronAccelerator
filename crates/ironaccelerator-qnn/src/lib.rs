//! # `ironaccelerator-qnn`
//!
//! Qualcomm AI Engine backend, dispatching through the **QNN SDK**:
//!
//! - **HTP** (Hexagon Tensor Processor) — primary NPU path; INT8 / FP16
//!   with HMX matrix-engine acceleration on Hexagon v73+.
//! - **HVX** — fallback Hexagon vector path for ops not yet supported by HMX.
//! - **Adreno GPU** — alternative target for FP16 graphs.
//!
//! IronAccelerator builds QNN graphs from the same [`Workload`](ironaccelerator_core::Workload)
//! description used by the CUDA/ROCm planners. Quantisation calibration is
//! handled lazily — the first execution with `Phase::Calibration` produces a
//! per-tensor scale table cached on disk.
//!
//! ## Status
//!
//! v0.1 scaffold. Library probe via libloading; FFI bindings for `QnnApi.h`,
//! `QnnContext.h`, `QnnGraph.h`, `QnnHtp.h` are tracked next.

#![allow(clippy::missing_safety_doc)]

pub mod backend;

pub use backend::{QnnBackend, QNN_BACKEND};

pub fn register(reg: &mut ironaccelerator_core::BackendRegistry) {
    let b: &'static backend::QnnBackend = &*QNN_BACKEND;
    reg.register(b);
}
