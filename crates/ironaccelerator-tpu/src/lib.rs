//! # `ironaccelerator-tpu`
//!
//! Google TPU backend via the **PJRT** (Pretty Just Runtime) C plugin
//! interface. PJRT is the stable cross-framework TPU entry point — the
//! same ABI JAX and PyTorch/XLA use. The backend dynamically loads
//! `pjrt_c_api_tpu_plugin.so` (or `libtpu.so` on older images), calls
//! `GetPjrtApi`, creates a `PJRT_Client`, and enumerates the attached
//! TPU chips as devices.
//!
//! Enumeration is all we need at the backend-trait level; compilation and
//! execution sit on top of a `StableHLO` graph that the higher layers
//! build and hand back through PJRT. That machinery lives in a follow-on
//! crate and is not part of 1.1.
//!
//! Hosts without the PJRT plugin — i.e. every machine that is not a
//! Cloud TPU VM or a `libtpu`-provisioned GKE pod — get an unavailable
//! backend and an empty device list.

#![allow(clippy::missing_safety_doc)]

pub mod backend;
pub mod drv;

pub use backend::{TpuBackend, TPU_BACKEND};

pub fn register(reg: &mut ironaccelerator_core::BackendRegistry) {
    reg.register(&TPU_BACKEND);
}
