//! # `ironaccelerator-levelzero`
//!
//! Intel accelerator backend via **oneAPI Level Zero**. Level Zero is the
//! lowest-level user-mode Intel compute API — it sits below SYCL /
//! OpenCL / oneDNN and exposes GPUs (Arc, Flex, Ponte Vecchio, Battlemage)
//! and NPUs (Meteor / Arrow / Lunar Lake VPU) through a single device
//! enumeration and command-queue model.
//!
//! We dynamically load `ze_loader` (`libze_loader.so.1` / `ze_loader.dll`),
//! call `zeInit`, then walk drivers → devices and expose each as a
//! `DeviceDescriptor`. `ze_device_type_t` distinguishes `GPU` and `VPU`
//! (NPU) so a single backend instance can surface both.

#![allow(clippy::missing_safety_doc)]

pub mod backend;
pub mod compute;
pub mod drv;

pub use backend::{LevelZeroBackend, LEVELZERO_BACKEND};
pub use compute::{Context, DeviceBuffer, Kernel, Module, Pipeline};

pub fn register(reg: &mut ironaccelerator_core::BackendRegistry) {
    reg.register(&LEVELZERO_BACKEND);
}
