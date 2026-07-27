//! # `ironaccelerator-core`
//!
//! Backend-agnostic foundation for **IronAccelerator** — a high-performance,
//! agentic-first *driver substrate* spanning CUDA, ROCm, Metal, and Qualcomm
//! NPUs. This crate intentionally contains *no* backend bindings; it defines
//! the trait surface every backend implements plus the vendor-neutral
//! descriptions of the things a driver actually owns: devices, capability
//! bits, memory, streams, events, and kernel launch geometry.
//!
//! **Scope.** Nothing above the driver lives here. Workload descriptors,
//! execution-strategy selection, tensor descriptors, quantization schemes,
//! and CPU reference kernels belong to the inference engine layered on top
//! ([IronWorks](https://github.com/nervosys/ironworks)), not to the substrate
//! that talks to the driver.
//!
//! IronAccelerator prioritises **throughput over guard-rails**. Where a
//! traditional safe wrapper would add bounds checks, allocation tracking, or
//! synchronous teardown, we expose a `_unchecked` fast path and let the agent
//! (or library author) opt back into safety.

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::missing_safety_doc)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod backend;
pub mod capability;
pub mod device;
pub mod dtype;
pub mod error;
pub mod handle;
pub mod kernel;
pub mod memory;
pub mod stream;

pub use backend::{Backend, BackendKind, BackendRegistry};
pub use capability::{Capability, CapabilityFlags, ComputeTier};
pub use device::{Device, DeviceDescriptor, DeviceId, Vendor};
pub use dtype::{DType, NumericClass};
pub use error::{Error, Result};
pub use kernel::{KernelLaunch, LaunchDims};
pub use memory::{Allocation, MemoryKind, MemoryPool};
pub use stream::{Event, Stream};
