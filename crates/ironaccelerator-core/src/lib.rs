//! # `ironaccelerator-core`
//!
//! Backend-agnostic foundation for **IronAccelerator** — a high-performance,
//! agentic-first acceleration library spanning CUDA, ROCm, Metal, and Qualcomm
//! NPUs. This crate intentionally contains *no* backend bindings; it defines
//! the trait surface that every backend implements and the lightweight
//! description types used by the discovery / ontology layer.
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
pub mod strategy;
pub mod tensor;
pub mod workload;

pub use backend::{Backend, BackendKind, BackendRegistry};
pub use capability::{Capability, CapabilityFlags, ComputeTier};
pub use device::{Device, DeviceDescriptor, DeviceId, Vendor};
pub use dtype::{DType, NumericClass};
pub use error::{Error, Result};
pub use kernel::{KernelLaunch, LaunchDims};
pub use memory::{Allocation, MemoryKind, MemoryPool};
pub use stream::{Event, Stream};
pub use strategy::{Strategy, StrategyHint, StrategyScore};
pub use tensor::{Layout, TensorDesc};
pub use workload::{Precision, Workload, WorkloadKind, WorkloadShape};
