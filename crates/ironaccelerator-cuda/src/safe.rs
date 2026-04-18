//! IronAccelerator-curated re-exports of the safe driver surface + helpers.
//!
//! New code should use [`crate::drv`] directly; this module exists to give
//! downstream callers a single stable import path.

pub use crate::drv::{
    CapturedGraph, Device, DeviceBuf, DeviceView, DeviceViewMut, Event, Function,
    GraphExec, KernelArg, LaunchArgs, LaunchCfg, Module, PinnedBuf, Priority, Repr,
    Stream, TimingEvent, ZeroBits,
};

pub use crate::alloc::{alloc, alloc_zeros, from_host};
pub use crate::kernel::{get_or_compile, CompileOptions, CompiledKernel};
pub use crate::launch::{launch_1d, launch_2d, launch_dims, raw_launch};
