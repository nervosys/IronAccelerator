//! # `ironaccelerator-dx12`
//!
//! Direct3D 12 backend. Covers the one GPU API on Windows that no other
//! IronAccelerator backend reaches: NVIDIA, AMD, Intel, and Qualcomm parts all
//! expose D3D12 on Windows whether or not a Vulkan ICD is installed, and on
//! Arm-based Windows devices it is frequently the only path.
//!
//! Scope is the driver line, as everywhere else in this workspace: enumerate
//! adapters, probe feature support, report capability bits, and hand back a
//! live `ID3D12Device`. Command queues, root signatures, descriptor heaps, and
//! compute pipelines are the consumer's to build — this crate does not wrap
//! them, the same way the CUDA backend hands over a context and stops.
//!
//! `d3d12.dll` and `dxgi.dll` are resolved with `libloading`, never linked, so
//! the crate builds on every target. Where they are absent — every non-Windows
//! host, and Windows installs predating D3D12 — [`Backend::is_available`]
//! reports `false` and enumeration returns empty.
//!
//! [`Backend::is_available`]: ironaccelerator_core::Backend::is_available

pub mod backend;
pub mod drv;

pub use backend::{Dx12Backend, DX12_BACKEND};
pub use drv::{Device, EnumeratedAdapter};

/// Register the D3D12 backend into the given registry. Idempotent.
pub fn register(reg: &mut ironaccelerator_core::BackendRegistry) {
    reg.register(&DX12_BACKEND);
}
