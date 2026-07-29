//! # `ironaccelerator-dx12`
//!
//! Direct3D 12 backend. Covers the one GPU API on Windows that no other
//! IronAccelerator backend reaches: NVIDIA, AMD, Intel, and Qualcomm parts all
//! expose D3D12 on Windows whether or not a Vulkan ICD is installed, and on
//! Arm-based Windows devices it is frequently the only path.
//!
//! Scope is the driver line, as everywhere else in this workspace: enumerate
//! adapters, probe feature support, report capability bits, allocate buffers,
//! submit work, and synchronise. [`compute::Context`] covers the submission
//! side — a COMPUTE queue, allocator, command list, and fence, plus buffers in
//! the three standard heaps and dispatch of a compute pipeline.
//!
//! It does **not** compile shaders. Bring DXIL, the same way the CUDA backend
//! takes PTX:
//!
//! ```text
//! dxc -T cs_6_0 -E main kernel.hlsl -Fo kernel.dxil
//! ```
//!
//! Note that `dxc` only signs its output when `dxil.dll` sits beside it, and
//! D3D12 rejects unsigned DXIL. The Windows SDK ships that pair; the Vulkan
//! SDK's `dxc` does not.
//!
//! `d3d12.dll` and `dxgi.dll` are resolved with `libloading`, never linked, so
//! the crate builds on any target that has dynamic loading — Windows, Linux,
//! macOS. Where the libraries are absent, which is every non-Windows host and
//! Windows installs predating D3D12, [`Backend::is_available`] reports `false`
//! and enumeration returns empty. `wasm32` is the exception: it has no dynamic
//! loader, so this crate does not build there. Only `ironaccelerator-core` and
//! `ironaccelerator-webgpu` target WASM.
//!
//! [`Backend::is_available`]: ironaccelerator_core::Backend::is_available

pub mod backend;
pub mod compute;
pub mod drv;

pub use backend::{Dx12Backend, DX12_BACKEND};
pub use compute::{BoundPipeline, Buffer, CommandQueue, Context, PipelineState, RootSignature};
pub use drv::{Device, EnumeratedAdapter};

/// Register the D3D12 backend into the given registry. Idempotent.
pub fn register(reg: &mut ironaccelerator_core::BackendRegistry) {
    reg.register(&DX12_BACKEND);
}
