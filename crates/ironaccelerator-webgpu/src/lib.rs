//! # `ironaccelerator-webgpu`
//!
//! WebGPU backend for IronAccelerator via `wgpu`. This is the primary
//! Rust→WASM compute path and also a native path that routes to Vulkan,
//! Metal, DX12, or GLES under the hood depending on the host platform.
//!
//! Enumeration returns one descriptor per live `wgpu::Adapter`. Adapter
//! probing is synchronous via `pollster::block_on`; on WASM the caller is
//! expected to do adapter selection ahead of time and hand us a ready
//! `wgpu::Device` — we expose [`drv::bind_device`] for that path.

#![allow(clippy::missing_safety_doc)]

pub mod backend;
pub mod compute;
pub mod drv;

pub use backend::{WebGpuBackend, WEBGPU_BACKEND};
pub use drv::{bind_device, AdapterInfo};

/// Register the WebGPU backend.
pub fn register(reg: &mut ironaccelerator_core::BackendRegistry) {
    reg.register(&WEBGPU_BACKEND);
}
