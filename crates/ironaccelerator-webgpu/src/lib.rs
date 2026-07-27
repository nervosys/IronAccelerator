//! # `ironaccelerator-webgpu`
//!
//! WebGPU backend for the browser. This is the WASM compute path, and only
//! that: on native targets, Vulkan, Metal, D3D12, and OpenGL reach the same
//! hardware directly, so routing through a portability layer would add a
//! translation step without adding coverage.
//!
//! ## How it works
//!
//! WebGPU adapter negotiation is asynchronous and
//! [`Backend`](ironaccelerator_core::Backend) is not, so this crate does not
//! negotiate. The host awaits `requestAdapter()` / `requestDevice()`, keeps the
//! resulting `GPUDevice`, and describes it once via [`drv::bind_adapter`].
//! From then on the adapter shows up in `Runtime::devices()` alongside every
//! native backend.
//!
//! ```
//! use ironaccelerator_webgpu::drv::{bind_adapter, AdapterInfo};
//!
//! // After `requestDevice()` resolves, from whatever binding layer you use:
//! bind_adapter(AdapterInfo {
//!     vendor: "nvidia".into(),
//!     architecture: "ampere".into(),
//!     shader_f16: true,
//!     max_buffer_size: 1 << 28,
//!     ..Default::default()
//! });
//! ```
//!
//! ## What it deliberately does not do
//!
//! No buffers, pipelines, or dispatch. Those need a live `GPUDevice`, which
//! stays with the host, and wrapping them would mean re-exporting a binding
//! crate's types through this API — the coupling that made the previous
//! `wgpu`-based version 98 transitive dependencies deep. The host already
//! holds the device and can call WebGPU directly.
//!
//! Consequently this crate has no dependencies beyond
//! [`ironaccelerator-core`](ironaccelerator_core) and compiles for every
//! target, `wasm32-unknown-unknown` included, with no feature flags and no
//! `--cfg` requirements.

pub mod backend;
pub mod drv;

pub use backend::{WebGpuBackend, WEBGPU_BACKEND};
pub use drv::{bind_adapter, bound_adapter, unbind_adapter, AdapterInfo};

/// Register the WebGPU backend into the given registry. Idempotent.
///
/// Registering before [`bind_adapter`] is fine — the backend simply reports
/// unavailable until a host binds one.
pub fn register(reg: &mut ironaccelerator_core::BackendRegistry) {
    reg.register(&WEBGPU_BACKEND);
}
