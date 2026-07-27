//! # `ironaccelerator-vulkan`
//!
//! Vulkan 1.3 Compute backend. Targets cross-vendor discrete and integrated
//! GPUs that expose a Vulkan ICD, plus the Rust→WASM compute path on browsers
//! that expose WebGPU on top of a Vulkan driver.
//!
//! The crate is deliberately thin: enumerate physical devices, report their
//! compute queue family + subgroup + FP16/INT8 support as capability bits, and
//! let higher layers build pipelines out of SPIR-V modules. Shader translation
//! is not this crate's job — bring your own SPIR-V, the same way the CUDA
//! backend takes PTX. It intentionally
//! avoids wrapping every Vulkan object in a safe RAII type — the same
//! "throughput over guard-rails" stance the rest of IronAccelerator takes.

#![allow(clippy::missing_safety_doc)]

pub mod backend;
#[cfg(not(target_arch = "wasm32"))]
pub mod compute;
#[cfg(not(target_arch = "wasm32"))]
pub mod drv;

pub use backend::{VulkanBackend, VULKAN_BACKEND};

/// Register the Vulkan backend into the given registry. Idempotent.
pub fn register(reg: &mut ironaccelerator_core::BackendRegistry) {
    reg.register(&VULKAN_BACKEND);
}
