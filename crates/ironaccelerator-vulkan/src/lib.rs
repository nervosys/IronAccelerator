//! # `ironaccelerator-vulkan`
//!
//! Vulkan 1.3 Compute backend. Targets cross-vendor discrete and integrated
//! GPUs that expose a Vulkan ICD, plus the Rust→WASM compute path on browsers
//! that expose WebGPU on top of a Vulkan driver.
//!
//! The crate is deliberately thin: enumerate physical devices, surface their
//! compute queue family + subgroup + FP16/INT8 support to the planner, and
//! let higher layers build pipelines out of SPIR-V modules. It intentionally
//! avoids wrapping every Vulkan object in a safe RAII type — the same
//! "throughput over guard-rails" stance the rest of IronAccelerator takes.

#![allow(clippy::missing_safety_doc)]

pub mod backend;
#[cfg(not(target_arch = "wasm32"))]
pub mod compute;
#[cfg(not(target_arch = "wasm32"))]
pub mod drv;
#[cfg(not(target_arch = "wasm32"))]
pub mod kernels;
#[cfg(not(target_arch = "wasm32"))]
pub mod shader;

pub use backend::{VulkanBackend, VULKAN_BACKEND};

/// Register the Vulkan backend into the given registry. Idempotent.
pub fn register(reg: &mut ironaccelerator_core::BackendRegistry) {
    reg.register(&VULKAN_BACKEND);
}
