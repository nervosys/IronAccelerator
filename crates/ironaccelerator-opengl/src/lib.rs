//! # `ironaccelerator-opengl`
//!
//! OpenGL 4.3+ compute shader backend for IronAccelerator. Used as a
//! legacy / embedded-GPU fallback when Vulkan, Metal, or WebGPU aren't
//! available (e.g. older Linux Mesa stacks, integrated Intel GPUs on
//! locked-down kernels).
//!
//! OpenGL has no standalone physical-device enumeration — a GL context must
//! already exist on the calling thread. The host supplies a loader closure
//! via [`bind_current_context`]; after that, `OpenGlBackend::enumerate`
//! reports the single device bound to the current context.

#![allow(clippy::missing_safety_doc)]

pub mod backend;
pub mod compute;
pub mod drv;

pub use backend::{OpenGlBackend, OPENGL_BACKEND};
pub use drv::bind_current_context;

/// Register the OpenGL backend. Call [`bind_current_context`] first if you
/// want the backend to report a live device.
pub fn register(reg: &mut ironaccelerator_core::BackendRegistry) {
    reg.register(&OPENGL_BACKEND);
}
