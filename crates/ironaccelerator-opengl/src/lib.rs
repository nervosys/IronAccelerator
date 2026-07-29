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
//!
//! Once bound, the compute path is buffer → program → dispatch → readback:
//!
//! ```no_run
//! use ironaccelerator_opengl::{bind_current_context, dispatch, gl, Program, Ssbo};
//!
//! // `loader` resolves GL symbols — e.g. `glfw`/`winit`'s `get_proc_address`.
//! # fn loader(_: &str) -> *const core::ffi::c_void { core::ptr::null() }
//! unsafe { bind_current_context(loader) };
//! let gl = gl().expect("a GL 4.3+ context is current");
//!
//! let src: Vec<u8> = (0..256u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
//! let ssbo = Ssbo::with_data(gl, &src).unwrap();
//! ssbo.bind(gl, 0);
//!
//! let program = Program::from_glsl(gl, r"#version 430
//!     layout(local_size_x = 64) in;
//!     layout(std430, binding = 0) buffer Data { float v[]; };
//!     void main() { v[gl_GlobalInvocationID.x] *= 2.0; }").unwrap();
//! dispatch(gl, &program, [256 / 64, 1, 1]);
//!
//! let mut out = vec![0u8; src.len()];
//! ssbo.read_bytes(gl, &mut out);
//! ```

#![allow(clippy::missing_safety_doc)]

pub mod backend;
pub mod compute;
pub mod drv;

pub use backend::{OpenGlBackend, OPENGL_BACKEND};
pub use compute::{dispatch, gl, Program, Ssbo};
pub use drv::{bind_current_context, info, GlInfo};

/// Register the OpenGL backend. Call [`bind_current_context`] first if you
/// want the backend to report a live device.
pub fn register(reg: &mut ironaccelerator_core::BackendRegistry) {
    reg.register(&OPENGL_BACKEND);
}
