//! OpenGL context + capability probing via `glow`.
//!
//! A GL context must already be current on the calling thread — we don't
//! spin up windowing. Hosts bind their context once at startup by calling
//! [`bind_current_context`] with a loader closure (typically `glfw`'s /
//! `winit`'s `get_proc_address`).

use glow::HasContext;
use once_cell::sync::OnceCell;

static GL: OnceCell<glow::Context> = OnceCell::new();
static INFO: OnceCell<GlInfo> = OnceCell::new();

#[derive(Debug, Clone)]
pub struct GlInfo {
    pub renderer: String,
    pub vendor: String,
    pub version: String,
    pub glsl_version: String,
    pub major: u32,
    pub minor: u32,
    pub max_compute_work_group_invocations: u32,
    pub max_compute_work_group_count: [u32; 3],
    pub max_shared_memory_bytes: u32,
    pub supports_compute: bool,
}

/// Bind the OpenGL backend to the currently-current GL context on this thread.
/// `loader` is called once per GL symbol to resolve pointers (e.g.
/// `|s| window.get_proc_address(s) as *const _`).
///
/// Safety: the caller must ensure a valid GL context is current on the
/// calling thread for the lifetime of the returned binding.
pub unsafe fn bind_current_context<F>(loader: F)
where
    F: FnMut(&str) -> *const core::ffi::c_void,
{
    let ctx = glow::Context::from_loader_function(loader);
    let info = probe(&ctx);
    let _ = GL.set(ctx);
    let _ = INFO.set(info);
}

pub fn info() -> Option<GlInfo> {
    INFO.get().cloned()
}

pub fn shared_context() -> Option<&'static glow::Context> {
    GL.get()
}

fn probe(gl: &glow::Context) -> GlInfo {
    unsafe {
        let renderer = gl.get_parameter_string(glow::RENDERER);
        let vendor = gl.get_parameter_string(glow::VENDOR);
        let version = gl.get_parameter_string(glow::VERSION);
        let glsl_version = gl.get_parameter_string(glow::SHADING_LANGUAGE_VERSION);
        let major = gl.get_parameter_i32(glow::MAJOR_VERSION).max(0) as u32;
        let minor = gl.get_parameter_i32(glow::MINOR_VERSION).max(0) as u32;
        let supports_compute = (major, minor) >= (4, 3);

        let (mcwgi, mcwgc, msm) = if supports_compute {
            let mcwgi = gl
                .get_parameter_i32(glow::MAX_COMPUTE_WORK_GROUP_INVOCATIONS)
                .max(0) as u32;
            let x = gl
                .get_parameter_indexed_i32(glow::MAX_COMPUTE_WORK_GROUP_COUNT, 0)
                .max(0) as u32;
            let y = gl
                .get_parameter_indexed_i32(glow::MAX_COMPUTE_WORK_GROUP_COUNT, 1)
                .max(0) as u32;
            let z = gl
                .get_parameter_indexed_i32(glow::MAX_COMPUTE_WORK_GROUP_COUNT, 2)
                .max(0) as u32;
            let msm = gl
                .get_parameter_i32(glow::MAX_COMPUTE_SHARED_MEMORY_SIZE)
                .max(0) as u32;
            (mcwgi, [x, y, z], msm)
        } else {
            (0, [0, 0, 0], 0)
        };

        GlInfo {
            renderer,
            vendor,
            version,
            glsl_version,
            major,
            minor,
            max_compute_work_group_invocations: mcwgi,
            max_compute_work_group_count: mcwgc,
            max_shared_memory_bytes: msm,
            supports_compute,
        }
    }
}
