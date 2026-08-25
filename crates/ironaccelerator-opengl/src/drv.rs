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

/// Parse `(major, minor)` from a `GL_VERSION` string such as `"4.3.0 NVIDIA…"`
/// or `"1.1.0"`. Unlike the `GL_MAJOR_VERSION` / `GL_MINOR_VERSION` enums —
/// which only exist from GL 3.0 and read back 0 on anything older — the version
/// *string* is defined on every GL version, so this is the reliable probe on an
/// ancient context (e.g. a headless runner's GL 1.1 software rasteriser).
fn parse_gl_version(version: &str) -> (u32, u32) {
    let head = version.split_whitespace().next().unwrap_or("");
    let mut parts = head.split('.');
    let major = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let minor = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    (major, minor)
}

fn probe(gl: &glow::Context) -> GlInfo {
    unsafe {
        let renderer = gl.get_parameter_string(glow::RENDERER);
        let vendor = gl.get_parameter_string(glow::VENDOR);
        let version = gl.get_parameter_string(glow::VERSION);
        let (major, minor) = parse_gl_version(&version);
        // GL_SHADING_LANGUAGE_VERSION only exists from GL 2.0. Querying it on an
        // older context raises GL_INVALID_ENUM, and glow's string getter *panics*
        // on the failed read ("context version too outdated") rather than
        // returning an error — a GDI-generic GL 1.1 context on a headless CI box
        // hits exactly this. Guard the query so probing an ancient context
        // reports "no compute" cleanly instead of bringing the process down.
        let glsl_version = if major >= 2 {
            gl.get_parameter_string(glow::SHADING_LANGUAGE_VERSION)
        } else {
            String::new()
        };
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

#[cfg(test)]
mod tests {
    use super::parse_gl_version;

    #[test]
    fn parses_vendor_suffixed_and_bare_versions() {
        assert_eq!(parse_gl_version("4.3.0 NVIDIA 552.44"), (4, 3));
        assert_eq!(
            parse_gl_version("4.6.0 Compatibility Profile Context"),
            (4, 6)
        );
        assert_eq!(parse_gl_version("1.1.0"), (1, 1)); // GDI-generic on a CI box
        assert_eq!(parse_gl_version("3.2 Core Profile"), (3, 2));
    }

    #[test]
    fn malformed_versions_degrade_to_zero_not_panic() {
        assert_eq!(parse_gl_version(""), (0, 0));
        assert_eq!(parse_gl_version("OpenGL ES 3.1"), (0, 0)); // leading non-numeric token
        assert_eq!(parse_gl_version("garbage"), (0, 0));
        assert_eq!(parse_gl_version("4"), (4, 0)); // major only, minor defaults
    }
}
