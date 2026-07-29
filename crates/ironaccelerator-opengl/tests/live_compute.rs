//! End-to-end OpenGL compute on live hardware.
//!
//! The OpenGL backend never creates a GL context — the host owns that. So to
//! prove the compute path (SSBO upload, GLSL program, dispatch, readback, and
//! the unified [`ComputeDevice`] impl) actually runs on a GPU, this test stands
//! up a throwaway one itself: a hidden-window WGL context at GL 4.3 core, via
//! `windows-sys` (a Windows-only dev-dependency, never shipped).
//!
//! It skips cleanly, with a printed reason, when a context cannot be created or
//! the driver is below 4.3 — a headless CI box, an RDP session without a GL
//! ICD, etc. On a normal Windows desktop with any modern GPU it runs for real.
//!
//! ```text
//! cargo test -p ironaccelerator-opengl --test live_compute -- --nocapture
//! ```

#![cfg(windows)]

use ironaccelerator_core::ComputeDevice;
use ironaccelerator_opengl::{bind_current_context, drv, gl, GlDevice};

use std::ffi::{c_void, CString};

use windows_sys::Win32::Foundation::{HMODULE, HWND};
use windows_sys::Win32::Graphics::Gdi::{GetDC, ReleaseDC, HDC};
use windows_sys::Win32::Graphics::OpenGL::{
    wglCreateContext, wglDeleteContext, wglGetProcAddress, wglMakeCurrent, ChoosePixelFormat,
    SetPixelFormat, HGLRC, PFD_DOUBLEBUFFER, PFD_DRAW_TO_WINDOW, PFD_MAIN_PLANE, PFD_SUPPORT_OPENGL,
    PFD_TYPE_RGBA, PIXELFORMATDESCRIPTOR,
};
use windows_sys::Win32::System::LibraryLoader::{
    GetModuleHandleW, GetProcAddress, LoadLibraryA,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, RegisterClassW, CS_OWNDC, CW_USEDEFAULT,
    WNDCLASSW, WS_OVERLAPPEDWINDOW,
};

// wglCreateContextAttribsARB and its attribute constants are extension-loaded,
// not in windows-sys — declare them here.
const WGL_CONTEXT_MAJOR_VERSION_ARB: i32 = 0x2091;
const WGL_CONTEXT_MINOR_VERSION_ARB: i32 = 0x2092;
const WGL_CONTEXT_PROFILE_MASK_ARB: i32 = 0x9126;
const WGL_CONTEXT_CORE_PROFILE_BIT_ARB: i32 = 0x0000_0001;
type WglCreateContextAttribsArb =
    unsafe extern "system" fn(HDC, HGLRC, *const i32) -> HGLRC;

const GLSL: &str = r#"#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Data { float data[]; };
void main() { data[gl_GlobalInvocationID.x] *= 2.0; }
"#;

/// A live GL 4.3 context bound to a hidden window, plus the handles needed to
/// tear it down. Returns `None` if any step fails — the caller then skips.
struct Context {
    hwnd: HWND,
    hdc: HDC,
    glrc: HGLRC,
    opengl32: HMODULE,
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            wglMakeCurrent(core::ptr::null_mut(), core::ptr::null_mut());
            if !self.glrc.is_null() {
                wglDeleteContext(self.glrc);
            }
            if !self.hdc.is_null() {
                ReleaseDC(self.hwnd, self.hdc);
            }
            if !self.hwnd.is_null() {
                DestroyWindow(self.hwnd);
            }
        }
    }
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

unsafe fn create_context() -> Option<Context> {
    let hinstance = GetModuleHandleW(core::ptr::null());
    let class_name = wide("IronAccelGlHidden");

    let mut wc: WNDCLASSW = core::mem::zeroed();
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = Some(DefWindowProcW);
    wc.hInstance = hinstance;
    wc.lpszClassName = class_name.as_ptr();
    // Ignore ERROR_CLASS_ALREADY_EXISTS: a prior run in this process is fine.
    RegisterClassW(&wc);

    let hwnd = CreateWindowExW(
        0,
        class_name.as_ptr(),
        wide("ia-gl").as_ptr(),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        16,
        16,
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        hinstance,
        core::ptr::null(),
    );
    if hwnd.is_null() {
        eprintln!("skipped: CreateWindowExW failed");
        return None;
    }
    // Deliberately not shown — a hidden window still yields a usable GL context.

    let hdc = GetDC(hwnd);
    if hdc.is_null() {
        DestroyWindow(hwnd);
        eprintln!("skipped: GetDC failed");
        return None;
    }

    let mut pfd: PIXELFORMATDESCRIPTOR = core::mem::zeroed();
    pfd.nSize = core::mem::size_of::<PIXELFORMATDESCRIPTOR>() as u16;
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    pfd.iLayerType = PFD_MAIN_PLANE as u8;

    let fmt = ChoosePixelFormat(hdc, &pfd);
    if fmt == 0 || SetPixelFormat(hdc, fmt, &pfd) == 0 {
        ReleaseDC(hwnd, hdc);
        DestroyWindow(hwnd);
        eprintln!("skipped: no compatible pixel format");
        return None;
    }

    // A legacy context first, so wglGetProcAddress can resolve the modern
    // context-creation entry point.
    let legacy = wglCreateContext(hdc);
    if legacy.is_null() || wglMakeCurrent(hdc, legacy) == 0 {
        if !legacy.is_null() {
            wglDeleteContext(legacy);
        }
        ReleaseDC(hwnd, hdc);
        DestroyWindow(hwnd);
        eprintln!("skipped: legacy wglCreateContext failed");
        return None;
    }

    let create_attribs = load_proc("wglCreateContextAttribsARB")
        .map(|p| core::mem::transmute::<*const c_void, WglCreateContextAttribsArb>(p));

    let glrc = if let Some(create) = create_attribs {
        let attribs = [
            WGL_CONTEXT_MAJOR_VERSION_ARB,
            4,
            WGL_CONTEXT_MINOR_VERSION_ARB,
            3,
            WGL_CONTEXT_PROFILE_MASK_ARB,
            WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
            0,
        ];
        let core_ctx = create(hdc, core::ptr::null_mut(), attribs.as_ptr());
        if core_ctx.is_null() {
            // Driver refused 4.3 core — keep the legacy context and let the
            // caller's supports_compute check decide.
            legacy
        } else {
            wglMakeCurrent(hdc, core_ctx);
            wglDeleteContext(legacy);
            core_ctx
        }
    } else {
        legacy
    };

    let opengl32 = LoadLibraryA(c"opengl32.dll".as_ptr() as *const u8);
    if opengl32.is_null() {
        wglMakeCurrent(core::ptr::null_mut(), core::ptr::null_mut());
        wglDeleteContext(glrc);
        ReleaseDC(hwnd, hdc);
        DestroyWindow(hwnd);
        eprintln!("skipped: LoadLibraryA(opengl32) failed");
        return None;
    }

    Some(Context {
        hwnd,
        hdc,
        glrc,
        opengl32,
    })
}

/// Resolve a GL entry point: `wglGetProcAddress` for extensions, falling back
/// to `opengl32.dll` for the GL 1.1 core that predates WGL resolution.
unsafe fn load_proc(name: &str) -> Option<*const c_void> {
    let c = CString::new(name).ok()?;
    let p = wglGetProcAddress(c.as_ptr() as *const u8);
    // Some drivers return 1/2/3/-1 as "not found" sentinels rather than null.
    let as_usize = p.map(|f| f as usize).unwrap_or(0);
    if !matches!(as_usize, 0 | 1 | 2 | 3 | usize::MAX) {
        return p.map(|f| f as *const c_void);
    }
    None
}

unsafe fn load_with_fallback(name: &str, opengl32: HMODULE) -> *const c_void {
    if let Some(p) = load_proc(name) {
        return p;
    }
    let Ok(c) = CString::new(name) else {
        return core::ptr::null();
    };
    match GetProcAddress(opengl32, c.as_ptr() as *const u8) {
        Some(f) => f as *const c_void,
        None => core::ptr::null(),
    }
}

#[test]
fn compute_doubles_a_buffer_on_a_real_gl_context() {
    let ctx = match unsafe { create_context() } {
        Some(c) => c,
        None => return, // reason already printed
    };
    let opengl32 = ctx.opengl32;

    // Hand the live context to the backend exactly as a real host would.
    unsafe { bind_current_context(|s| load_with_fallback(s, opengl32)) };

    let info = drv::info().expect("context bound but no info probed");
    eprintln!(
        "GL {}.{} — {} / {}",
        info.major, info.minor, info.vendor, info.renderer
    );
    if !info.supports_compute {
        eprintln!("skipped: driver reports GL < 4.3 (no compute shaders)");
        return;
    }

    // Drive the compute path through the unified trait — same code the Vulkan
    // and D3D12 backends run.
    const N: usize = 256;
    let input: Vec<u8> = (0..N as u32).flat_map(|i| (i as f32).to_le_bytes()).collect();

    let dev = GlDevice::new();
    let buf = dev.upload(&input).expect("upload");
    assert_eq!(dev.buffer_len(&buf), input.len() as u64);
    let pipe = dev.pipeline(GLSL.as_bytes(), 1).expect("compile GLSL");
    dev.dispatch(&pipe, &[&buf], [(N / 64) as u32, 1, 1])
        .expect("dispatch");
    let mut out = vec![0u8; input.len()];
    dev.download(&buf, &mut out).expect("download");

    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for (i, g) in got.iter().enumerate() {
        assert!(
            (g - (i as f32 * 2.0)).abs() < 1e-6,
            "element {i}: got {g}, want {}",
            i as f32 * 2.0
        );
    }
    eprintln!("opengl — generic ComputeDevice roundtrip verified over {N} floats");

    // Free GL objects while the context is still current.
    if let Some(gl) = gl() {
        buf.destroy(gl);
        pipe.destroy(gl);
    }
}
