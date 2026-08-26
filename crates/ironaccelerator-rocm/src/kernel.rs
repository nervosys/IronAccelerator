//! HIPRTC runtime kernel compilation + an in-memory module cache — the ROCm
//! analogue of `ironaccelerator_cuda::kernel`.
//!
//! Compiles HIP C++ source to a code object with HIPRTC and loads it via
//! `hipModuleLoadData`, caching the resulting `Arc<Module>` per (source, arch,
//! options, entry point). The cache is **race-free by construction**: two
//! threads that miss the same key concurrently converge on a single shared
//! module through a double-checked insert — the exact property the CUDA kernel
//! cache had to be hardened to guarantee, built in here from the start.
//!
//! ## Status: compiles clean, not live-tested
//!
//! This workspace has no AMD GPU, so the compile→load→launch path here has not
//! run against real hardware. It mirrors the validated CUDA path structurally
//! (same cache shape, same HIPRTC/NVRTC-parallel API). Treat it as
//! ready-for-hardware, not proven on it.

use std::collections::HashMap;
use std::ffi::{c_char, CString};
use std::hash::{Hash, Hasher};
use std::ptr;
use std::sync::{Arc, LazyLock, RwLock};

use iron_rocm_sys::hiprtc as rtc;

use crate::drv::{Device, Error, Function, Module, Result};

/// Options passed to HIPRTC for a runtime compile.
#[derive(Debug, Clone, Default)]
pub struct CompileOptions {
    /// `--offload-arch=<gfx…>` target (e.g. `"gfx942"`). When `None`, HIPRTC
    /// compiles for the architecture it detects for the current device; set it
    /// explicitly for reproducible cross-device builds.
    pub offload_arch: Option<String>,
    /// Extra raw HIPRTC options appended verbatim (e.g. `"-ffast-math"`).
    pub extras: Vec<String>,
}

/// A compiled kernel: the owning module plus a launchable function handle.
pub struct CompiledKernel {
    pub module: Arc<Module>,
    pub function: Function,
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct CacheKey {
    src_hash: u64,
    arch: String,
    opts_hash: u64,
    fn_name: String,
}

struct CacheEntry {
    module: Arc<Module>,
}

static CACHE: LazyLock<RwLock<HashMap<CacheKey, CacheEntry>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

#[inline]
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn opts_hash(o: &CompileOptions) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    format!("{o:?}").hash(&mut h);
    h.finish()
}

fn rtc_fns() -> Result<&'static rtc::HiprtcFns> {
    rtc::fns().map_err(|e| Error::NotAvailable {
        lib: "hiprtc",
        detail: format!("{e}"),
    })
}

/// Read the HIPRTC build log for a program (empty string if unavailable).
///
/// # Safety
/// `prog` must be a valid, not-yet-destroyed `HiprtcProgram`.
unsafe fn program_log(f: &rtc::HiprtcFns, prog: rtc::HiprtcProgram) -> String {
    let mut size = 0usize;
    if !(f.hiprtcGetProgramLogSize)(prog, &mut size).is_ok() || size <= 1 {
        return String::new();
    }
    let mut buf = vec![0u8; size];
    if !(f.hiprtcGetProgramLog)(prog, buf.as_mut_ptr() as *mut c_char).is_ok() {
        return String::new();
    }
    // Drop the trailing NUL the API includes in `size`.
    while matches!(buf.last(), Some(0)) {
        buf.pop();
    }
    String::from_utf8_lossy(&buf).into_owned()
}

/// Compile HIP source to a code object with HIPRTC.
fn compile(src: &str, opts: &CompileOptions) -> Result<Vec<u8>> {
    let f = rtc_fns()?;

    let mut opt_strings: Vec<CString> = Vec::new();
    if let Some(arch) = &opts.offload_arch {
        if let Ok(c) = CString::new(format!("--offload-arch={arch}")) {
            opt_strings.push(c);
        }
    }
    for e in &opts.extras {
        if let Ok(c) = CString::new(e.as_str()) {
            opt_strings.push(c);
        }
    }
    let opt_ptrs: Vec<*const c_char> = opt_strings.iter().map(|c| c.as_ptr()).collect();

    let csrc = CString::new(src).map_err(|_| Error::Precondition {
        op: "hiprtcCreateProgram",
        msg: "source contains NUL".into(),
    })?;
    let cname = CString::new("ironaccelerator.hip").unwrap();

    let mut prog = rtc::HiprtcProgram::default();
    unsafe {
        (f.hiprtcCreateProgram)(
            &mut prog,
            csrc.as_ptr(),
            cname.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
        )
    }
    .ok()
    .map_err(|r| Error::Precondition {
        op: "hiprtcCreateProgram",
        msg: format!("{r:?}"),
    })?;

    let opts_ptr = if opt_ptrs.is_empty() {
        ptr::null()
    } else {
        opt_ptrs.as_ptr()
    };
    let compiled = unsafe { (f.hiprtcCompileProgram)(prog, opt_ptrs.len() as i32, opts_ptr) };
    if !compiled.is_ok() {
        let log = unsafe { program_log(f, prog) };
        unsafe {
            let _ = (f.hiprtcDestroyProgram)(&mut prog);
        }
        return Err(Error::Precondition {
            op: "hiprtcCompileProgram",
            msg: if log.is_empty() {
                format!("{compiled:?}")
            } else {
                log
            },
        });
    }

    let mut size = 0usize;
    let size_res = unsafe { (f.hiprtcGetCodeSize)(prog, &mut size) };
    let code = if size_res.is_ok() && size > 0 {
        let mut code = vec![0u8; size];
        let get = unsafe { (f.hiprtcGetCode)(prog, code.as_mut_ptr() as *mut c_char) };
        if get.is_ok() {
            Some(code)
        } else {
            None
        }
    } else {
        None
    };
    unsafe {
        let _ = (f.hiprtcDestroyProgram)(&mut prog);
    }
    code.ok_or(Error::Precondition {
        op: "hiprtcGetCode",
        msg: "failed to read compiled code object".into(),
    })
}

/// Get-or-compile a kernel for `device`, caching the module process-wide.
pub fn get_or_compile(
    device: &Arc<Device>,
    src: &str,
    fn_name: &str,
    opts: &CompileOptions,
) -> Result<CompiledKernel> {
    let arch = opts
        .offload_arch
        .clone()
        .unwrap_or_else(|| "default".into());
    let key = CacheKey {
        src_hash: fnv1a(src),
        arch,
        opts_hash: opts_hash(opts),
        fn_name: fn_name.to_string(),
    };

    if let Some(e) = CACHE.read().unwrap_or_else(|e| e.into_inner()).get(&key) {
        let function = e.module.function(fn_name)?;
        return Ok(CompiledKernel {
            module: e.module.clone(),
            function,
        });
    }

    let code = compile(src, opts)?;
    let compiled = Module::load(device.clone(), &code)?;

    // Double-checked insert: a parallel caller may have compiled the same key
    // while we were compiling. Keep whichever module reached the cache first so
    // every caller for a key shares one `Arc<Module>`.
    let module = {
        let mut w = CACHE.write().unwrap_or_else(|e| e.into_inner());
        w.entry(key)
            .or_insert(CacheEntry { module: compiled })
            .module
            .clone()
    };
    let function = module.function(fn_name)?;
    Ok(CompiledKernel { module, function })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_hash_is_stable_and_distinguishes() {
        assert_eq!(fnv1a("saxpy"), fnv1a("saxpy"));
        assert_ne!(fnv1a("saxpy"), fnv1a("saxpz"));
    }

    #[test]
    fn options_hash_reflects_arch() {
        let a = CompileOptions {
            offload_arch: Some("gfx942".into()),
            ..Default::default()
        };
        let b = CompileOptions {
            offload_arch: Some("gfx90a".into()),
            ..Default::default()
        };
        assert_eq!(opts_hash(&a), opts_hash(&a.clone()));
        assert_ne!(opts_hash(&a), opts_hash(&b));
    }
}
