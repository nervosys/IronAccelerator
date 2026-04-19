//! Process-wide kernel cache.
//!
//! NVRTC compilation is expensive (10-100s of ms). The cache is keyed by
//! `(source_hash, arch, options_hash, fn_name)` so an agent that re-asks for
//! the same kernel — even across `Strategy` decisions — gets the cached
//! [`Function`] immediately. Entries are `Arc`-shared and live for the
//! lifetime of the process.

use crate::drv::{Device, Function, Module};
use iron_cuda_sys::nvrtc::{self, NvrtcProgram};
use ironaccelerator_core::{BackendKind, Error, Result};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ffi::CString;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Compiler options passed to NVRTC.
#[derive(Clone, Debug, Default)]
pub struct CompileOptions {
    /// `-arch=` value, e.g. `"compute_90"`. Filled in automatically when empty.
    pub arch: Option<String>,
    /// Additional `-I` include paths.
    pub include_paths: Vec<String>,
    /// Raw extra flags (e.g. `--use_fast_math`).
    pub extras: Vec<String>,
}

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
    _fn_name: CString,
}

static CACHE: Lazy<RwLock<HashMap<CacheKey, CacheEntry>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

#[inline]
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() { h ^= *b as u64; h = h.wrapping_mul(0x100000001b3); }
    h
}

fn opts_hash(o: &CompileOptions) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    format!("{o:?}").hash(&mut h);
    h.finish()
}

fn nvrtc_err(op: &'static str) -> Error {
    Error::Other(op)
}

/// Get-or-compile a kernel for `device`.
pub fn get_or_compile(
    device: &Arc<Device>,
    src: &str,
    fn_name: &str,
    opts: &CompileOptions,
) -> Result<CompiledKernel> {
    let (maj, min) = device.compute_capability()?;
    let arch = opts.arch.clone().unwrap_or_else(|| format!("compute_{maj}{min}"));
    let key = CacheKey {
        src_hash: fnv1a(src),
        arch: arch.clone(),
        opts_hash: opts_hash(opts),
        fn_name: fn_name.to_string(),
    };

    if let Some(e) = CACHE.read().get(&key) {
        let f = e.module.function(fn_name)?;
        return Ok(CompiledKernel { module: e.module.clone(), function: f });
    }

    let ptx = compile(src, &arch, opts)?;
    let module = Module::load(device.clone(), &ptx)?;
    let function = module.function(fn_name)?;

    CACHE.write().insert(key, CacheEntry {
        module: module.clone(),
        _fn_name: CString::new(fn_name).map_err(|_| nvrtc_err("fn_name: NUL in string"))?,
    });
    Ok(CompiledKernel { module, function })
}

fn compile(src: &str, arch: &str, opts: &CompileOptions) -> Result<Vec<u8>> {
    let n = nvrtc::fns().map_err(|_| Error::Backend {
        backend: BackendKind::Cuda, code: -1,
    })?;

    let c_src = CString::new(src).map_err(|_| nvrtc_err("src contains NUL"))?;
    let c_name = CString::new("iron.cu").unwrap();

    let mut prog = NvrtcProgram::default();
    unsafe {
        (n.nvrtcCreateProgram)(
            &mut prog, c_src.as_ptr(), c_name.as_ptr(),
            0, std::ptr::null(), std::ptr::null(),
        ).ok().map_err(|_| nvrtc_err("nvrtcCreateProgram"))?;
    }

    // Build options.
    let mut flags: Vec<CString> = Vec::new();
    flags.push(CString::new(format!("--gpu-architecture={arch}")).unwrap());
    for inc in &opts.include_paths {
        flags.push(CString::new(format!("-I{inc}")).unwrap());
    }
    for ex in &opts.extras { flags.push(CString::new(ex.as_str()).unwrap()); }
    let ptrs: Vec<*const std::ffi::c_char> = flags.iter().map(|c| c.as_ptr()).collect();

    let compile_res = unsafe {
        (n.nvrtcCompileProgram)(prog, ptrs.len() as i32, ptrs.as_ptr())
    };

    if !compile_res.is_ok() {
        // Pull compile log for diagnostics, then destroy.
        let mut sz: usize = 0;
        unsafe { let _ = (n.nvrtcGetProgramLogSize)(prog, &mut sz); }
        let mut buf = vec![0i8; sz.max(1)];
        unsafe { let _ = (n.nvrtcGetProgramLog)(prog, buf.as_mut_ptr()); }
        unsafe { let _ = (n.nvrtcDestroyProgram)(&mut prog); }
        return Err(nvrtc_err("nvrtcCompileProgram failed"));
    }

    let mut sz: usize = 0;
    unsafe {
        (n.nvrtcGetPTXSize)(prog, &mut sz).ok().map_err(|_| nvrtc_err("nvrtcGetPTXSize"))?;
    }
    let mut ptx = vec![0u8; sz];
    unsafe {
        (n.nvrtcGetPTX)(prog, ptx.as_mut_ptr() as *mut i8)
            .ok().map_err(|_| nvrtc_err("nvrtcGetPTX"))?;
    }
    unsafe { let _ = (n.nvrtcDestroyProgram)(&mut prog); }
    Ok(ptx)
}

