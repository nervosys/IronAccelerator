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
    /// `--ftz={true,false}` — flush subnormals to zero. Combined with
    /// `prec_div=Some(false)` enables fast-math on Ampere (~5–10% kernel
    /// speedup, esp. on softmax-exp tails and reciprocal normalization).
    pub ftz: Option<bool>,
    /// `--prec-div={true,false}` — when `Some(false)`, NVRTC emits
    /// approximate division (faster, no IEEE-correct rounding).
    pub prec_div: Option<bool>,
    /// `--prec-sqrt={true,false}` — when `Some(false)`, NVRTC emits
    /// approximate sqrt.
    pub prec_sqrt: Option<bool>,
    /// `--fmad={true,false}` — fused-multiply-add contraction. Defaults to on.
    pub fmad: Option<bool>,
    /// `--use_fast_math` — superset shortcut for ftz=true, prec-div=false,
    /// prec-sqrt=false, fmad=true. Use individual flags above for finer control.
    pub use_fast_math: Option<bool>,
    /// `--maxrregcount=N` — cap per-thread registers. Sometimes lifts
    /// occupancy on register-pressured kernels.
    pub maxrregcount: Option<u32>,
    /// Raw extra flags appended after the structured fields.
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
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
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

/// Return a list of CUDA toolkit `include/` directories to add to every
/// NVRTC compile. Checks `CUDA_PATH`, `IRON_CUDA_INCLUDE`, and the standard
/// install locations on Linux / Windows / macOS.
fn default_cuda_include_paths() -> Vec<String> {
    let mut out = Vec::new();
    let mut push = |p: String| {
        if std::path::Path::new(&p).is_dir() && !out.contains(&p) {
            out.push(p);
        }
    };
    if let Ok(p) = std::env::var("IRON_CUDA_INCLUDE") {
        for part in p.split(if cfg!(windows) { ';' } else { ':' }) {
            if !part.is_empty() {
                push(part.to_string());
            }
        }
    }
    if let Ok(p) = std::env::var("CUDA_PATH") {
        push(format!("{p}/include"));
    }
    if let Ok(p) = std::env::var("CUDA_HOME") {
        push(format!("{p}/include"));
    }
    // Common defaults. Scan install roots DYNAMICALLY (newest version first)
    // instead of a hardcoded version list — a list goes stale the day a new
    // toolkit ships (v13.3 on the dev box was missed by the old list, breaking
    // every NVRTC compile whose PTX cache missed). Only dirs that actually
    // contain cuda_fp16.h count (stub version dirs exist without headers).
    if cfg!(target_os = "linux") {
        push("/usr/local/cuda/include".into());
    }
    let roots: &[&str] = if cfg!(target_os = "windows") {
        &["C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"]
    } else {
        &["/usr/local"] // cuda-12.x / cuda-13.x installs (DGX Spark)
    };
    for root in roots {
        let Ok(entries) = std::fs::read_dir(root) else {
            continue;
        };
        let mut vers: Vec<String> = entries
            .flatten()
            .filter_map(|e| {
                let inc = e.path().join("include");
                if inc.join("cuda_fp16.h").is_file() {
                    Some(inc.to_string_lossy().into_owned())
                } else {
                    None
                }
            })
            .collect();
        vers.sort();
        for v in vers.into_iter().rev() {
            push(v);
        }
    }
    out
}

/// Get-or-compile a kernel for `device`.
pub fn get_or_compile(
    device: &Arc<Device>,
    src: &str,
    fn_name: &str,
    opts: &CompileOptions,
) -> Result<CompiledKernel> {
    let (maj, min) = device.compute_capability()?;
    let arch = opts
        .arch
        .clone()
        .unwrap_or_else(|| format!("compute_{maj}{min}"));
    let key = CacheKey {
        src_hash: fnv1a(src),
        arch: arch.clone(),
        opts_hash: opts_hash(opts),
        fn_name: fn_name.to_string(),
    };

    if let Some(e) = CACHE.read().get(&key) {
        let f = e.module.function(fn_name)?;
        return Ok(CompiledKernel {
            module: e.module.clone(),
            function: f,
        });
    }

    // Persistent on-disk PTX cache. A hit here skips NVRTC entirely (the 10-100ms
    // compile) while still rebinding the function fresh per process.
    let ptx = match disk_cache_load(&key) {
        Some(bytes) => bytes,
        None => {
            let bytes = compile(src, &arch, opts)?;
            disk_cache_store(&key, &bytes);
            bytes
        }
    };
    let module = Module::load(device.clone(), &ptx)?;
    let function = module.function(fn_name)?;

    CACHE.write().insert(
        key,
        CacheEntry {
            module: module.clone(),
            _fn_name: CString::new(fn_name).map_err(|_| nvrtc_err("fn_name: NUL in string"))?,
        },
    );
    Ok(CompiledKernel { module, function })
}

// ────────────────────────────────────────────────────────────────────────────
// On-disk PTX cache
// ────────────────────────────────────────────────────────────────────────────
//
// A second-level cache survives the process so restarts don't re-compile the
// same kernels. Disabled by setting `IRON_CUDA_PTX_CACHE=0`.
//
// Layout: `<root>/<arch>/<16-hex src_hash>_<16-hex opts_hash>.ptx`
// The fn_name is intentionally NOT part of the filename — a single PTX image
// typically exports multiple kernels, so two keys that differ only by fn_name
// can share the compiled image. The in-memory `CACHE` is still keyed by
// fn_name so different `Function` objects stay distinct.

fn disk_cache_disabled() -> bool {
    matches!(
        std::env::var("IRON_CUDA_PTX_CACHE").as_deref(),
        Ok("0") | Ok("off") | Ok("false")
    )
}

fn disk_cache_root() -> Option<std::path::PathBuf> {
    if disk_cache_disabled() {
        return None;
    }
    if let Ok(p) = std::env::var("IRON_CUDA_PTX_CACHE_DIR") {
        if !p.is_empty() {
            return Some(std::path::PathBuf::from(p));
        }
    }
    let mut p = std::env::temp_dir();
    p.push("ironaccelerator");
    p.push("ptx");
    Some(p)
}

fn disk_cache_path(key: &CacheKey) -> Option<std::path::PathBuf> {
    let mut p = disk_cache_root()?;
    p.push(&key.arch);
    let _ = std::fs::create_dir_all(&p);
    p.push(format!("{:016x}_{:016x}.ptx", key.src_hash, key.opts_hash));
    Some(p)
}

fn disk_cache_load(key: &CacheKey) -> Option<Vec<u8>> {
    let path = disk_cache_path(key)?;
    std::fs::read(&path).ok()
}

fn disk_cache_store(key: &CacheKey, ptx: &[u8]) {
    let Some(path) = disk_cache_path(key) else {
        return;
    };
    // Best-effort: atomically rename from a temp sibling so a crash mid-write
    // doesn't leave a truncated .ptx that future runs would try to load.
    let tmp = path.with_extension("ptx.tmp");
    if std::fs::write(&tmp, ptx).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    }
}

/// Compile CUDA C++ source to PTX. Arch defaults to `compute_80` if not
/// supplied via `opts.arch`.
pub fn compile(src: &str, arch: &str, opts: &CompileOptions) -> Result<Vec<u8>> {
    let n = nvrtc::fns().map_err(|_| Error::Backend {
        backend: BackendKind::Cuda,
        code: -1,
    })?;

    let c_src = CString::new(src).map_err(|_| nvrtc_err("src contains NUL"))?;
    let c_name = CString::new("iron.cu").unwrap();

    let mut prog = NvrtcProgram::default();
    unsafe {
        (n.nvrtcCreateProgram)(
            &mut prog,
            c_src.as_ptr(),
            c_name.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
        .ok()
        .map_err(|_| nvrtc_err("nvrtcCreateProgram"))?;
    }

    // Build options.
    let mut flags: Vec<CString> = Vec::new();
    flags.push(CString::new(format!("--gpu-architecture={arch}")).unwrap());
    for inc in &opts.include_paths {
        flags.push(CString::new(format!("-I{inc}")).unwrap());
    }
    // Best-effort auto-detect of the CUDA toolkit `include/` directory so
    // kernels that use `cuda_fp16.h` / `cuda_bf16.h` etc. resolve without
    // callers having to plumb the path themselves.
    for inc in default_cuda_include_paths() {
        flags.push(CString::new(format!("-I{inc}")).unwrap());
    }
    if let Some(v) = opts.ftz {
        flags.push(CString::new(format!("--ftz={v}")).unwrap());
    }
    if let Some(v) = opts.prec_div {
        flags.push(CString::new(format!("--prec-div={v}")).unwrap());
    }
    if let Some(v) = opts.prec_sqrt {
        flags.push(CString::new(format!("--prec-sqrt={v}")).unwrap());
    }
    if let Some(v) = opts.fmad {
        flags.push(CString::new(format!("--fmad={v}")).unwrap());
    }
    if opts.use_fast_math == Some(true) {
        flags.push(CString::new("--use_fast_math").unwrap());
    }
    if let Some(n) = opts.maxrregcount {
        flags.push(CString::new(format!("--maxrregcount={n}")).unwrap());
    }
    for ex in &opts.extras {
        flags.push(CString::new(ex.as_str()).unwrap());
    }
    let ptrs: Vec<*const std::ffi::c_char> = flags.iter().map(|c| c.as_ptr()).collect();

    let compile_res = unsafe { (n.nvrtcCompileProgram)(prog, ptrs.len() as i32, ptrs.as_ptr()) };

    if !compile_res.is_ok() {
        // Pull compile log for diagnostics, then destroy.
        let mut sz: usize = 0;
        unsafe {
            let _ = (n.nvrtcGetProgramLogSize)(prog, &mut sz);
        }
        let mut buf = vec![0u8; sz.max(1)];
        unsafe {
            let _ = (n.nvrtcGetProgramLog)(prog, buf.as_mut_ptr() as *mut i8);
        }
        unsafe {
            let _ = (n.nvrtcDestroyProgram)(&mut prog);
        }
        let log = String::from_utf8_lossy(&buf)
            .trim_end_matches('\0')
            .to_string();
        eprintln!("[nvrtc] compile log:\n{log}");
        return Err(nvrtc_err("nvrtcCompileProgram failed"));
    }

    let mut sz: usize = 0;
    unsafe {
        (n.nvrtcGetPTXSize)(prog, &mut sz)
            .ok()
            .map_err(|_| nvrtc_err("nvrtcGetPTXSize"))?;
    }
    let mut ptx = vec![0u8; sz];
    unsafe {
        (n.nvrtcGetPTX)(prog, ptx.as_mut_ptr() as *mut i8)
            .ok()
            .map_err(|_| nvrtc_err("nvrtcGetPTX"))?;
    }
    unsafe {
        let _ = (n.nvrtcDestroyProgram)(&mut prog);
    }
    Ok(ptx)
}
