//! NVTX — profiler range/mark annotations.
//!
//! NVTX's ABI is header-only + a tiny shim library (`libnvToolsExt`). All
//! entry points we use here are ASCII variants; the Unicode forms are
//! available in the same `.so` if someone needs them later.

use crate::loader::{sym, sym_opt, try_load, LoadError};
use libloading::Library;
use std::ffi::c_char;
use std::sync::{LazyLock, OnceLock};

/// NVTX range ID returned by `RangeStart` and consumed by `RangeEnd`.
pub type NvtxRangeId = u64;

pub struct NvtxFns {
    pub nvtxMarkA: unsafe extern "C" fn(*const c_char),
    pub nvtxRangePushA: unsafe extern "C" fn(*const c_char) -> i32,
    pub nvtxRangePop: unsafe extern "C" fn() -> i32,
    pub nvtxRangeStartA: unsafe extern "C" fn(*const c_char) -> NvtxRangeId,
    pub nvtxRangeEnd: unsafe extern "C" fn(NvtxRangeId),
    pub nvtxNameCudaStreamA: Option<unsafe extern "C" fn(*mut std::ffi::c_void, *const c_char)>,
}

fn candidates() -> &'static [&'static str] {
    &["libnvToolsExt.so.1", "libnvToolsExt.so",
      "nvToolsExt64_1.dll", "nvToolsExt.dll"]
}

static LIB: OnceLock<Library> = OnceLock::new();
static FNS: LazyLock<Result<NvtxFns, LoadError>> = LazyLock::new(|| {
    let lib = try_load(candidates())?;
    let lib = LIB.get_or_init(|| lib);
    unsafe {
        Ok(NvtxFns {
            nvtxMarkA: sym(lib, "nvtx", "nvtxMarkA")?,
            nvtxRangePushA: sym(lib, "nvtx", "nvtxRangePushA")?,
            nvtxRangePop: sym(lib, "nvtx", "nvtxRangePop")?,
            nvtxRangeStartA: sym(lib, "nvtx", "nvtxRangeStartA")?,
            nvtxRangeEnd: sym(lib, "nvtx", "nvtxRangeEnd")?,
            nvtxNameCudaStreamA: sym_opt(lib, "nvtxNameCudaStreamA"),
        })
    }
});

pub fn fns() -> Result<&'static NvtxFns, &'static LoadError> { FNS.as_ref() }
pub fn is_available() -> bool { FNS.is_ok() }
