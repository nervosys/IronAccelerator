//! Xilinx Runtime (XRT) loader.
//!
//! FPGA support is a **probe/enumerate scaffold**, at the same maturity as the
//! TPU and Neuron backends and for the same structural reason: an FPGA "kernel"
//! is a pre-synthesised bitstream (`.xclbin`), produced offline by Vitis in
//! minutes-to-hours, so there is no runtime-compile driver path the way CUDA
//! has NVRTC. What a driver substrate *can* do here is discover the cards.
//!
//! We dynamically load `libxrt_core` and call the stable C enumerator
//! `xclProbe`, which returns the number of XRT-managed devices. Nothing here
//! links XRT at build time — the crate compiles on any host, and a machine with
//! no XRT install simply reports zero devices.
//!
//! `drop(sym)` on a `libloading::Symbol` ends the borrow on its `Library`;
//! `Symbol` doesn't impl `Drop` but the lifetime parameter does the work.

#![allow(clippy::drop_non_drop)]

use libloading::{Library, Symbol};
use once_cell::sync::OnceCell;

/// `unsigned xclProbe(void)` — returns the count of XRT-visible devices.
/// The oldest and most stable enumeration entry point in the XRT C API.
type XclProbeFn = unsafe extern "C" fn() -> core::ffi::c_uint;

static LIB: OnceCell<Option<Loaded>> = OnceCell::new();

pub struct Loaded {
    _lib: Library,
    device_count: u32,
}

unsafe impl Send for Loaded {}
unsafe impl Sync for Loaded {}

#[cfg(target_os = "linux")]
const LIB_CANDIDATES: &[&str] = &[
    "libxrt_core.so.2",
    "libxrt_core.so",
    "/opt/xilinx/xrt/lib/libxrt_core.so.2",
    "/opt/xilinx/xrt/lib/libxrt_core.so",
];
#[cfg(target_os = "windows")]
const LIB_CANDIDATES: &[&str] = &["xrt_core.dll"];
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
const LIB_CANDIDATES: &[&str] = &[];

pub(crate) fn loaded() -> Option<&'static Loaded> {
    LIB.get_or_init(load).as_ref()
}

fn load() -> Option<Loaded> {
    for name in LIB_CANDIDATES {
        let lib = match unsafe { Library::new(*name) } {
            Ok(l) => l,
            Err(_) => continue,
        };
        unsafe {
            let probe: Symbol<XclProbeFn> = match lib.get(b"xclProbe\0") {
                Ok(s) => s,
                Err(_) => continue,
            };
            let count = probe();
            drop(probe);
            return Some(Loaded {
                _lib: lib,
                device_count: count,
            });
        }
    }
    None
}

/// Whether the XRT runtime library was located on this host.
pub fn is_available() -> bool {
    loaded().is_some()
}

/// Number of XRT-visible FPGA devices (0 if XRT is absent).
pub fn device_count() -> u32 {
    loaded().map(|l| l.device_count).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_does_not_panic_and_is_consistent() {
        // On a host without XRT this is simply (false, 0); the point is that
        // probing never panics and the two answers stay consistent.
        let avail = is_available();
        let n = device_count();
        if n > 0 {
            assert!(avail, "a nonzero device count implies the library loaded");
        }
        if !avail {
            assert_eq!(n, 0, "no library means no devices");
        }
    }
}
