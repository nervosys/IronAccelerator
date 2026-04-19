//! PJRT plugin loader.
//!
//! We probe the standard Cloud TPU VM / GKE plugin paths for either the
//! modern `pjrt_c_api_tpu_plugin.so` or legacy `libtpu.so`, then resolve
//! `GetPjrtApi`. If the symbol is present we consider the backend
//! available. Device shape is read from the environment variables that
//! Cloud TPU VMs set — `TPU_ACCELERATOR_TYPE` (e.g. `v5litepod-8`,
//! `v6e-16`), `TPU_NUM_DEVICES`, and `TPU_CHIPS_PER_HOST` — because
//! running a real `PJRT_Client_Create` is a heavier operation than a
//! planner-level enumerate ought to do.

use libloading::{Library, Symbol};
use once_cell::sync::OnceCell;

static PLUGIN: OnceCell<Option<Plugin>> = OnceCell::new();

struct Plugin {
    _lib: Library,
    /// Opaque pointer to the `PJRT_Api` table. Non-null after
    /// `GetPjrtApi` succeeds. Reserved for higher-layer graph compilation.
    #[allow(dead_code)]
    api: *const core::ffi::c_void,
}

// SAFETY: `PJRT_Api` is a read-only function table owned by the plugin
// for the lifetime of the process. Sending the pointer between threads is
// fine — every `PJRT_Api_*` call is required by the spec to be
// thread-safe.
unsafe impl Send for Plugin {}
unsafe impl Sync for Plugin {}

type GetPjrtApiFn = unsafe extern "C" fn() -> *const core::ffi::c_void;

const CANDIDATES: &[&str] = &[
    // Cloud TPU VM path used by `libtpu` 2024+ images.
    "pjrt_c_api_tpu_plugin.so",
    "/lib/libtpu.so",
    "libtpu.so",
    // GKE TPU node pool path.
    "/usr/local/lib/libtpu.so",
];

fn plugin() -> Option<&'static Plugin> {
    PLUGIN.get_or_init(load_plugin).as_ref()
}

fn load_plugin() -> Option<Plugin> {
    for name in CANDIDATES {
        let lib = match unsafe { Library::new(*name) } {
            Ok(l) => l,
            Err(_) => continue,
        };
        let get_api: Symbol<GetPjrtApiFn> = match unsafe { lib.get(b"GetPjrtApi\0") } {
            Ok(s) => s,
            Err(_) => continue,
        };
        let api = unsafe { get_api() };
        if api.is_null() {
            continue;
        }
        // Drop the Symbol before moving `lib` into Plugin.
        drop(get_api);
        return Some(Plugin { _lib: lib, api });
    }
    None
}

#[derive(Debug, Clone)]
pub struct TpuTopology {
    /// Raw `TPU_ACCELERATOR_TYPE` string — e.g. `"v5litepod-8"`, `"v6e-16"`.
    pub accelerator_type: String,
    /// Number of chips visible to this host (via `TPU_NUM_DEVICES`).
    pub num_devices: u32,
    /// Chips per host slice, from `TPU_CHIPS_PER_HOST`. Defaults to
    /// `num_devices` when unset.
    pub chips_per_host: u32,
}

pub fn is_plugin_available() -> bool {
    plugin().is_some()
}

pub fn topology() -> Option<TpuTopology> {
    let accel = std::env::var("TPU_ACCELERATOR_TYPE").ok()?;
    let num_devices = std::env::var("TPU_NUM_DEVICES")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .or_else(|| parse_trailing_slice_size(&accel))
        .unwrap_or(1);
    let chips_per_host = std::env::var("TPU_CHIPS_PER_HOST")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(num_devices);
    Some(TpuTopology {
        accelerator_type: accel,
        num_devices,
        chips_per_host,
    })
}

/// Extract the slice count from an accelerator type string, e.g.
/// `"v5litepod-8"` → 8, `"v6e-16"` → 16. Returns `None` if the suffix
/// after the final `-` is not a number.
fn parse_trailing_slice_size(s: &str) -> Option<u32> {
    s.rsplit('-').next().and_then(|t| t.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_trailing_slice_size() {
        assert_eq!(parse_trailing_slice_size("v5litepod-8"), Some(8));
        assert_eq!(parse_trailing_slice_size("v6e-16"), Some(16));
        assert_eq!(parse_trailing_slice_size("v4-256"), Some(256));
        assert_eq!(parse_trailing_slice_size("nonsense"), None);
    }

    #[test]
    fn plugin_probe_does_not_panic() {
        let _ = is_plugin_available();
    }
}
