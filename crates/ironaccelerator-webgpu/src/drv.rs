//! Host-bound WebGPU adapter registration.
//!
//! WebGPU has no synchronous enumeration. `navigator.gpu.requestAdapter()` and
//! `GPUAdapter.requestDevice()` both return promises, and [`Backend`] is a
//! synchronous trait — so this crate cannot negotiate an adapter itself
//! without blocking, which is not permitted on a browser's main thread.
//!
//! The host therefore does the async negotiation and registers the result with
//! [`bind_adapter`]. Everything downstream — [`enumerate`], the [`Backend`]
//! impl, `Runtime::devices()` — reads what was bound. Until a host binds, the
//! backend honestly reports unavailable.
//!
//! [`Backend`]: ironaccelerator_core::Backend

use std::sync::Mutex;

/// What a host learned about its adapter, in WebGPU's own vocabulary.
///
/// The fields mirror `GPUAdapterInfo`, `GPUSupportedLimits`, and the two
/// optional `GPUFeatureName`s that matter for compute. WebGPU deliberately
/// does not expose PCI vendor/device IDs — `vendor` and `architecture` are
/// lowercase strings such as `"nvidia"` / `"ampere"`, and may be empty when
/// the browser chooses to withhold them for fingerprinting reasons.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AdapterInfo {
    /// `GPUAdapterInfo.vendor`, e.g. `"nvidia"`, `"amd"`, `"intel"`, `"apple"`.
    pub vendor: String,
    /// `GPUAdapterInfo.architecture`, e.g. `"ampere"`, `"rdna-3"`.
    pub architecture: String,
    /// `GPUAdapterInfo.device` — often empty in browsers.
    pub device: String,
    /// `GPUAdapterInfo.description` — often empty in browsers.
    pub description: String,
    /// `GPUAdapter.isFallbackAdapter` — a software rasteriser.
    pub is_fallback: bool,
    /// The `"shader-f16"` feature was granted on the device.
    pub shader_f16: bool,
    /// The `"subgroups"` feature was granted on the device.
    pub subgroups: bool,
    /// `GPUSupportedLimits.maxBufferSize`.
    pub max_buffer_size: u64,
    /// `GPUSupportedLimits.maxStorageBufferBindingSize`.
    pub max_storage_buffer_binding_size: u64,
    /// `GPUSupportedLimits.maxComputeWorkgroupSizeX`.
    pub max_compute_workgroup_size_x: u32,
    /// `GPUSupportedLimits.maxComputeInvocationsPerWorkgroup`.
    pub max_compute_invocations_per_workgroup: u32,
}

static BOUND: Mutex<Option<AdapterInfo>> = Mutex::new(None);

/// Test-only: a single lock shared by every test in this crate that touches the
/// process-global [`BOUND`] binding. The `drv` and `backend` test modules run
/// in one binary across parallel threads and both mutate `BOUND`; without a
/// *shared* guard each module would serialise only against itself and the two
/// would race. Every such test takes this lock first.
#[cfg(test)]
pub(crate) static TEST_BINDING_LOCK: Mutex<()> = Mutex::new(());

/// Register the adapter this host negotiated. Replaces any previous binding.
///
/// Call after `requestAdapter()` / `requestDevice()` resolve. The `GPUDevice`
/// itself stays with the host — this crate records what the adapter *is* so it
/// appears in the device survey, and does not take ownership of the handle.
pub fn bind_adapter(info: AdapterInfo) {
    *BOUND.lock().unwrap_or_else(|e| e.into_inner()) = Some(info);
}

/// Drop the current binding. After this the backend reports unavailable again.
pub fn unbind_adapter() {
    *BOUND.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

/// The currently bound adapter, if any.
pub fn bound_adapter() -> Option<AdapterInfo> {
    BOUND.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Whether a host has bound an adapter.
pub fn is_available() -> bool {
    BOUND
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|a| !a.is_fallback)
}

/// The bound adapter as a one-element list, or empty. WebGPU exposes exactly
/// one adapter per `requestAdapter` call; there is no multi-device survey.
pub fn enumerate() -> Vec<AdapterInfo> {
    match bound_adapter() {
        Some(a) if !a.is_fallback => vec![a],
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialises the tests below against every other test in the crate that
    /// touches the process-wide binding, via the shared [`TEST_BINDING_LOCK`].
    fn with_clean_binding<T>(f: impl FnOnce() -> T) -> T {
        let _g = TEST_BINDING_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unbind_adapter();
        let out = f();
        unbind_adapter();
        out
    }

    fn sample() -> AdapterInfo {
        AdapterInfo {
            vendor: "nvidia".into(),
            architecture: "ampere".into(),
            shader_f16: true,
            subgroups: true,
            max_buffer_size: 1 << 31,
            ..Default::default()
        }
    }

    #[test]
    fn unbound_is_unavailable_and_enumerates_empty() {
        with_clean_binding(|| {
            assert!(!is_available());
            assert!(enumerate().is_empty());
        });
    }

    #[test]
    fn bound_adapter_round_trips() {
        with_clean_binding(|| {
            bind_adapter(sample());
            assert!(is_available());
            assert_eq!(enumerate(), vec![sample()]);
            assert_eq!(bound_adapter(), Some(sample()));
        });
    }

    #[test]
    fn fallback_adapter_is_not_offered() {
        with_clean_binding(|| {
            bind_adapter(AdapterInfo {
                is_fallback: true,
                ..sample()
            });
            assert!(!is_available(), "software fallback must not be offered");
            assert!(enumerate().is_empty());
        });
    }

    #[test]
    fn unbind_clears() {
        with_clean_binding(|| {
            bind_adapter(sample());
            unbind_adapter();
            assert!(!is_available());
            assert_eq!(bound_adapter(), None);
        });
    }
}
