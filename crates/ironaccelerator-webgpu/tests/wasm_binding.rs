//! Binding round-trip for the WebGPU backend.
//!
//! Compiled for both native and `wasm32`. Under `wasm-bindgen-test` these run
//! in a headless browser:
//!
//! ```text
//! wasm-pack test --headless --chrome -p ironaccelerator-webgpu
//! ```
//!
//! They exercise the binding surface, not WebGPU itself — a real
//! `navigator.gpu` round-trip needs the host's async negotiation, which by
//! design lives outside this crate. What is verified here is that whatever the
//! host reports arrives intact at `Backend`, on the target it will actually run
//! on.

use ironaccelerator_core::{Backend, CapabilityFlags, Vendor};
use ironaccelerator_webgpu::{bind_adapter, unbind_adapter, AdapterInfo, WEBGPU_BACKEND};

#[cfg(not(target_arch = "wasm32"))]
use std::prelude::v1::test as test_case;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test_case;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

/// The binding is process-wide, so these tests must not interleave. Native
/// `cargo test` runs them on parallel threads; `wasm-bindgen-test` is
/// single-threaded, where this degrades to a no-op.
#[cfg(not(target_arch = "wasm32"))]
fn serialize() -> std::sync::MutexGuard<'static, ()> {
    static GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GUARD.lock().unwrap_or_else(|e| e.into_inner())
}
#[cfg(target_arch = "wasm32")]
fn serialize() {}

/// Mirrors what a host would report for a discrete NVIDIA part in Chrome.
fn chrome_like() -> AdapterInfo {
    AdapterInfo {
        vendor: "nvidia".into(),
        architecture: "ampere".into(),
        shader_f16: true,
        subgroups: true,
        max_buffer_size: 1 << 28,
        max_storage_buffer_binding_size: 1 << 27,
        max_compute_workgroup_size_x: 256,
        max_compute_invocations_per_workgroup: 256,
        ..Default::default()
    }
}

#[test_case]
fn host_binding_round_trips_through_the_backend() {
    let _g = serialize();
    unbind_adapter();
    assert!(
        !WEBGPU_BACKEND.is_available(),
        "unbound must be unavailable"
    );

    bind_adapter(chrome_like());
    assert!(WEBGPU_BACKEND.is_available());

    let devices = WEBGPU_BACKEND.enumerate().expect("enumerate");
    assert_eq!(devices.len(), 1, "WebGPU exposes one adapter per binding");
    assert_eq!(devices[0].vendor, Vendor::Nvidia);
    assert_eq!(devices[0].arch, "webgpu-ampere");
    assert_eq!(devices[0].total_memory_bytes, 1 << 28);

    let flags = WEBGPU_BACKEND.capabilities(0).expect("capabilities");
    assert_eq!(flags, CapabilityFlags::FP32 | CapabilityFlags::FP16);
    assert_eq!(flags, devices[0].capability.flags);

    unbind_adapter();
    assert!(!WEBGPU_BACKEND.is_available(), "unbind must clear");
}

#[test_case]
fn fallback_adapters_are_never_offered() {
    let _g = serialize();
    unbind_adapter();
    bind_adapter(AdapterInfo {
        is_fallback: true,
        ..chrome_like()
    });
    assert!(!WEBGPU_BACKEND.is_available());
    assert!(WEBGPU_BACKEND.enumerate().unwrap().is_empty());
    unbind_adapter();
}

#[test_case]
fn missing_shader_f16_is_not_reported_as_fp16() {
    let _g = serialize();
    unbind_adapter();
    bind_adapter(AdapterInfo {
        shader_f16: false,
        ..chrome_like()
    });
    assert_eq!(
        WEBGPU_BACKEND.capabilities(0).unwrap(),
        CapabilityFlags::FP32
    );
    unbind_adapter();
}
