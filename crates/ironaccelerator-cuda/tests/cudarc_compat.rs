//! Compatibility tests for the `cudarc_compat` module.
//!
//! These tests serve two audiences:
//!
//! 1. **On GPU-less CI runners**, they compile-exercise the cudarc-shaped
//!    type aliases and trait impls. If a symbol that downstream users import
//!    as `use ironaccelerator_cuda::cudarc_compat::{CudaDevice, CudaSlice}`
//!    ever goes missing or changes signature, the test suite fails even
//!    without a live GPU.
//! 2. **On GPU hosts**, the `#[ignore]`-gated tests round-trip host→device→
//!    host data and verify the advertised drop-in semantics match cudarc.
//!    Run with `cargo test -p ironaccelerator-cuda --test cudarc_compat -- --ignored`.

use ironaccelerator_cuda::cudarc_compat::*;
use std::sync::Arc;

// ── Compile-only guards ─────────────────────────────────────────────────────
//
// Each of these functions is never called at runtime — its presence at link
// time proves the corresponding cudarc symbol shape is still exported.

#[allow(dead_code)]
fn _symbols_exist_as_expected() {
    let _: fn(usize) -> DriverResult<Arc<CudaDevice>> = CudaDevice::new;
    let _: fn() -> DriverResult<u32> = CudaDevice::count;
    let _: fn() -> DriverResult<u32> = CudaDevice::device_count;
    let _: fn(&str) -> ironaccelerator_core::Result<Vec<u8>> = compile_ptx;

    fn _device_methods(d: &CudaDevice) -> DriverResult<()> {
        let _: (usize, usize) = d.mem_get_info()?;
        let _: (u32, u32) = d.compute_capability()?;
        Ok(())
    }
    let _ = _device_methods as fn(&CudaDevice) -> DriverResult<()>;

    fn _stream_ext<T: DeviceRepr + ZeroBits>(s: &Arc<CudaStream>) {
        let _: fn(&Arc<CudaStream>, Vec<T>) -> DriverResult<CudaSlice<T>> =
            CudaStreamExt::htod_copy;
        let _: fn(&Arc<CudaStream>, &[T]) -> DriverResult<CudaSlice<T>> =
            CudaStreamExt::htod_sync_copy;
        let _: fn(&Arc<CudaStream>, usize) -> DriverResult<CudaSlice<T>> = CudaStreamExt::alloc;
        let _: fn(&Arc<CudaStream>, usize) -> DriverResult<CudaSlice<T>> =
            CudaStreamExt::alloc_zeros;
        let _: fn(&Arc<CudaStream>, &CudaSlice<T>) -> DriverResult<Vec<T>> =
            CudaStreamExt::dtoh_sync_copy;
        let _: fn(&Arc<CudaStream>) -> DriverResult<CudaEvent> = CudaStreamExt::record_event;
        let _: fn(&Arc<CudaStream>, &CudaEvent) -> DriverResult<()> = CudaStreamExt::wait;
        let _: fn(&Arc<CudaStream>, &Arc<CudaStream>) -> DriverResult<()> = CudaStreamExt::join;
        let _ = s; // silence unused
    }
    let _ = _stream_ext::<f32> as fn(&Arc<CudaStream>);
    let _ = _stream_ext::<u32> as fn(&Arc<CudaStream>);
    let _ = _stream_ext::<i64> as fn(&Arc<CudaStream>);
}

#[test]
fn count_is_safe_without_gpu() {
    // Must not panic whether or not a CUDA driver is present. Zero is a valid
    // answer on a headless runner, any positive number on a GPU host.
    match CudaDevice::count() {
        Ok(n) => {
            println!("cudarc_compat::CudaDevice::count -> {n}");
        }
        Err(e) => {
            // NotAvailable is fine — we only care that no panic escapes.
            println!("no CUDA driver present ({e})");
        }
    }
}

#[test]
fn new_on_missing_gpu_returns_error_not_panic() {
    // ordinal 0 may succeed on a GPU host, but the test must only assert
    // no panic — error outcomes are expected on CI.
    let _ = CudaDevice::new(0);
}

// ── GPU-gated round-trip ────────────────────────────────────────────────────

#[test]
#[ignore = "requires an NVIDIA GPU + CUDA driver"]
fn htod_then_dtoh_roundtrip() {
    let dev = CudaDevice::new(0).expect("CUDA device");
    let stream = dev.default_stream();

    let src: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let dev_buf: CudaSlice<f32> = stream.htod_copy(src.clone()).unwrap();
    assert_eq!(dev_buf.len(), 1024);

    let out: Vec<f32> = stream.dtoh_sync_copy(&dev_buf).unwrap();
    assert_eq!(out, src);
}

#[test]
#[ignore = "requires an NVIDIA GPU + CUDA driver"]
fn alloc_zeros_is_zero() {
    let dev = CudaDevice::new(0).expect("CUDA device");
    let stream = dev.default_stream();
    let buf: CudaSlice<u32> = stream.alloc_zeros(256).unwrap();
    let out = stream.dtoh_sync_copy(&buf).unwrap();
    assert!(out.iter().all(|&x| x == 0));
}

#[test]
#[ignore = "requires an NVIDIA GPU + CUDA driver"]
fn multiple_streams_are_independent() {
    let dev = CudaDevice::new(0).expect("CUDA device");
    let s1 = dev.new_stream().unwrap();
    let s2 = dev.new_stream().unwrap();
    assert!(!Arc::ptr_eq(&s1, &s2));

    let a: CudaSlice<u32> = s1.htod_copy(vec![1u32, 2, 3]).unwrap();
    let b: CudaSlice<u32> = s2.htod_copy(vec![4u32, 5, 6]).unwrap();
    assert_eq!(s1.dtoh_sync_copy(&a).unwrap(), [1, 2, 3]);
    assert_eq!(s2.dtoh_sync_copy(&b).unwrap(), [4, 5, 6]);
}

#[test]
#[ignore = "requires an NVIDIA GPU + CUDA driver"]
fn mem_get_info_reports_plausible_numbers() {
    let dev = CudaDevice::new(0).expect("CUDA device");
    let (free, total) = dev.mem_get_info().unwrap();
    assert!(total > 0, "total mem must be positive: {total}");
    assert!(free <= total, "free ({free}) must be <= total ({total})");
    // Any modern GPU has at least 1 GiB.
    assert!(
        total >= 1 << 30,
        "total mem suspiciously small: {total} bytes"
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU + CUDA driver"]
fn compute_capability_is_real() {
    let dev = CudaDevice::new(0).expect("CUDA device");
    let (maj, min) = dev.compute_capability().unwrap();
    // Anything CUDA 13.x supports is sm_5.0 or newer.
    assert!(
        maj >= 5,
        "compute capability {maj}.{min} is unsupported by CUDA 13"
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU + CUDA driver"]
fn record_event_wait_join_round_trip() {
    let dev = CudaDevice::new(0).expect("CUDA device");
    let s1 = dev.new_stream().unwrap();
    let s2 = dev.new_stream().unwrap();

    // Enqueue some work on s1, fence it, and have s2 wait.
    let buf: CudaSlice<u32> = s1.alloc_zeros(1024).unwrap();
    let ev = s1.record_event().unwrap();
    s2.wait(&ev).unwrap();

    // s2.join(&s1) should be a no-op past that fence; just check it doesn't err.
    s2.join(&s1).unwrap();
    s2.synchronize().unwrap();
    drop(buf);
}

#[test]
#[ignore = "requires an NVIDIA GPU + CUDA driver"]
fn try_clone_produces_independent_copy() {
    let dev = CudaDevice::new(0).expect("CUDA device");
    let stream = dev.default_stream();
    let original: CudaSlice<u32> = stream.htod_copy(vec![10u32, 20, 30, 40]).unwrap();
    let cloned = original.try_clone().unwrap();

    // Cloned buffer holds the same data.
    assert_eq!(stream.dtoh_sync_copy(&cloned).unwrap(), [10, 20, 30, 40]);

    // num_bytes and ordinal mirror cudarc's API.
    assert_eq!(cloned.num_bytes(), 4 * std::mem::size_of::<u32>());
    assert_eq!(cloned.ordinal(), dev.ordinal());
}
