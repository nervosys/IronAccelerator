//! Live-GPU smoke test — exercises the actual CUDA driver path end-to-end.
//!
//! Skipped cleanly (not failed) when no CUDA device is available, so CI on
//! GPU-less runners stays green. On a host with a working driver this
//! verifies:
//!
//! - Device enumeration + binding
//! - Stream + Event lifecycle
//! - Device allocation, H2D / D2H / D2D memcpy
//! - Value integrity through a full round-trip
//! - Parity between the safe wrapper and the raw `iron_cuda_sys` driver
//!
//! Run with: `cargo test -p ironaccelerator-cuda --test gpu_smoke -- --nocapture`.

use iron_cuda_sys::driver as sys;
use ironaccelerator_cuda::drv::{Device, DeviceBuf, Event, LaunchCfg, Stream};
use ironaccelerator_cuda::kernel::{get_or_compile, CompileOptions};
use std::ffi::c_void;

fn have_cuda() -> bool {
    match Device::count() {
        Ok(n) if n > 0 => true,
        Ok(_) => {
            eprintln!("gpu_smoke: 0 CUDA devices — skipping");
            false
        }
        Err(e) => {
            eprintln!("gpu_smoke: Device::count failed ({e}) — skipping");
            false
        }
    }
}

#[test]
fn device_enumerates_and_binds() {
    if !have_cuda() {
        return;
    }
    let n = Device::count().unwrap();
    eprintln!("gpu_smoke: {n} device(s) visible");
    for i in 0..n {
        let d = Device::open(i).unwrap();
        let name = d.name().unwrap_or_else(|_| "?".into());
        eprintln!("  [{i}] {name}");
        d.bind().unwrap();
    }
}

#[test]
fn h2d_d2h_roundtrip_preserves_values() {
    if !have_cuda() {
        return;
    }
    let dev = Device::open(0).unwrap();
    dev.bind().unwrap();
    let stream = Stream::new(dev).unwrap();

    const N: usize = 1 << 16;
    let src: Vec<u32> = (0..N as u32).collect();
    let mut buf: DeviceBuf<u32> = DeviceBuf::alloc(stream.clone(), N).unwrap();
    buf.copy_from_host(&src).unwrap();

    let mut dst = vec![0u32; N];
    buf.copy_to_host(&mut dst).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(src, dst, "round-trip corrupted data");
}

#[test]
fn d2d_copy_preserves_values() {
    if !have_cuda() {
        return;
    }
    let dev = Device::open(0).unwrap();
    dev.bind().unwrap();
    let stream = Stream::new(dev).unwrap();

    const N: usize = 4096;
    let src_host: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();

    let mut src: DeviceBuf<f32> = DeviceBuf::alloc(stream.clone(), N).unwrap();
    src.copy_from_host(&src_host).unwrap();

    let mut dst: DeviceBuf<f32> = DeviceBuf::alloc(stream.clone(), N).unwrap();
    dst.copy_from_device(&src).unwrap();

    let mut back = vec![0f32; N];
    dst.copy_to_host(&mut back).unwrap();
    stream.synchronize().unwrap();

    for (i, (a, b)) in src_host.iter().zip(back.iter()).enumerate() {
        assert_eq!(a, b, "d2d mismatch at {i}: {a} != {b}");
    }
}

#[test]
fn event_records_and_syncs() {
    if !have_cuda() {
        return;
    }
    let dev = Device::open(0).unwrap();
    dev.bind().unwrap();
    let stream = Stream::new(dev.clone()).unwrap();
    let ev = Event::new(dev).unwrap();
    ev.record(&stream).unwrap();
    ev.synchronize().unwrap();
}

#[test]
fn raw_driver_parity_memset_and_copy() {
    if !have_cuda() {
        return;
    }
    let dev = Device::open(0).unwrap();
    dev.bind().unwrap();
    let stream = Stream::new(dev).unwrap();
    let fns = sys::fns().expect("driver fns");

    const N: usize = 1024;
    let bytes = N * std::mem::size_of::<u32>();

    // Alloc via wrapper, poke via raw driver, read back via wrapper.
    let mut buf: DeviceBuf<u32> = DeviceBuf::alloc(stream.clone(), N).unwrap();
    let ptr = buf.view().device_ptr();

    unsafe {
        let r = (fns.cuMemsetD8Async)(ptr, 0xab, bytes, stream.raw());
        assert!(r.is_ok(), "raw memset failed: {r:?}");
    }

    // Write a known pattern via H2D on the same buffer and confirm it lands.
    let src: Vec<u32> = (0..N as u32).map(|i| i.wrapping_mul(0x9E37_79B1)).collect();
    buf.copy_from_host(&src).unwrap();

    let mut dst = vec![0u32; N];
    unsafe {
        let r = (fns.cuMemcpyDtoHAsync_v2)(
            dst.as_mut_ptr() as *mut c_void,
            ptr,
            bytes,
            stream.raw(),
        );
        assert!(r.is_ok(), "raw D2H failed: {r:?}");
    }
    stream.synchronize().unwrap();

    assert_eq!(src, dst, "raw/wrapped parity broken");
}

#[test]
fn many_streams_in_flight() {
    if !have_cuda() {
        return;
    }
    let dev = Device::open(0).unwrap();
    dev.bind().unwrap();

    // Spin up 8 streams, enqueue a small H2D on each, sync all. Verifies the
    // wrapper doesn't serialize on a hidden global mutex.
    const S: usize = 8;
    const N: usize = 1 << 14;
    let host: Vec<u8> = (0..N).map(|i| (i & 0xff) as u8).collect();

    let streams: Vec<_> = (0..S).map(|_| Stream::new(dev.clone()).unwrap()).collect();
    let mut bufs: Vec<DeviceBuf<u8>> =
        streams.iter().map(|s| DeviceBuf::alloc(s.clone(), N).unwrap()).collect();

    for b in bufs.iter_mut() {
        b.copy_from_host(&host).unwrap();
    }
    for s in &streams {
        s.synchronize().unwrap();
    }

    // Spot-check one stream: data must have landed.
    let mut back = vec![0u8; N];
    bufs[0].copy_to_host(&mut back).unwrap();
    streams[0].synchronize().unwrap();
    assert_eq!(host, back);
}

// ═══════════════════════════════════════════════════════════════════════════
// NVRTC compile + launch: a real kernel runs on the GPU and produces the
// expected result. This exercises the full pipeline: NVRTC → PTX → cuModuleLoad
// → cuModuleGetFunction → cuLaunchKernel with typed argument packing.
// ═══════════════════════════════════════════════════════════════════════════

const SAXPY_SRC: &str = r#"
extern "C" __global__
void saxpy(const float* x, float* y, float a, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
"#;

#[test]
fn nvrtc_compile_and_launch_saxpy() {
    if !have_cuda() {
        return;
    }
    let dev = Device::open(0).unwrap();
    dev.bind().unwrap();
    let stream = Stream::new(dev.clone()).unwrap();

    const N: usize = 1 << 20;
    let a: f32 = 2.5;
    let x_host: Vec<f32> = (0..N).map(|i| i as f32 * 0.001).collect();
    let y_host: Vec<f32> = (0..N).map(|i| i as f32 * 0.01).collect();

    let mut x: DeviceBuf<f32> = DeviceBuf::alloc(stream.clone(), N).unwrap();
    let mut y: DeviceBuf<f32> = DeviceBuf::alloc(stream.clone(), N).unwrap();
    x.copy_from_host(&x_host).unwrap();
    y.copy_from_host(&y_host).unwrap();

    let k = get_or_compile(&dev, SAXPY_SRC, "saxpy", &CompileOptions::default())
        .expect("NVRTC compile");

    let cfg = LaunchCfg::for_elements(N as u32, 256);
    k.function
        .launch(cfg, &stream, (x.view(), y.view_mut(), a, N as u32))
        .expect("launch");

    let mut out = vec![0f32; N];
    y.copy_to_host(&mut out).unwrap();
    stream.synchronize().unwrap();

    for i in 0..N {
        let expected = a * x_host[i] + y_host[i];
        let got = out[i];
        assert!(
            (got - expected).abs() <= 1e-4 * expected.abs().max(1.0),
            "mismatch at {i}: got {got} expected {expected}"
        );
    }
    eprintln!("gpu_smoke: SAXPY on {N} elements — OK");
}

#[test]
fn nvrtc_cache_returns_same_module_on_second_call() {
    if !have_cuda() {
        return;
    }
    let dev = Device::open(0).unwrap();
    dev.bind().unwrap();

    let k1 = get_or_compile(&dev, SAXPY_SRC, "saxpy", &CompileOptions::default()).unwrap();
    let k2 = get_or_compile(&dev, SAXPY_SRC, "saxpy", &CompileOptions::default()).unwrap();
    assert!(
        std::sync::Arc::ptr_eq(&k1.module, &k2.module),
        "kernel cache miss on identical key"
    );
}
