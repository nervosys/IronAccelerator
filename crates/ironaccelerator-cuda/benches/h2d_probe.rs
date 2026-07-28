//! Diagnostic probe for the H2D path. Isolates allocation from copy, and
//! pageable sources from pinned ones, so a regression can be attributed.
//!
//! Not part of the published comparison set — this exists to answer "where did
//! the time go" when `vs_cudarc/memcpy/h2d_roundtrip` moves.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ironaccelerator_cuda::cudarc_compat as iron;
use ironaccelerator_cuda::cudarc_compat::CudaStreamExt;
use ironaccelerator_cuda::drv;
use ironaccelerator_cuda::sys::driver as sys;
use std::ffi::c_void;
use std::sync::Arc;

const SIZES: &[(usize, &str)] = &[(1 << 20, "1MB"), (16 << 20, "16MB")];

struct Ctx {
    iron_stream: Arc<iron::CudaStream>,
    lo_device: Arc<drv::Device>,
    lo_stream: Arc<drv::Stream>,
    cudarc_stream: Arc<cudarc::driver::CudaStream>,
    fns: &'static sys::DriverFns,
}

fn try_init() -> Option<Ctx> {
    let iron_dev = iron::CudaDevice::new(0).ok()?;
    let iron_stream = iron_dev.default_stream();
    let lo_device = drv::Device::open(0).ok()?;
    lo_device.bind().ok()?;
    let lo_stream = drv::Stream::new(lo_device.clone()).ok()?;
    let cudarc_ctx = cudarc::driver::CudaContext::new(0).ok()?;
    let cudarc_stream = cudarc_ctx.default_stream();
    let fns = sys::fns().ok()?;
    Some(Ctx {
        iron_stream,
        lo_device,
        lo_stream,
        cudarc_stream,
        fns,
    })
}

fn bench(c: &mut Criterion) {
    let Some(ctx) = try_init() else {
        eprintln!("skipped: no CUDA device");
        return;
    };

    for (bytes, label) in SIZES {
        let n = *bytes;
        let host = vec![0u8; n];
        let mut g = c.benchmark_group(format!("h2d_probe/{label}"));
        g.throughput(Throughput::Bytes(n as u64));

        // Full round-trip, as the published bench measures it.
        g.bench_with_input(BenchmarkId::new("ia_alloc_copy_sync", label), &n, |b, _| {
            b.iter(|| {
                let buf = ctx.iron_stream.htod_sync_copy(&host).unwrap();
                black_box(&buf);
            });
        });
        g.bench_with_input(
            BenchmarkId::new("cudarc_alloc_copy_sync", label),
            &n,
            |b, _| {
                b.iter(|| {
                    let buf = ctx.cudarc_stream.clone_htod(&host).unwrap();
                    ctx.cudarc_stream.synchronize().unwrap();
                    black_box(&buf);
                });
            },
        );

        // Copy only, into a buffer allocated once.
        let mut ia_dst: drv::DeviceBuf<u8> =
            drv::DeviceBuf::alloc(ctx.lo_stream.clone(), n).unwrap();
        ctx.lo_stream.synchronize().unwrap();
        g.bench_with_input(
            BenchmarkId::new("ia_copy_only_pageable", label),
            &n,
            |b, _| {
                b.iter(|| {
                    ia_dst.copy_from_host(&host).unwrap();
                    ctx.lo_stream.synchronize().unwrap();
                });
            },
        );

        // Same copy, but issued as the blocking driver entry point.
        g.bench_with_input(
            BenchmarkId::new("ia_copy_only_blocking", label),
            &n,
            |b, _| {
                b.iter(|| unsafe {
                    let r = (ctx.fns.cuMemcpyHtoD_v2)(
                        ia_dst.device_ptr(),
                        host.as_ptr() as *const c_void,
                        n,
                    );
                    assert_eq!(r, sys::CUresult::Success);
                });
            },
        );

        // Pinned source, async on a real stream — true DMA, no driver staging.
        let mut pinned: *mut c_void = std::ptr::null_mut();
        unsafe {
            assert_eq!(
                (ctx.fns.cuMemHostAlloc)(&mut pinned, n, 0),
                sys::CUresult::Success
            );
            std::ptr::copy_nonoverlapping(host.as_ptr(), pinned as *mut u8, n);
        }
        g.bench_with_input(
            BenchmarkId::new("ia_copy_only_pinned", label),
            &n,
            |b, _| {
                b.iter(|| unsafe {
                    let r = (ctx.fns.cuMemcpyHtoDAsync_v2)(
                        ia_dst.device_ptr(),
                        pinned,
                        n,
                        ctx.lo_stream.raw(),
                    );
                    assert_eq!(r, sys::CUresult::Success);
                    ctx.lo_stream.synchronize().unwrap();
                });
            },
        );
        // Pinned source including the host-side staging memcpy, which is what
        // a staged implementation would actually have to pay.
        g.bench_with_input(
            BenchmarkId::new("ia_copy_staged_pinned", label),
            &n,
            |b, _| {
                b.iter(|| unsafe {
                    std::ptr::copy_nonoverlapping(host.as_ptr(), pinned as *mut u8, n);
                    let r = (ctx.fns.cuMemcpyHtoDAsync_v2)(
                        ia_dst.device_ptr(),
                        pinned,
                        n,
                        ctx.lo_stream.raw(),
                    );
                    assert_eq!(r, sys::CUresult::Success);
                    ctx.lo_stream.synchronize().unwrap();
                });
            },
        );

        // Allocation alone, to separate it from the copy.
        g.bench_with_input(BenchmarkId::new("ia_alloc_only", label), &n, |b, _| {
            b.iter(|| {
                let buf: drv::DeviceBuf<u8> =
                    drv::DeviceBuf::alloc(ctx.lo_stream.clone(), n).unwrap();
                black_box(buf.device_ptr());
            });
        });

        g.finish();
        unsafe {
            (ctx.fns.cuMemFreeHost)(pinned);
        }
        let _ = &ctx.lo_device;
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
