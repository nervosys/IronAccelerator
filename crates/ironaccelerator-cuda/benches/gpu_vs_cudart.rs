//! GPU-bound overhead benches: IronAccelerator wrappers vs raw driver-API.
//!
//! "cudart" here means the NVIDIA driver/runtime call underneath each helper.
//! Since IronAccelerator does not link libcudart, we compare against the
//! lowest-level thing available — direct `iron_cuda_sys::driver` fn-pointer
//! calls. That isolates pure wrapper cost from driver/runtime cost.
//!
//! Requires a working CUDA install + at least one GPU. If `Device::open(0)`
//! fails at startup, the bench prints a notice and exits cleanly.
//!
//! Each op is benched two ways:
//!
//!   * `wrapped` — through `ironaccelerator_cuda::drv`
//!   * `raw`     — through `iron_cuda_sys::driver::fns()` directly
//!
//! Same GPU, same primary context. Any delta is wrapper overhead.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use iron_cuda_sys::driver as sys;
use ironaccelerator_cuda::drv::{Device, DeviceBuf, Event, Stream};
use std::ffi::c_void;
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
// Setup — shared across every bench so we measure steady-state overhead.
// ═══════════════════════════════════════════════════════════════════════════

struct Ctx {
    device: Arc<Device>,
    stream: Arc<Stream>,
    raw_stream: sys::CUstream,
    fns: &'static sys::DriverFns,
}

fn try_init() -> Option<Ctx> {
    let device = match Device::open(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Device::open(0) failed: {e}");
            return None;
        }
    };
    if let Err(e) = device.bind() {
        eprintln!("bind failed: {e}");
        return None;
    }
    let stream = match Stream::new(device.clone()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Stream::new failed: {e}");
            return None;
        }
    };
    let fns = match sys::fns() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("sys::fns failed: {e}");
            return None;
        }
    };
    let mut raw_stream = sys::CUstream::default();
    unsafe {
        let r = (fns.cuStreamCreateWithPriority)(&mut raw_stream, sys::CU_STREAM_NON_BLOCKING, 0);
        if !r.is_ok() {
            eprintln!("raw cuStreamCreate failed: {r:?}");
            return None;
        }
    }
    Some(Ctx {
        device,
        stream,
        raw_stream,
        fns,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Stream lifecycle
// ═══════════════════════════════════════════════════════════════════════════

fn bench_stream_lifecycle(c: &mut Criterion, ctx: &Ctx) {
    let mut g = c.benchmark_group("stream/create_destroy");

    g.bench_function("wrapped", |b| {
        b.iter(|| {
            let s = Stream::new(ctx.device.clone()).unwrap();
            black_box(&s);
        });
    });

    g.bench_function("raw", |b| {
        b.iter(|| {
            let mut s = sys::CUstream::default();
            unsafe {
                (ctx.fns.cuStreamCreateWithPriority)(&mut s, sys::CU_STREAM_NON_BLOCKING, 0)
                    .ok()
                    .unwrap();
                (ctx.fns.cuStreamDestroy_v2)(s).ok().unwrap();
            }
            black_box(s);
        });
    });

    g.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// Stream synchronize — empty sync (no pending work)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_stream_sync(c: &mut Criterion, ctx: &Ctx) {
    let mut g = c.benchmark_group("stream/synchronize_empty");

    g.bench_function("wrapped", |b| {
        b.iter(|| ctx.stream.synchronize().unwrap());
    });

    g.bench_function("raw", |b| {
        b.iter(|| unsafe {
            (ctx.fns.cuStreamSynchronize)(ctx.raw_stream).ok().unwrap();
        });
    });

    g.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// Event lifecycle: create + record + sync + destroy
// ═══════════════════════════════════════════════════════════════════════════

fn bench_event_lifecycle(c: &mut Criterion, ctx: &Ctx) {
    let mut g = c.benchmark_group("event/create_record_sync_destroy");

    g.bench_function("wrapped", |b| {
        b.iter(|| {
            let e = Event::new(ctx.device.clone()).unwrap();
            e.record(&ctx.stream).unwrap();
            e.synchronize().unwrap();
            black_box(&e);
        });
    });

    g.bench_function("raw", |b| {
        b.iter(|| unsafe {
            let mut e = sys::CUevent::default();
            (ctx.fns.cuEventCreate)(&mut e, sys::CUevent_flags::DisableTiming as u32)
                .ok()
                .unwrap();
            (ctx.fns.cuEventRecord)(e, ctx.raw_stream).ok().unwrap();
            (ctx.fns.cuEventSynchronize)(e).ok().unwrap();
            (ctx.fns.cuEventDestroy_v2)(e).ok().unwrap();
            black_box(e);
        });
    });

    g.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// Alloc + free (async) across sizes. End-to-end with sync so allocations
// actually complete on the pool before the next iter.
// ═══════════════════════════════════════════════════════════════════════════

const SIZES: &[(usize, &str)] = &[
    (1 << 10, "1KB"),
    (64 << 10, "64KB"),
    (1 << 20, "1MB"),
    (16 << 20, "16MB"),
    (256 << 20, "256MB"),
];

fn bench_alloc_free(c: &mut Criterion, ctx: &Ctx) {
    let mut g = c.benchmark_group("alloc/async_alloc_free");

    for (bytes, label) in SIZES {
        let n = *bytes / std::mem::size_of::<u8>();

        g.bench_with_input(BenchmarkId::new("wrapped", label), &n, |b, &n| {
            b.iter(|| {
                let buf: DeviceBuf<u8> = DeviceBuf::alloc(ctx.stream.clone(), n).unwrap();
                // Drop = cuMemFreeAsync on wrapper.
                drop(buf);
            });
            ctx.stream.synchronize().unwrap();
        });

        g.bench_with_input(BenchmarkId::new("raw", label), &n, |b, &n| {
            b.iter(|| unsafe {
                let mut p: sys::CUdeviceptr = 0;
                (ctx.fns.cuMemAllocAsync)(&mut p, n, ctx.raw_stream)
                    .ok()
                    .unwrap();
                (ctx.fns.cuMemFreeAsync)(p, ctx.raw_stream).ok().unwrap();
            });
            unsafe {
                (ctx.fns.cuStreamSynchronize)(ctx.raw_stream).ok().unwrap();
            }
        });
    }

    g.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// Memset async — enqueue latency (no sync per-iter so we measure API cost,
// not GPU work). We sync once per batch at teardown.
// ═══════════════════════════════════════════════════════════════════════════

fn bench_memset_enqueue(c: &mut Criterion, ctx: &Ctx) {
    // Pre-allocate one big buffer for all sizes.
    let buf: DeviceBuf<u8> = DeviceBuf::alloc(ctx.stream.clone(), 256 << 20).unwrap();
    ctx.stream.synchronize().unwrap();
    let raw_ptr = buf.view().device_ptr();

    let mut g = c.benchmark_group("memset/async_enqueue");
    for (bytes, label) in SIZES {
        g.throughput(Throughput::Bytes(*bytes as u64));

        g.bench_with_input(BenchmarkId::new("wrapped", label), bytes, |b, &n| {
            b.iter(|| unsafe {
                (ctx.fns.cuMemsetD8Async)(raw_ptr, 0xab, n, ctx.stream.raw())
                    .ok()
                    .unwrap();
            });
            ctx.stream.synchronize().unwrap();
        });

        g.bench_with_input(BenchmarkId::new("raw", label), bytes, |b, &n| {
            b.iter(|| unsafe {
                (ctx.fns.cuMemsetD8Async)(raw_ptr, 0xab, n, ctx.raw_stream)
                    .ok()
                    .unwrap();
            });
            unsafe {
                (ctx.fns.cuStreamSynchronize)(ctx.raw_stream).ok().unwrap();
            }
        });
    }
    g.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// H2D memcpy — async enqueue + sync per iter (to keep queue depth bounded).
// This is the round-trip latency most apps actually see.
// ═══════════════════════════════════════════════════════════════════════════

fn bench_memcpy_h2d(c: &mut Criterion, ctx: &Ctx) {
    let mut g = c.benchmark_group("memcpy/h2d_roundtrip");

    for (bytes, label) in SIZES {
        let n = *bytes;
        let host = vec![0u8; n];
        let mut buf: DeviceBuf<u8> = DeviceBuf::alloc(ctx.stream.clone(), n).unwrap();
        ctx.stream.synchronize().unwrap();
        g.throughput(Throughput::Bytes(n as u64));

        g.bench_with_input(BenchmarkId::new("wrapped", label), &n, |b, _| {
            b.iter(|| {
                buf.copy_from_host(&host).unwrap();
                ctx.stream.synchronize().unwrap();
            });
        });

        // Raw path: same ptr, same stream, just skip all the wrapper checks.
        let raw_ptr = buf.view().device_ptr();
        g.bench_with_input(BenchmarkId::new("raw", label), &n, |b, _| {
            b.iter(|| unsafe {
                (ctx.fns.cuMemcpyHtoDAsync_v2)(
                    raw_ptr,
                    host.as_ptr() as *const c_void,
                    n,
                    ctx.raw_stream,
                )
                .ok()
                .unwrap();
                (ctx.fns.cuStreamSynchronize)(ctx.raw_stream).ok().unwrap();
            });
        });
    }

    g.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// D2D memcpy — same pattern.
// ═══════════════════════════════════════════════════════════════════════════

fn bench_memcpy_d2d(c: &mut Criterion, ctx: &Ctx) {
    let mut g = c.benchmark_group("memcpy/d2d_roundtrip");

    for (bytes, label) in SIZES {
        let n = *bytes;
        let mut dst: DeviceBuf<u8> = DeviceBuf::alloc(ctx.stream.clone(), n).unwrap();
        let src: DeviceBuf<u8> = DeviceBuf::alloc(ctx.stream.clone(), n).unwrap();
        ctx.stream.synchronize().unwrap();
        g.throughput(Throughput::Bytes(n as u64));

        g.bench_with_input(BenchmarkId::new("wrapped", label), &n, |b, _| {
            b.iter(|| {
                dst.copy_from_device(&src).unwrap();
                ctx.stream.synchronize().unwrap();
            });
        });

        let d_ptr = dst.view().device_ptr();
        let s_ptr = src.view().device_ptr();
        g.bench_with_input(BenchmarkId::new("raw", label), &n, |b, _| {
            b.iter(|| unsafe {
                (ctx.fns.cuMemcpyDtoDAsync_v2)(d_ptr, s_ptr, n, ctx.raw_stream)
                    .ok()
                    .unwrap();
                (ctx.fns.cuStreamSynchronize)(ctx.raw_stream).ok().unwrap();
            });
        });
    }

    g.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry point
// ═══════════════════════════════════════════════════════════════════════════

fn all(c: &mut Criterion) {
    let Some(ctx) = try_init() else {
        eprintln!("gpu_vs_cudart: no CUDA device available — skipping.");
        return;
    };
    eprintln!(
        "gpu_vs_cudart: device = {}",
        ctx.device.name().unwrap_or_else(|_| "?".into())
    );

    bench_stream_lifecycle(c, &ctx);
    bench_stream_sync(c, &ctx);
    bench_event_lifecycle(c, &ctx);
    bench_alloc_free(c, &ctx);
    bench_memset_enqueue(c, &ctx);
    bench_memcpy_h2d(c, &ctx);
    bench_memcpy_d2d(c, &ctx);

    // Release the raw stream cleanly.
    unsafe {
        let _ = (ctx.fns.cuStreamDestroy_v2)(ctx.raw_stream);
    }
}

criterion_group!(benches, all);
criterion_main!(benches);
