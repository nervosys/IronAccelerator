//! Mempool retain-on-free microbench.
//!
//! Validates the perf win from setting `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD =
//! u64::MAX` on the device's default stream-ordered memory pool.
//!
//! Without retention (threshold=0, the driver default), every `cuMemFreeAsync`
//! returns memory to the OS at the next sync, costing page-mapping on the
//! next `cuMemAllocAsync`. With retention, the pool reuses freed memory.
//!
//! `Device::open` auto-sets MAX. This bench flips it manually to compare.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use iron_cuda_sys::driver as sys;
use ironaccelerator_cuda::drv::{Device, DeviceBuf, Stream};
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Duration;

struct Ctx {
    device: Arc<Device>,
    stream: Arc<Stream>,
    pool: sys::CUmemPool,
    fns: &'static sys::DriverFns,
}

fn try_init() -> Option<Ctx> {
    let device = Device::open(0).ok()?;
    device.bind().ok()?;
    let stream = Stream::new(device.clone()).ok()?;
    let fns = sys::fns().ok()?;
    let mut pool = sys::CUmemPool::default();
    unsafe {
        if (fns.cuDeviceGetDefaultMemPool)(&mut pool, device.raw_device())
            != sys::CUresult::Success
        {
            return None;
        }
    }
    Some(Ctx { device, stream, pool, fns })
}

fn set_threshold(ctx: &Ctx, threshold: u64) {
    let mut v = threshold;
    unsafe {
        let _ = (ctx.fns.cuMemPoolSetAttribute)(
            ctx.pool,
            sys::CUmemPool_attribute::ReleaseThreshold,
            &mut v as *mut u64 as *mut c_void,
        );
    }
}

/// One round: alloc N buffers of `bytes_each`, sync, drop them (frees on
/// stream), sync. With threshold=0 the post-sync free returns memory to
/// the OS; with threshold=MAX it stays in the pool.
fn alloc_free_round(ctx: &Ctx, count: usize, elems_each: usize) {
    let mut bufs: Vec<DeviceBuf<u8>> = Vec::with_capacity(count);
    for _ in 0..count {
        bufs.push(DeviceBuf::alloc(ctx.stream.clone(), elems_each).expect("alloc"));
    }
    ctx.stream.synchronize().expect("sync");
    drop(bufs);
    ctx.stream.synchronize().expect("sync");
}

fn bench_mempool(c: &mut Criterion) {
    let Some(ctx) = try_init() else {
        eprintln!("CUDA device 0 unavailable, skipping mempool bench");
        return;
    };

    let mut g = c.benchmark_group("mempool_retain_vs_release");
    g.measurement_time(Duration::from_secs(8));
    g.sample_size(40);

    // Sizes chosen to span (a) small per-token KV scratch and (b) larger
    // intermediate activations. 256 KiB is a typical RMSNorm scratch; 4 MiB
    // approximates a per-layer hidden-state slab for 1B-class models.
    for &(label, count, bytes_each) in &[
        ("16x256KiB", 16usize, 256 * 1024usize),
        ("16x4MiB", 16usize, 4 * 1024 * 1024usize),
        ("64x256KiB", 64usize, 256 * 1024usize),
    ] {
        // ── retain-on-free (IronAccelerator's default after Device::open)
        set_threshold(&ctx, u64::MAX);
        // warm-up so the pool is populated.
        alloc_free_round(&ctx, count, bytes_each);
        g.bench_with_input(
            BenchmarkId::new("retain", label),
            &(count, bytes_each),
            |b, &(c_, b_)| {
                b.iter(|| {
                    alloc_free_round(&ctx, c_, b_);
                    black_box(());
                });
            },
        );

        // ── default-release (threshold=0, driver default before our change)
        set_threshold(&ctx, 0);
        // Force one round to sync the pool back to OS.
        alloc_free_round(&ctx, count, bytes_each);
        g.bench_with_input(
            BenchmarkId::new("release", label),
            &(count, bytes_each),
            |b, &(c_, b_)| {
                b.iter(|| {
                    alloc_free_round(&ctx, c_, b_);
                    black_box(());
                });
            },
        );
    }

    // Restore retention so any later benches in the same binary aren't penalised.
    set_threshold(&ctx, u64::MAX);
    g.finish();
}

criterion_group!(benches, bench_mempool);
criterion_main!(benches);
