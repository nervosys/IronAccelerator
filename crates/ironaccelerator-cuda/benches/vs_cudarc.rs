//! Head-to-head benchmarks: IronAccelerator vs the `cudarc` crate.
//!
//! Both crates are thin Rust wrappers around the CUDA driver API. This bench
//! measures every operation an LLM serving stack runs at high frequency:
//! stream lifecycle, sync, alloc/free, H2D / D2H / D2D memcpy. Smaller is
//! better. The win we are looking for in IronAccelerator is that its
//! `cudarc_compat` surface matches cudarc on raw throughput while shaving
//! per-launch overhead on the hot path.
//!
//! Skipped cleanly on GPU-less runners.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cudarc::driver::PushKernelArg;
use ironaccelerator_cuda::cudarc_compat as iron;
use ironaccelerator_cuda::cudarc_compat::CudaStreamExt; // brings htod_copy / dtoh_sync_copy / alloc into scope
use std::sync::Arc;

// ─── Setup ──────────────────────────────────────────────────────────────────

struct Ctx {
    iron_dev: Arc<iron::CudaDevice>,
    iron_stream: Arc<iron::CudaStream>,
    cudarc_ctx: Arc<cudarc::driver::CudaContext>,
    cudarc_stream: Arc<cudarc::driver::CudaStream>,
}

fn try_init() -> Option<Ctx> {
    let iron_dev = match iron::CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("iron CudaDevice::new(0) failed: {e}");
            return None;
        }
    };
    let iron_stream = iron_dev.default_stream();

    let cudarc_ctx = match cudarc::driver::CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cudarc CudaContext::new(0) failed: {e:?}");
            return None;
        }
    };
    let cudarc_stream = cudarc_ctx.default_stream();

    Some(Ctx {
        iron_dev,
        iron_stream,
        cudarc_ctx,
        cudarc_stream,
    })
}

const SIZES: &[(usize, &str)] = &[
    (1 << 10, "1KB"),
    (64 << 10, "64KB"),
    (1 << 20, "1MB"),
    (16 << 20, "16MB"),
];

// ─── Stream lifecycle (create + drop) ───────────────────────────────────────

fn bench_stream_lifecycle(c: &mut Criterion, ctx: &Ctx) {
    let mut g = c.benchmark_group("vs_cudarc/stream/create_destroy");

    g.bench_function("ironaccelerator", |b| {
        b.iter(|| {
            let s = ctx.iron_dev.new_stream().unwrap();
            black_box(&s);
        });
    });

    g.bench_function("cudarc", |b| {
        b.iter(|| {
            let s = ctx.cudarc_ctx.new_stream().unwrap();
            black_box(&s);
        });
    });

    g.finish();
}

// ─── Kernel launch — the central hot path for any dispatch loop ────────────

const NOOP_KERNEL_SRC: &str = r#"
extern "C" __global__
void noop(const float* x, float* y, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
"#;

fn bench_kernel_launch(c: &mut Criterion, ctx: &Ctx) {
    // Pre-compile once for each runtime. We are measuring per-launch wrapper
    // overhead, not NVRTC compile time.
    use ironaccelerator_cuda::drv::{LaunchCfg, Module};
    use ironaccelerator_cuda::kernel::{compile, CompileOptions};

    let (maj, min) = ctx.iron_dev.raw().compute_capability().unwrap();
    let arch = format!("compute_{maj}{min}");
    let ptx = compile(NOOP_KERNEL_SRC, &arch, &CompileOptions::default()).unwrap();
    let iron_mod = Module::load(ctx.iron_dev.raw().clone(), &ptx).unwrap();
    let iron_fn = iron_mod.function("noop").unwrap();
    let iron_buf = ctx.iron_stream.alloc::<f32>(1024).unwrap();
    let mut iron_out = ctx.iron_stream.alloc::<f32>(1024).unwrap();
    let cfg = LaunchCfg::for_elements(1024, 256);

    // cudarc: same NVRTC PTX, loaded as a cudarc module.
    let ptx_for_cudarc = cudarc::nvrtc::Ptx::from(
        String::from_utf8_lossy(&ptx)
            .trim_end_matches('\0')
            .to_string(),
    );
    let cudarc_mod = ctx.cudarc_ctx.load_module(ptx_for_cudarc).unwrap();
    let cudarc_fn = cudarc_mod.load_function("noop").unwrap();
    let cudarc_in: cudarc::driver::CudaSlice<f32> =
        unsafe { ctx.cudarc_stream.alloc::<f32>(1024).unwrap() };
    let cudarc_out: cudarc::driver::CudaSlice<f32> =
        unsafe { ctx.cudarc_stream.alloc::<f32>(1024).unwrap() };

    let mut g = c.benchmark_group("vs_cudarc/launch/noop_1024");
    g.bench_function("ironaccelerator", |b| {
        b.iter(|| {
            iron_fn
                .launch(
                    cfg,
                    &ctx.iron_stream,
                    (iron_buf.view(), iron_out.view_mut(), 1024u32),
                )
                .unwrap();
        });
        ctx.iron_stream.synchronize().unwrap();
    });
    g.bench_function("cudarc", |b| {
        b.iter(|| {
            let mut lb = ctx.cudarc_stream.launch_builder(&cudarc_fn);
            lb.arg(&cudarc_in).arg(&cudarc_out).arg(&1024u32);
            unsafe {
                lb.launch(cudarc::driver::LaunchConfig {
                    grid_dim: (cfg.grid.0, cfg.grid.1, cfg.grid.2),
                    block_dim: (cfg.block.0, cfg.block.1, cfg.block.2),
                    shared_mem_bytes: cfg.shared_bytes,
                })
                .unwrap();
            }
        });
        ctx.cudarc_stream.synchronize().unwrap();
    });
    g.finish();
}

// ─── Event lifecycle (create + record + sync + destroy) ────────────────────

fn bench_event_lifecycle(c: &mut Criterion, ctx: &Ctx) {
    let mut g = c.benchmark_group("vs_cudarc/event/create_record_sync_destroy");

    g.bench_function("ironaccelerator", |b| {
        b.iter(|| {
            let e = iron::CudaEvent::new(ctx.iron_dev.raw().clone()).unwrap();
            e.record(&ctx.iron_stream).unwrap();
            e.synchronize().unwrap();
            black_box(&e);
        });
    });

    g.bench_function("cudarc", |b| {
        b.iter(|| {
            let e = ctx.cudarc_stream.record_event(None).unwrap();
            e.synchronize().unwrap();
            black_box(&e);
        });
    });

    g.finish();
}

// ─── Stream synchronize (empty queue) ───────────────────────────────────────

fn bench_stream_sync(c: &mut Criterion, ctx: &Ctx) {
    let mut g = c.benchmark_group("vs_cudarc/stream/synchronize_empty");

    g.bench_function("ironaccelerator", |b| {
        b.iter(|| ctx.iron_stream.synchronize().unwrap());
    });
    g.bench_function("cudarc", |b| {
        b.iter(|| ctx.cudarc_stream.synchronize().unwrap());
    });

    g.finish();
}

// ─── Pooled alloc + free — Iron's recycling fast path ──────────────────────

fn bench_pooled_alloc_free(c: &mut Criterion, ctx: &Ctx) {
    use ironaccelerator_cuda::pool::MemPool;
    let pool = MemPool::new(ctx.iron_stream.clone());

    let mut g = c.benchmark_group("vs_cudarc/alloc/pooled_alloc_free");
    for (bytes, label) in SIZES {
        let n = *bytes;
        g.bench_with_input(
            BenchmarkId::new("ironaccelerator_pool", label),
            &n,
            |b, &n| {
                b.iter(|| {
                    let buf = pool.alloc::<u8>(n).unwrap();
                    drop(buf);
                });
                ctx.iron_stream.synchronize().unwrap();
            },
        );
        g.bench_with_input(BenchmarkId::new("cudarc_no_pool", label), &n, |b, &n| {
            b.iter(|| {
                let buf: cudarc::driver::CudaSlice<u8> =
                    unsafe { ctx.cudarc_stream.alloc::<u8>(n).unwrap() };
                drop(buf);
            });
            ctx.cudarc_stream.synchronize().unwrap();
        });
    }
    g.finish();
}

// ─── Async alloc + free across sizes ────────────────────────────────────────

fn bench_alloc_free(c: &mut Criterion, ctx: &Ctx) {
    let mut g = c.benchmark_group("vs_cudarc/alloc/async_alloc_free");

    for (bytes, label) in SIZES {
        let n = *bytes;

        g.bench_with_input(BenchmarkId::new("ironaccelerator", label), &n, |b, &n| {
            b.iter(|| {
                let buf: iron::CudaSlice<u8> = ctx.iron_stream.alloc(n).unwrap();
                drop(buf);
            });
            ctx.iron_stream.synchronize().unwrap();
        });

        g.bench_with_input(BenchmarkId::new("cudarc", label), &n, |b, &n| {
            b.iter(|| {
                let buf: cudarc::driver::CudaSlice<u8> =
                    unsafe { ctx.cudarc_stream.alloc::<u8>(n).unwrap() };
                drop(buf);
            });
            ctx.cudarc_stream.synchronize().unwrap();
        });
    }
    g.finish();
}

// ─── H2D round-trip (memcpy + sync) ─────────────────────────────────────────

fn bench_h2d_roundtrip(c: &mut Criterion, ctx: &Ctx) {
    let mut g = c.benchmark_group("vs_cudarc/memcpy/h2d_roundtrip");

    for (bytes, label) in SIZES {
        let n = *bytes;
        let host = vec![0u8; n];
        g.throughput(Throughput::Bytes(n as u64));

        g.bench_with_input(BenchmarkId::new("ironaccelerator", label), &n, |b, _| {
            b.iter(|| {
                let buf = ctx.iron_stream.htod_sync_copy(&host).unwrap();
                black_box(&buf);
            });
        });
        g.bench_with_input(BenchmarkId::new("cudarc", label), &n, |b, _| {
            b.iter(|| {
                let buf = ctx.cudarc_stream.clone_htod(&host).unwrap();
                ctx.cudarc_stream.synchronize().unwrap();
                black_box(&buf);
            });
        });
    }
    g.finish();
}

// ─── D2H round-trip ─────────────────────────────────────────────────────────

fn bench_d2h_roundtrip(c: &mut Criterion, ctx: &Ctx) {
    let mut g = c.benchmark_group("vs_cudarc/memcpy/d2h_roundtrip");

    for (bytes, label) in SIZES {
        let n = *bytes;
        let host = vec![0u8; n];
        let iron_dev = ctx.iron_stream.htod_sync_copy(&host).unwrap();
        let cudarc_dev = ctx.cudarc_stream.clone_htod(&host).unwrap();
        ctx.iron_stream.synchronize().unwrap();
        ctx.cudarc_stream.synchronize().unwrap();
        g.throughput(Throughput::Bytes(n as u64));

        g.bench_with_input(BenchmarkId::new("ironaccelerator", label), &n, |b, _| {
            b.iter(|| {
                let out: Vec<u8> = ctx.iron_stream.dtoh_sync_copy(&iron_dev).unwrap();
                black_box(out);
            });
        });
        g.bench_with_input(BenchmarkId::new("cudarc", label), &n, |b, _| {
            b.iter(|| {
                let out: Vec<u8> = ctx.cudarc_stream.clone_dtoh(&cudarc_dev).unwrap();
                black_box(out);
            });
        });
    }
    g.finish();
}

// ─── Entry ──────────────────────────────────────────────────────────────────

fn all(c: &mut Criterion) {
    let Some(ctx) = try_init() else {
        eprintln!("vs_cudarc: no CUDA device — skipping.");
        return;
    };
    eprintln!(
        "vs_cudarc: ironaccelerator dev = {}",
        ctx.iron_dev.name().unwrap_or_else(|_| "?".into())
    );

    bench_stream_lifecycle(c, &ctx);
    bench_stream_sync(c, &ctx);
    bench_event_lifecycle(c, &ctx);
    bench_kernel_launch(c, &ctx);
    bench_alloc_free(c, &ctx);
    bench_pooled_alloc_free(c, &ctx);
    bench_h2d_roundtrip(c, &ctx);
    bench_d2h_roundtrip(c, &ctx);
}

criterion_group!(benches, all);
criterion_main!(benches);
