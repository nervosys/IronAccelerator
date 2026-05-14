//! Host-side hot-path benchmarks.
//!
//! These bench what IronAccelerator's wrappers do *around* every GPU call:
//! kernel-arg packing and the NVRTC source-cache hash. GPU-bound work
//! (actual memcpy/launch/throughput) lives in the live-GPU benches.
//!
//! Overhead budget: every path here should finish in **well under a
//! microsecond** so the wrapper layer never shows up in a kernel-launch
//! profile.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ironaccelerator_cuda::drv::LaunchArgs;

// ─── Kernel-arg packing ─────────────────────────────────────────────────────

fn bench_launch_args(c: &mut Criterion) {
    let mut g = c.benchmark_group("launch_args/pack");
    g.bench_function("tuple1_u64", |b| {
        let mut storage = <(u64,) as LaunchArgs>::storage();
        let mut ptrs = <(u64,) as LaunchArgs>::ptrs_init();
        b.iter(|| {
            let args = (black_box(42u64),);
            args.pack(&mut storage, &mut ptrs);
            black_box(&ptrs);
        });
    });
    g.bench_function("tuple4_mixed", |b| {
        type T = (u64, i32, f32, u32);
        let mut storage = <T as LaunchArgs>::storage();
        let mut ptrs = <T as LaunchArgs>::ptrs_init();
        b.iter(|| {
            let args: T = (
                black_box(42u64),
                black_box(7i32),
                black_box(1.5f32),
                black_box(3u32),
            );
            args.pack(&mut storage, &mut ptrs);
            black_box(&ptrs);
        });
    });
    g.bench_function("tuple8_ptrs", |b| {
        type T = (u64, u64, u64, u64, u64, u64, i32, i32);
        let mut storage = <T as LaunchArgs>::storage();
        let mut ptrs = <T as LaunchArgs>::ptrs_init();
        b.iter(|| {
            let args: T = (
                black_box(0x1_0000u64),
                black_box(0x2_0000u64),
                black_box(0x3_0000u64),
                black_box(0x4_0000u64),
                black_box(0x5_0000u64),
                black_box(0x6_0000u64),
                black_box(4096i32),
                black_box(4096i32),
            );
            args.pack(&mut storage, &mut ptrs);
            black_box(&ptrs);
        });
    });
    g.bench_function("tuple12_max", |b| {
        type T = (u64, u64, u64, u64, u64, u64, u64, u64, i32, i32, i32, i32);
        let mut storage = <T as LaunchArgs>::storage();
        let mut ptrs = <T as LaunchArgs>::ptrs_init();
        b.iter(|| {
            let args: T = (
                black_box(1u64),
                black_box(2u64),
                black_box(3u64),
                black_box(4u64),
                black_box(5u64),
                black_box(6u64),
                black_box(7u64),
                black_box(8u64),
                black_box(1i32),
                black_box(2i32),
                black_box(3i32),
                black_box(4i32),
            );
            args.pack(&mut storage, &mut ptrs);
            black_box(&ptrs);
        });
    });
    g.finish();
}

// ─── fnv1a source hashing (kernel cache key cost) ──────────────────────────

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn bench_fnv1a(c: &mut Criterion) {
    // A realistic kernel source size: a mid-sized fused elementwise+reduce
    // ~2 KiB of CUDA C++ after preprocessing.
    let short_src = include_str!("../src/lib.rs"); // ~2 KiB
    let long_src = include_str!("../src/drv.rs"); // ~25 KiB

    let mut g = c.benchmark_group("kernel_cache/fnv1a");
    g.throughput(Throughput::Bytes(short_src.len() as u64));
    g.bench_function(
        BenchmarkId::new("src", format!("{}B", short_src.len())),
        |b| {
            b.iter(|| black_box(fnv1a(black_box(short_src))));
        },
    );
    g.throughput(Throughput::Bytes(long_src.len() as u64));
    g.bench_function(
        BenchmarkId::new("src", format!("{}B", long_src.len())),
        |b| {
            b.iter(|| black_box(fnv1a(black_box(long_src))));
        },
    );
    g.finish();
}

criterion_group!(benches, bench_launch_args, bench_fnv1a);
criterion_main!(benches);
