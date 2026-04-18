//! Host-side hot-path benchmarks.
//!
//! These bench what IronAccelerator's wrappers do *around* every GPU call:
//! capability detection, planner dispatch, recipe validation, kernel-arg
//! packing, gemm-key hashing, and fnv-style source hashing. GPU-bound work
//! (actual memcpy/matmul/FFT throughput) is out of scope — those need
//! hardware and belong in a `--ignored` runner.
//!
//! Overhead budget target: every path here should finish in **well under a
//! microsecond** so the planner and launch helpers never show up in a
//! kernel-launch profile.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ironaccelerator_core::{DType, Workload};
use ironaccelerator_cuda::backend::{capability_from_arch, heuristic_score, plan_strategy};
use ironaccelerator_cuda::drv::LaunchArgs;
use ironaccelerator_cuda::fp8::{AmaxHistoryLen, Fp8Recipe};
use ironaccelerator_cuda::tune::GemmKey;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ─── Capability ────────────────────────────────────────────────────────────

fn bench_capability(c: &mut Criterion) {
    let mut g = c.benchmark_group("capability");
    for (maj, min, label) in [
        (8, 0, "sm80_a100"),
        (8, 9, "sm89_4090"),
        (9, 0, "sm90_h100"),
        (10, 0, "sm100_b200"),
    ] {
        g.bench_function(label, |b| {
            b.iter(|| {
                let cap = capability_from_arch(
                    black_box(maj), black_box(min),
                    black_box(80 * 1024 * 1024 * 1024),
                    black_box(6_500_000),
                    black_box(5120),
                );
                black_box(cap);
            });
        });
    }
    g.finish();
}

// ─── Planner ───────────────────────────────────────────────────────────────

fn bench_planner(c: &mut Criterion) {
    let cap_h100 = capability_from_arch(9, 0, 80 * 1024 * 1024 * 1024, 6_500_000, 5120);
    let cap_4090 = capability_from_arch(8, 9, 24 * 1024 * 1024 * 1024, 10_500_000, 384);

    let shapes: &[(u32, u32, u32, DType, &str)] = &[
        (128, 128, 128, DType::Bf16,     "tiny_bf16"),
        (4096, 4096, 4096, DType::Bf16,  "llm_mlp_bf16"),
        (8192, 8192, 8192, DType::F8E4M3,"llm_mlp_fp8"),
        (16, 4096, 4096, DType::Bf16,    "decode_bf16"),
        (32768, 32768, 128, DType::F16,  "embedding_fp16"),
    ];

    let mut g = c.benchmark_group("planner/plan_strategy");
    for (m, n, k, dt, label) in shapes {
        let w = Workload::gemm(*m, *n, *k, *dt);
        g.bench_with_input(BenchmarkId::new("h100", label), &w, |b, w| {
            b.iter(|| black_box(plan_strategy(black_box(&cap_h100), black_box(w))));
        });
        g.bench_with_input(BenchmarkId::new("rtx4090", label), &w, |b, w| {
            b.iter(|| black_box(plan_strategy(black_box(&cap_4090), black_box(w))));
        });
    }
    g.finish();

    let mut g = c.benchmark_group("planner/heuristic_score");
    for (m, n, k, dt, label) in shapes {
        let w = Workload::gemm(*m, *n, *k, *dt);
        g.bench_with_input(BenchmarkId::new("h100", label), &w, |b, w| {
            b.iter(|| black_box(heuristic_score(black_box(&cap_h100), black_box(w))));
        });
    }
    g.finish();
}

// ─── FP8 recipe ────────────────────────────────────────────────────────────

fn bench_fp8_recipe(c: &mut Criterion) {
    let mut g = c.benchmark_group("fp8");
    g.bench_function("validate/hopper_default", |b| {
        let r = Fp8Recipe::hopper_default();
        b.iter(|| black_box(r.validate().ok()));
    });
    g.bench_function("validate/blackwell_mx", |b| {
        let r = Fp8Recipe::blackwell_mx();
        b.iter(|| black_box(r.validate().ok()));
    });
    g.bench_function("validate/custom_hist", |b| {
        let mut r = Fp8Recipe::hopper_default();
        r.amax_history = AmaxHistoryLen::Custom(256);
        b.iter(|| black_box(r.validate().ok()));
    });
    g.finish();
}

// ─── Kernel-arg packing ────────────────────────────────────────────────────

fn bench_launch_args(c: &mut Criterion) {
    let mut g = c.benchmark_group("launch_args/pack");
    g.bench_function("tuple1_u64", |b| {
        let mut slots = <(u64,) as LaunchArgs>::slots();
        b.iter(|| {
            let args = (black_box(42u64),);
            let packed = args.pack(&mut slots);
            black_box(packed.len())
        });
    });
    g.bench_function("tuple4_mixed", |b| {
        let mut slots = <(u64, i32, f32, u32) as LaunchArgs>::slots();
        b.iter(|| {
            let args = (black_box(42u64), black_box(7i32), black_box(1.5f32), black_box(3u32));
            let packed = args.pack(&mut slots);
            black_box(packed.len())
        });
    });
    g.bench_function("tuple8_ptrs", |b| {
        let mut slots = <(u64, u64, u64, u64, u64, u64, i32, i32) as LaunchArgs>::slots();
        b.iter(|| {
            let args = (
                black_box(0x1_0000u64), black_box(0x2_0000u64),
                black_box(0x3_0000u64), black_box(0x4_0000u64),
                black_box(0x5_0000u64), black_box(0x6_0000u64),
                black_box(4096i32),     black_box(4096i32),
            );
            let packed = args.pack(&mut slots);
            black_box(packed.len())
        });
    });
    g.bench_function("tuple12_max", |b| {
        type T = (u64, u64, u64, u64, u64, u64, u64, u64, i32, i32, i32, i32);
        let mut slots = <T as LaunchArgs>::slots();
        b.iter(|| {
            let args: T = (
                black_box(1u64), black_box(2u64), black_box(3u64), black_box(4u64),
                black_box(5u64), black_box(6u64), black_box(7u64), black_box(8u64),
                black_box(1i32), black_box(2i32), black_box(3i32), black_box(4i32),
            );
            let packed = args.pack(&mut slots);
            black_box(packed.len())
        });
    });
    g.finish();
}

// ─── GemmKey hashing (autotuner cache lookup cost) ─────────────────────────

fn bench_gemm_key(c: &mut Criterion) {
    let mut g = c.benchmark_group("tune/gemm_key");
    let keys: Vec<GemmKey> = (0..1024)
        .map(|i| GemmKey {
            m: 1024 + (i % 32) * 128, n: 1024 + (i % 16) * 128,
            k: 1024 + (i % 8) * 128, dtype: i as u8 % 16,
        })
        .collect();
    g.throughput(Throughput::Elements(keys.len() as u64));
    g.bench_function("hash_1k_keys", |b| {
        b.iter(|| {
            let mut h = DefaultHasher::new();
            for k in &keys {
                k.hash(&mut h);
            }
            black_box(h.finish())
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
    let short_src = include_str!("../src/lib.rs");       // ~2 KiB
    let long_src  = include_str!("../src/drv.rs");       // ~25 KiB

    let mut g = c.benchmark_group("kernel_cache/fnv1a");
    g.throughput(Throughput::Bytes(short_src.len() as u64));
    g.bench_function(BenchmarkId::new("src", format!("{}B", short_src.len())), |b| {
        b.iter(|| black_box(fnv1a(black_box(short_src))));
    });
    g.throughput(Throughput::Bytes(long_src.len() as u64));
    g.bench_function(BenchmarkId::new("src", format!("{}B", long_src.len())), |b| {
        b.iter(|| black_box(fnv1a(black_box(long_src))));
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_capability,
    bench_planner,
    bench_fp8_recipe,
    bench_launch_args,
    bench_gemm_key,
    bench_fnv1a,
);
criterion_main!(benches);
