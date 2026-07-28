//! Interleaved A/B comparison against cudarc.
//!
//! Criterion measures each implementation in its own contiguous block, so any
//! drift in machine state during a run — a background workload starting, the
//! GPU changing power state, the PCIe link retraining — lands entirely on one
//! side and shows up as a difference that is not in the code. On a shared
//! desktop that effect is larger than the difference being measured.
//!
//! This alternates the two implementations round by round and reports the
//! median of each, so drift affects both equally and cancels in the ratio.
//!
//! ```text
//! cargo run --release -p ironaccelerator-cuda --example ab_vs_cudarc
//! ```

use ironaccelerator_cuda::cudarc_compat as iron;
use ironaccelerator_cuda::cudarc_compat::CudaStreamExt;
use std::time::{Duration, Instant};

const ROUNDS: usize = 15;

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}

/// Time `f` over `inner` repetitions, returning per-repetition duration.
fn timed(inner: usize, mut f: impl FnMut()) -> Duration {
    let t = Instant::now();
    for _ in 0..inner {
        f();
    }
    t.elapsed() / inner as u32
}

fn main() {
    let Ok(iron_dev) = iron::CudaDevice::new(0) else {
        eprintln!("no CUDA device");
        return;
    };
    let iron_stream = iron_dev.default_stream();
    let Ok(cudarc_ctx) = cudarc::driver::CudaContext::new(0) else {
        eprintln!("cudarc init failed");
        return;
    };
    let cudarc_stream = cudarc_ctx.default_stream();

    println!(
        "device: {}\nrounds: {ROUNDS} (interleaved, median reported)\n",
        iron_dev.name().unwrap_or_else(|_| "?".into())
    );
    println!(
        "{:<10} {:>8} {:>14} {:>14} {:>9}  verdict",
        "op", "size", "ironaccel", "cudarc", "ratio"
    );
    println!("{}", "-".repeat(72));

    let sizes: &[(usize, &str, usize)] = &[
        (1 << 10, "1KB", 200),
        (64 << 10, "64KB", 100),
        (1 << 20, "1MB", 30),
        (16 << 20, "16MB", 5),
    ];

    for &(n, label, inner) in sizes {
        let host = vec![0u8; n];

        // ── H2D ──
        let mut ia = Vec::new();
        let mut cu = Vec::new();
        for _ in 0..ROUNDS {
            ia.push(timed(inner, || {
                let b = iron_stream.htod_sync_copy(&host).unwrap();
                std::hint::black_box(&b);
            }));
            cu.push(timed(inner, || {
                let b = cudarc_stream.clone_htod(&host).unwrap();
                cudarc_stream.synchronize().unwrap();
                std::hint::black_box(&b);
            }));
        }
        report("h2d", label, median(ia), median(cu));

        // ── D2H ──
        let ia_dev = iron_stream.htod_sync_copy(&host).unwrap();
        let cu_dev = cudarc_stream.clone_htod(&host).unwrap();
        cudarc_stream.synchronize().unwrap();
        let mut ia = Vec::new();
        let mut cu = Vec::new();
        for _ in 0..ROUNDS {
            ia.push(timed(inner, || {
                let v: Vec<u8> = iron_stream.dtoh_sync_copy(&ia_dev).unwrap();
                std::hint::black_box(v);
            }));
            cu.push(timed(inner, || {
                let v = cudarc_stream.clone_dtoh(&cu_dev).unwrap();
                std::hint::black_box(v);
            }));
        }
        report("d2h", label, median(ia), median(cu));
    }
}

fn report(op: &str, size: &str, ia: Duration, cu: Duration) {
    let ratio = cu.as_secs_f64() / ia.as_secs_f64();
    let verdict = if ratio >= 1.05 {
        "IA faster"
    } else if ratio <= 0.95 {
        "IA SLOWER"
    } else {
        "parity"
    };
    println!(
        "{:<10} {:>8} {:>14?} {:>14?} {:>8.2}x  {}",
        op, size, ia, cu, ratio, verdict
    );
}
