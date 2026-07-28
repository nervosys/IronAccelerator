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

/// Relative spread above which a row's ratio is not worth believing.
const UNSTABLE_SPREAD: f64 = 0.25;

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}

/// (max - min) / median, as a unitless measure of how much the machine moved
/// underneath the measurement.
fn spread(v: &[Duration]) -> f64 {
    let mut s = v.to_vec();
    s.sort_unstable();
    let (lo, hi, mid) = (s[0], s[s.len() - 1], s[s.len() / 2]);
    (hi.as_secs_f64() - lo.as_secs_f64()) / mid.as_secs_f64().max(f64::MIN_POSITIVE)
}

/// Physical index of the CUDA device this process will use.
fn physical_index() -> String {
    std::env::var("CUDA_VISIBLE_DEVICES")
        .ok()
        .and_then(|v| v.split(',').next().map(str::to_owned))
        .unwrap_or_else(|| "0".into())
}

/// Report anything about the device that would make timings meaningless.
///
/// A benchmark that cannot tell you when to distrust it is worse than none:
/// on a shared desktop, contention and power state move results by more than
/// the difference being measured.
fn preflight(idx: &str) -> Vec<String> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "-i",
            idx,
            "--query-gpu=utilization.gpu,pstate,clocks.sm,clocks.max.sm,\
             pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,\
             pcie.link.width.max,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(out) = out else {
        return vec!["nvidia-smi unavailable; cannot validate device state".into()];
    };
    let line = String::from_utf8_lossy(&out.stdout);
    let f: Vec<String> = line
        .trim()
        .split(',')
        .map(|s| s.trim().to_owned())
        .collect();
    if f.len() < 9 {
        return vec!["could not parse nvidia-smi output".into()];
    }
    let num = |s: &str| s.parse::<f64>().unwrap_or(0.0);
    let mut warnings = Vec::new();

    if num(&f[0]) > 10.0 {
        warnings.push(format!(
            "device is {}% busy with other work — timings will be contended",
            f[0]
        ));
    }
    if num(&f[2]) < 0.75 * num(&f[3]) {
        warnings.push(format!(
            "clocks throttled to {} MHz of {} MHz ({})",
            f[2], f[3], f[1]
        ));
    }
    if num(&f[4]) < num(&f[5]) || num(&f[6]) < num(&f[7]) {
        warnings.push(format!(
            "PCIe link downtrained to gen{} x{} of gen{} x{} — memcpy numbers \
             will not reflect the hardware",
            f[4], f[6], f[5], f[7]
        ));
    }
    if num(&f[8]) > 1024.0 {
        warnings.push(format!(
            "{} MiB already resident from other processes",
            f[8]
        ));
    }
    warnings
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

    let idx = physical_index();
    println!(
        "device: {} (physical index {idx})\nrounds: {ROUNDS} (interleaved, median reported)",
        iron_dev.name().unwrap_or_else(|_| "?".into())
    );

    let warnings = preflight(&idx);
    if warnings.is_empty() {
        println!("preflight: device looks quiet\n");
    } else {
        println!("\n!! PREFLIGHT WARNINGS — results below are NOT trustworthy:");
        for w in &warnings {
            println!("   - {w}");
        }
        println!("   Free the device (or pick another with CUDA_VISIBLE_DEVICES)\n   and lock clocks with `nvidia-smi -i {idx} -lgc <mhz>` before believing a ratio.\n");
    }

    println!(
        "{:<10} {:>8} {:>13} {:>13} {:>8} {:>8}  verdict",
        "op", "size", "ironaccel", "cudarc", "ratio", "spread"
    );
    println!("{}", "-".repeat(80));

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
        report("h2d", label, &ia, &cu);

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
        report("d2h", label, &ia, &cu);
    }

    if !warnings.is_empty() {
        println!(
            "\nReminder: the preflight warnings above apply to every row. Do not \
             quote these numbers."
        );
    }
}

fn report(op: &str, size: &str, ia: &[Duration], cu: &[Duration]) {
    let (m_ia, m_cu) = (median(ia.to_vec()), median(cu.to_vec()));
    let ratio = m_cu.as_secs_f64() / m_ia.as_secs_f64();
    let sp = spread(ia).max(spread(cu));
    // A ratio is only meaningful if it is larger than the noise that produced
    // it, so an unstable row reports no verdict rather than a wrong one.
    let verdict = if sp > UNSTABLE_SPREAD {
        "UNSTABLE — no verdict"
    } else if ratio >= 1.05 {
        "IA faster"
    } else if ratio <= 0.95 {
        "IA SLOWER"
    } else {
        "parity"
    };
    println!(
        "{:<10} {:>8} {:>13?} {:>13?} {:>7.2}x {:>7.0}%  {}",
        op,
        size,
        m_ia,
        m_cu,
        ratio,
        sp * 100.0,
        verdict
    );
}
