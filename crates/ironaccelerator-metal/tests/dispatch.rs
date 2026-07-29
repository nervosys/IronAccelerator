//! End-to-end compute dispatch on a live Metal device. macOS-only.
//!
//! Compiles a trivial MSL kernel to a `.metallib` with `xcrun`, uploads a
//! buffer, runs the kernel through the unified `ComputeDevice` trait, and reads
//! the result back. The whole file is gated to `target_vendor = "apple"`, so on
//! any other host it compiles to nothing and reports zero tests — this
//! workspace's CI is Windows, so the run happens only on an actual Mac.
//!
//! Skips cleanly, with a printed reason, when there is no Metal device or no
//! `xcrun` toolchain.
//!
//! Note the kernel is *not* named `main`: MSL reserves it, which is exactly why
//! the trait's `pipeline` takes the metallib's first kernel rather than a fixed
//! entry name.

#![cfg(target_vendor = "apple")]

use ironaccelerator_core::ComputeDevice;
use ironaccelerator_metal::Context;
use std::process::Command;

const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void double_kernel(device float* data [[buffer(0)]],
                          uint tid [[thread_position_in_grid]]) {
    data[tid] = data[tid] * 2.0;
}
"#;

fn build_metallib() -> Option<Vec<u8>> {
    let dir = std::env::temp_dir().join("ia_metal_dispatch_test");
    std::fs::create_dir_all(&dir).ok()?;
    let src = dir.join("k.metal");
    let air = dir.join("k.air");
    let lib = dir.join("k.metallib");
    std::fs::write(&src, MSL).ok()?;

    let air_ok = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-c"])
        .arg(&src)
        .arg("-o")
        .arg(&air)
        .status()
        .ok()?
        .success();
    if !air_ok {
        return None;
    }
    let lib_ok = Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .arg(&air)
        .arg("-o")
        .arg(&lib)
        .status()
        .ok()?
        .success();
    if !lib_ok {
        return None;
    }
    std::fs::read(&lib).ok()
}

#[test]
fn dispatch_doubles_a_buffer() {
    let Some(ctx) = Context::system_default() else {
        eprintln!("skipped: no Metal device");
        return;
    };
    let Some(metallib) = build_metallib() else {
        eprintln!("skipped: xcrun could not build a metallib");
        return;
    };

    const N: usize = 1024;
    let input: Vec<u8> = (0..N as u32).flat_map(|i| (i as f32 * 0.5).to_le_bytes()).collect();

    let buf = ctx.upload(&input).expect("upload");
    assert_eq!(ctx.buffer_len(&buf), input.len() as u64);
    let pipe = ctx.pipeline(&metallib, 1).expect("build pipeline");
    ctx.dispatch(&pipe, &[&buf], [(N / 64) as u32, 1, 1])
        .expect("dispatch");

    let mut out = vec![0u8; input.len()];
    ctx.download(&buf, &mut out).expect("download");
    let got: Vec<f32> = out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for (i, g) in got.iter().enumerate() {
        let want = i as f32 * 0.5 * 2.0;
        assert!((g - want).abs() < 1e-6, "element {i}: got {g}, want {want}");
    }
    eprintln!("metal — ComputeDevice roundtrip verified over {N} floats");
}
