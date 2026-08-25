//! End-to-end compute dispatch on a live Level Zero device.
//!
//! Level Zero consumes OpenCL/SYCL-flavored (`Kernel`-model) SPIR-V, which needs
//! Intel's `ocloc` or `clang -target spir64` to produce — not the Vulkan SDK's
//! `glslc`, whose output uses the `GLCompute` model and would be rejected. So
//! rather than compile a kernel inline, this test takes one via the
//! `IA_LEVELZERO_SHADER` environment variable (a path to a SPIR-V binary whose
//! `main` doubles a `__global float*`).
//!
//! It skips cleanly, with a printed reason, when there is no Level Zero device
//! (no `ze_loader`, no Intel GPU/NPU) or no shader was supplied — which is every
//! host without Intel compute, this CI included. When a device *and* a shader
//! are present it runs the full `ComputeDevice` round-trip.
//!
//! ```text
//! IA_LEVELZERO_SHADER=double.spv \
//!   cargo test -p ironaccelerator-levelzero --test dispatch -- --nocapture
//! ```

use ironaccelerator_core::ComputeDevice;
use ironaccelerator_levelzero::Context;

#[test]
fn dispatch_doubles_a_buffer() {
    let Some(ctx) = Context::new(0) else {
        eprintln!("skipped: no Level Zero device (no ze_loader / Intel compute)");
        return;
    };
    let Ok(path) = std::env::var("IA_LEVELZERO_SHADER") else {
        eprintln!(
            "skipped: device present but no IA_LEVELZERO_SHADER supplied \
             (Kernel-model SPIR-V needs ocloc / clang -target spir64)"
        );
        return;
    };
    let code = std::fs::read(&path).expect("IA_LEVELZERO_SHADER unreadable");

    const N: usize = 1024;
    let input: Vec<u8> = (0..N as u32)
        .flat_map(|i| (i as f32 * 0.5).to_le_bytes())
        .collect();

    let buf = ctx.upload(&input).expect("upload");
    assert_eq!(ctx.buffer_len(&buf), input.len() as u64);
    let pipe = ctx.pipeline(&code, 1).expect("build pipeline");
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
    eprintln!("level-zero — ComputeDevice roundtrip verified over {N} floats");
}
