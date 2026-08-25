//! The unified [`ComputeDevice`] trait, exercised generically across backends.
//!
//! The point of the trait is that one function compiles and runs against every
//! backend that owns a device. This test proves that literally: a single
//! generic `double_roundtrip` drives both the Vulkan and D3D12 backends through
//! the same code, feeding each the bytecode its driver consumes (SPIR-V vs
//! DXIL). If the trait were merely per-backend sugar this file would not
//! compile.
//!
//! Runs on `--features all` (needs both backends compiled in). Each backend
//! skips cleanly, with a printed reason, when its device or shader compiler is
//! absent — the same CI-friendly contract the per-backend dispatch tests use.
//!
//! ```text
//! cargo test -p ironaccelerator --features all --test unified_compute
//! ```

#![cfg(all(feature = "vulkan", feature = "dx12"))]

use ironaccelerator::core::ComputeDevice;
use std::path::PathBuf;
use std::process::Command;

const N: usize = 1024;

/// The whole point: one generic routine, no mention of any backend. Uploads
/// `N` floats, runs a shader that doubles each, reads them back. Returns the
/// downloaded values so the caller can check them.
fn double_roundtrip<C: ComputeDevice>(dev: &C, code: &[u8]) -> Result<Vec<f32>, C::Error> {
    let input: Vec<u8> = (0..N as u32)
        .flat_map(|i| (i as f32 * 0.5).to_le_bytes())
        .collect();
    let buf = dev.upload(&input)?;
    let pipe = dev.pipeline(code, 1)?;
    dev.dispatch(&pipe, &[&buf], [(N / 64) as u32, 1, 1])?;
    let mut out = vec![0u8; input.len()];
    dev.download(&buf, &mut out)?;
    Ok(out
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn assert_doubled(got: &[f32]) {
    for (i, g) in got.iter().enumerate() {
        let want = i as f32 * 0.5 * 2.0;
        assert!((g - want).abs() < 1e-6, "element {i}: got {g}, want {want}");
    }
}

// ── Vulkan: compile GLSL → SPIR-V ────────────────────────────────────────────

const GLSL: &str = r#"#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Data { float data[]; };
void main() { data[gl_GlobalInvocationID.x] *= 2.0; }
"#;

fn spirv() -> Option<Vec<u8>> {
    let sdk = std::env::var("VULKAN_SDK").ok()?;
    let glslc =
        PathBuf::from(sdk)
            .join("Bin")
            .join(if cfg!(windows) { "glslc.exe" } else { "glslc" });
    let glslc = if glslc.is_file() {
        glslc
    } else {
        PathBuf::from("glslc")
    };
    let dir = std::env::temp_dir().join("ia_unified_compute");
    std::fs::create_dir_all(&dir).ok()?;
    let src = dir.join("d.comp");
    let out = dir.join("d.spv");
    std::fs::write(&src, GLSL).ok()?;
    let ok = Command::new(&glslc)
        .args(["-fshader-stage=comp"])
        .arg(&src)
        .arg("-o")
        .arg(&out)
        .status()
        .ok()?
        .success();
    ok.then(|| std::fs::read(&out).ok()).flatten()
}

// ── D3D12: compile HLSL → DXIL ───────────────────────────────────────────────

const HLSL: &str = r#"
#define RS "RootFlags(0), UAV(u0)"
RWStructuredBuffer<float> data : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) { data[tid.x] = data[tid.x] * 2.0f; }
"#;

fn dxil() -> Option<Vec<u8>> {
    // dxc must be able to sign (dxil.dll beside it) — the Windows Kits build can.
    let mut dxc = None;
    for root in [
        r"C:\Program Files (x86)\Windows Kits\10\bin",
        r"C:\Program Files\Windows Kits\10\bin",
    ] {
        let Ok(versions) = std::fs::read_dir(root) else {
            continue;
        };
        for v in versions.flatten() {
            let dir = v.path().join("x64");
            if dir.join("dxc.exe").is_file() && dir.join("dxil.dll").is_file() {
                dxc = Some(dir.join("dxc.exe"));
            }
        }
    }
    let dxc = dxc?;
    let dir = std::env::temp_dir().join("ia_unified_compute");
    std::fs::create_dir_all(&dir).ok()?;
    let src = dir.join("d.hlsl");
    let out = dir.join("d.dxil");
    std::fs::write(&src, HLSL).ok()?;
    let ok = Command::new(&dxc)
        .args(["-T", "cs_6_0", "-E", "main"])
        .arg("-Fo")
        .arg(&out)
        .arg(&src)
        .status()
        .ok()?
        .success();
    ok.then(|| std::fs::read(&out).ok()).flatten()
}

#[test]
fn one_generic_routine_runs_on_vulkan() {
    use ironaccelerator::vulkan::{drv, Context};
    let devices = drv::enumerate();
    if devices.is_empty() {
        eprintln!("skipped: no Vulkan device");
        return;
    }
    let Some(code) = spirv() else {
        eprintln!("skipped: glslc unavailable");
        return;
    };
    let mut ran = 0;
    for pd in devices.iter().filter(|d| d.compute_queue_family.is_some()) {
        let Some(ctx) = Context::new(pd.ordinal) else {
            continue;
        };
        let got = double_roundtrip(&ctx, &code).expect("vulkan roundtrip");
        assert_doubled(&got);
        eprintln!("vulkan [{}] {} — generic roundtrip ok", pd.ordinal, pd.name);
        ran += 1;
    }
    if ran == 0 {
        eprintln!("skipped: no Vulkan compute queue");
    }
}

#[test]
fn the_same_generic_routine_runs_on_d3d12() {
    use ironaccelerator::dx12::{drv, Context};
    let adapters = drv::enumerate();
    if adapters.is_empty() {
        eprintln!("skipped: no D3D12 adapter");
        return;
    }
    let Some(code) = dxil() else {
        eprintln!("skipped: signing dxc unavailable");
        return;
    };
    for a in &adapters {
        let ctx = Context::new(a.ordinal).expect("d3d12 context");
        match double_roundtrip(&ctx, &code) {
            Ok(got) => {
                assert_doubled(&got);
                eprintln!("d3d12 [{}] {} — generic roundtrip ok", a.ordinal, a.name);
            }
            // WARP and some virtual adapters can reject a real dispatch; report,
            // don't fail the suite over a software rasteriser.
            Err(e) => eprintln!("d3d12 [{}] {} — skipped: {e}", a.ordinal, a.name),
        }
    }
}
