//! End-to-end compute dispatch on a live Vulkan device.
//!
//! Compiles a trivial GLSL compute shader to SPIR-V with the Vulkan SDK's
//! `glslc` (falling back to `glslangValidator`), uploads a buffer, runs the
//! shader, and reads the result back. This is what proves the whole
//! `compute.rs` path — descriptor-set layout, pipeline layout, the transfer
//! barriers around `dispatch`, and the staging round-trip — actually hangs
//! together on real silicon rather than merely compiling.
//!
//! Skips cleanly, with a printed reason, when there is no Vulkan device or no
//! SPIR-V compiler on the host — which is exactly the CI-without-a-GPU case.

#![cfg(not(target_arch = "wasm32"))]

use ironaccelerator_vulkan::{drv, Buffer, ComputePipeline, Context};
use std::path::PathBuf;
use std::process::Command;

/// Doubles every element. `binding = 0` is the single storage buffer the
/// pipeline binds at descriptor slot 0.
const KERNEL: &str = r#"#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Data { float data[]; };
void main() {
    data[gl_GlobalInvocationID.x] *= 2.0;
}
"#;

/// Locate a SPIR-V compiler. Prefer `glslc` (Shaderc, the standard in the
/// Vulkan SDK); fall back to `glslangValidator`. `$VULKAN_SDK/Bin` is searched
/// before PATH so the SDK build wins over any stray copy.
fn find_compiler() -> Option<(PathBuf, bool)> {
    let exe = |base: &str| {
        if cfg!(windows) {
            format!("{base}.exe")
        } else {
            base.to_string()
        }
    };
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        let bin = PathBuf::from(&sdk).join("Bin");
        let glslc = bin.join(exe("glslc"));
        if glslc.is_file() {
            return Some((glslc, true));
        }
        let glslang = bin.join(exe("glslangValidator"));
        if glslang.is_file() {
            return Some((glslang, false));
        }
    }
    for (name, is_glslc) in [("glslc", true), ("glslangValidator", false)] {
        if Command::new(name).arg("--version").output().is_ok() {
            return Some((PathBuf::from(name), is_glslc));
        }
    }
    None
}

fn compile_spirv(tool: &PathBuf, is_glslc: bool) -> Option<Vec<u32>> {
    let dir = std::env::temp_dir().join("ia_vulkan_dispatch_test");
    std::fs::create_dir_all(&dir).ok()?;
    let src = dir.join("double.comp");
    let spv = dir.join("double.spv");
    std::fs::write(&src, KERNEL).ok()?;

    let out = if is_glslc {
        Command::new(tool)
            .args(["-fshader-stage=comp"])
            .arg(&src)
            .arg("-o")
            .arg(&spv)
            .output()
    } else {
        // glslangValidator infers the stage from the `.comp` extension; `-V`
        // targets Vulkan SPIR-V.
        Command::new(tool)
            .arg("-V")
            .arg(&src)
            .arg("-o")
            .arg(&spv)
            .output()
    }
    .ok()?;
    if !out.status.success() {
        eprintln!(
            "shader compile failed: {}{}",
            String::from_utf8_lossy(&out.stdout).trim(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
        return None;
    }
    let bytes = std::fs::read(&spv).ok()?;
    if bytes.len() % 4 != 0 {
        eprintln!("SPIR-V length {} is not a multiple of 4", bytes.len());
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

#[test]
fn dispatch_doubles_a_buffer_on_every_device() {
    let devices = drv::enumerate();
    if devices.is_empty() {
        eprintln!("skipped: no Vulkan device on this host");
        return;
    }
    let Some((tool, is_glslc)) = find_compiler() else {
        eprintln!("skipped: no glslc / glslangValidator (install the Vulkan SDK)");
        return;
    };
    let Some(spirv) = compile_spirv(&tool, is_glslc) else {
        eprintln!("skipped: SPIR-V compiler could not produce output");
        return;
    };
    assert_eq!(spirv.first(), Some(&0x0723_0203), "not a SPIR-V module");

    const N: usize = 1024;
    let input: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();
    let bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut ran = 0usize;
    for pd in &devices {
        if pd.compute_queue_family.is_none() {
            continue;
        }
        let Some(ctx) = Context::new(pd.ordinal) else {
            eprintln!("[{}] {}: no compute context, skipping", pd.ordinal, pd.name);
            continue;
        };

        // Device-local storage buffer, filled from the host through staging.
        let buf = Buffer::device_local(&ctx, bytes.len() as u64).expect("device buffer");
        let staging = Buffer::host_visible(&ctx, bytes.len() as u64).expect("staging");
        staging.write_bytes(&bytes).expect("write staging");
        ctx.copy_buffer(&staging, &buf, bytes.len() as u64)
            .expect("upload copy");

        let pipeline = ComputePipeline::new(&ctx, &spirv, c"main", &[&buf])
            .unwrap_or_else(|e| panic!("[{}] {}: pipeline {e:?}", pd.ordinal, pd.name));
        ctx.dispatch(&pipeline, [(N / 64) as u32, 1, 1])
            .expect("dispatch");

        let mut out = vec![0u8; bytes.len()];
        ctx.download(&buf, &mut out).expect("download");
        let got: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        for (i, (g, e)) in got.iter().zip(input.iter().map(|v| v * 2.0)).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "[{}] {} element {i}: got {g}, want {e}",
                pd.ordinal,
                pd.name
            );
        }
        eprintln!("[{}] {} — dispatch verified over {N} floats", pd.ordinal, pd.name);
        ran += 1;
    }
    if ran == 0 {
        eprintln!("skipped: devices enumerated but none exposed a compute queue");
    }
}
