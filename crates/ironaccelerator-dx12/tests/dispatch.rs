//! End-to-end compute dispatch on a live D3D12 adapter.
//!
//! Compiles a trivial HLSL kernel to DXIL with `dxc`, uploads a buffer, runs
//! the kernel, and reads the result back. This is the test that proves the
//! hand-written COM vtables in `compute.rs` are laid out correctly — a wrong
//! slot index shows up here as a failed `HRESULT` or a wrong answer, not as a
//! compile error.
//!
//! Skips cleanly when there is no D3D12 adapter or no `dxc` on the host, which
//! covers CI. It does not silently pass in that case: it prints why.

use ironaccelerator_dx12::compute::Context;
use ironaccelerator_dx12::drv;
use std::path::PathBuf;
use std::process::Command;

/// Doubles every element. `u0` is bound as a root UAV, so it must be a raw or
/// structured buffer — root descriptors do not support typed buffers.
const KERNEL: &str = r#"
#define RS "RootFlags(0), UAV(u0)"

RWStructuredBuffer<float> data : register(u0);

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    data[tid.x] = data[tid.x] * 2.0f;
}
"#;

/// Find a `dxc` that can **sign** its output.
///
/// D3D12 rejects unsigned DXIL with `E_INVALIDARG` at pipeline creation unless
/// the machine is in Developer Mode. `dxc` only signs when `dxil.dll` sits
/// beside it — the Windows SDK ships that pair, the Vulkan SDK's `dxc` does
/// not (it targets SPIR-V). Since the Vulkan SDK is what usually wins on PATH,
/// searching the Windows Kits first is what makes this test work rather than
/// mysteriously fail.
fn find_dxc() -> Option<PathBuf> {
    let roots = [
        r"C:\Program Files (x86)\Windows Kits\10\bin",
        r"C:\Program Files\Windows Kits\10\bin",
    ];
    let mut best: Option<PathBuf> = None;
    for root in roots {
        let Ok(versions) = std::fs::read_dir(root) else {
            continue;
        };
        for v in versions.flatten() {
            let dir = v.path().join("x64");
            let dxc = dir.join("dxc.exe");
            if dxc.is_file() && dir.join("dxil.dll").is_file() {
                // Later SDK directories sort higher; keep the newest.
                if best.as_ref().is_none_or(|b| b < &dxc) {
                    best = Some(dxc);
                }
            }
        }
    }
    if best.is_some() {
        return best;
    }
    // Last resort: PATH. May emit unsigned DXIL, which the test will report.
    Command::new("dxc")
        .arg("--version")
        .output()
        .ok()
        .map(|_| PathBuf::from("dxc"))
}

fn compile_dxil(dxc: &PathBuf) -> Option<Vec<u8>> {
    let dir = std::env::temp_dir().join("ia_dx12_dispatch_test");
    std::fs::create_dir_all(&dir).ok()?;
    let hlsl = dir.join("double.hlsl");
    let dxil = dir.join("double.dxil");
    std::fs::write(&hlsl, KERNEL).ok()?;

    let out = Command::new(dxc)
        .args(["-T", "cs_6_0", "-E", "main"])
        .arg("-Fo")
        .arg(&dxil)
        .arg(&hlsl)
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "dxc failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
        return None;
    }
    std::fs::read(&dxil).ok()
}

#[test]
fn dispatch_doubles_a_buffer_on_every_adapter() {
    let adapters = drv::enumerate();
    if adapters.is_empty() {
        eprintln!("skipped: no D3D12 adapter on this host");
        return;
    }
    // Escape hatch for bringing your own precompiled blob (DXIL or DXBC),
    // which is also how this test gets bisected when a driver rejects one.
    if let Ok(p) = std::env::var("IA_DX12_SHADER") {
        let blob = std::fs::read(&p).expect("IA_DX12_SHADER unreadable");
        eprintln!("using shader blob from {p} ({} bytes)", blob.len());
        run_dispatch(&adapters, &blob);
        return;
    }
    let Some(dxc) = find_dxc() else {
        eprintln!("skipped: dxc not found (PATH, Vulkan SDK, or Windows Kits)");
        return;
    };
    let Some(dxil) = compile_dxil(&dxc) else {
        eprintln!("skipped: dxc could not produce DXIL");
        return;
    };
    assert!(
        dxil.starts_with(b"DXBC"),
        "dxc output is not a DXIL container"
    );
    run_dispatch(&adapters, &dxil);
}

fn run_dispatch(adapters: &[drv::EnumeratedAdapter], shader: &[u8]) {
    const N: usize = 1024;
    let input: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();
    let bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();

    for a in adapters {
        let ctx = Context::new(a.ordinal).expect("context");
        // Use the root signature we serialise ourselves, not the one embedded
        // in the shader — that is the path under test.
        let root = ctx.root_signature_with_uavs(1).expect("root signature");
        let pso = ctx
            .compute_pipeline(Some(&root), shader)
            .unwrap_or_else(|e| panic!("[{}] {}: {e}", a.ordinal, a.name));

        let buf = ctx.upload(&bytes).expect("upload");
        ctx.dispatch(&root, &pso, &[&buf], [(N / 64) as u32, 1, 1])
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
                "[{}] adapter {} element {i}: got {g}, want {e}",
                a.ordinal,
                a.name
            );
        }
        eprintln!(
            "[{}] {} — dispatch verified over {N} floats",
            a.ordinal, a.name
        );
    }
}
