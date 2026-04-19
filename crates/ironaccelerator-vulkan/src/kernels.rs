//! Reference Vulkan compute kernels. Source is WGSL — compiled to
//! SPIR-V at runtime by [`crate::shader::wgsl_to_spirv`] — so the same
//! text also drives the WebGPU backend.

use crate::compute::{Buffer, ComputePipeline, Context};
use crate::shader::{wgsl_to_spirv, ShaderError};

/// Naive tiled GEMM (`C = A * B`, row-major f32). Matches the WGSL
/// exposed by `ironaccelerator-webgpu::kernels::GEMM_F32_WGSL` so the
/// two backends share one source.
pub const GEMM_F32_WGSL: &str = r#"
struct Dims { m: u32, n: u32, k: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       a:    array<f32>;
@group(0) @binding(1) var<storage, read>       b:    array<f32>;
@group(0) @binding(2) var<storage, read_write> c:    array<f32>;
@group(0) @binding(3) var<storage, read>       dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= dims.m || col >= dims.n) { return; }
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < dims.k; k = k + 1u) {
        acc = acc + a[row * dims.k + k] * b[k * dims.n + col];
    }
    c[row * dims.n + col] = acc;
}
"#;

/// SAXPY: `y[i] = alpha * x[i] + y[i]`. Binding 0 = `x`, 1 = `y`,
/// 2 = `params { alpha: f32, n: u32 }` (uniform-ish storage buffer).
pub const SAXPY_WGSL: &str = r#"
struct Params { alpha: f32, n: u32 };
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read>       params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    y[i] = params.alpha * x[i] + y[i];
}
"#;

#[derive(Debug)]
pub enum KernelError {
    Shader(ShaderError),
    Vulkan(ash::vk::Result),
}

impl From<ShaderError> for KernelError {
    fn from(e: ShaderError) -> Self { KernelError::Shader(e) }
}

impl From<ash::vk::Result> for KernelError {
    fn from(e: ash::vk::Result) -> Self { KernelError::Vulkan(e) }
}

/// Compile the SAXPY kernel and dispatch across `n` elements. Caller
/// owns `x`, `y`, `params` — all three must be STORAGE_BUFFER-usable
/// and bound in that order.
pub fn axpy_f32(
    ctx: &Context,
    x: &Buffer,
    y: &Buffer,
    params: &Buffer,
    n: u32,
) -> Result<(), KernelError> {
    let spirv = wgsl_to_spirv(SAXPY_WGSL)?;
    let entry = std::ffi::CString::new("main").unwrap();
    let pipeline = ComputePipeline::new(ctx, &spirv, &entry, &[x, y, params])?;
    let groups = [(n + 63) / 64, 1, 1];
    ctx.dispatch(&pipeline, groups)?;
    Ok(())
}

/// Compile the tiled GEMM WGSL to SPIR-V and dispatch across `[M, N]`.
pub fn gemm_f32(
    ctx: &Context,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    dims: &Buffer,
    m: u32,
    n: u32,
) -> Result<(), KernelError> {
    let spirv = wgsl_to_spirv(GEMM_F32_WGSL)?;
    let entry = std::ffi::CString::new("main").unwrap();
    let pipeline = ComputePipeline::new(ctx, &spirv, &entry, &[a, b, c, dims])?;
    let groups = [(n + 15) / 16, (m + 15) / 16, 1];
    ctx.dispatch(&pipeline, groups)?;
    Ok(())
}
