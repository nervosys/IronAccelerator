//! Reference WGSL kernels. First real payload for the WebGPU backend —
//! SAXPY (`y[i] = alpha * x[i] + y[i]`) on `f32`.

use crate::compute::{dispatch, ComputePipeline, Context};

/// SAXPY source. Binding 0 = `x` (read), binding 1 = `y` (read-write),
/// binding 2 = `params` (alpha + n, uniform-ish through a storage
/// buffer to stay on the basic limits set).
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

/// Naive tiled GEMM: `C = A * B`, row-major, f32. `A` is `[M, K]`,
/// `B` is `[K, N]`, `C` is `[M, N]`. Workgroup size 16×16; each
/// invocation accumulates one `C[row, col]`. Same source is reused by
/// the Vulkan backend via `naga` WGSL→SPIR-V.
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

/// Compile (once) + dispatch a SAXPY across `n` elements. Returns after
/// the command buffer is submitted; the queue is polled by the caller.
/// `x`, `y`, `params` are storage buffers the caller has already filled
/// via [`Context::storage_buffer_init`] / [`Context::storage_buffer`].
pub fn axpy_f32(
    ctx: &Context,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    params: &wgpu::Buffer,
    n: u32,
) {
    let pipeline = ComputePipeline::from_wgsl(ctx, SAXPY_WGSL, "main", 3);
    let groups = [(n + 63) / 64, 1, 1];
    dispatch(ctx, &pipeline, &[x, y, params], groups);
}

/// Dispatch a naive tiled GEMM across `[M, N]`. Caller owns `a`, `b`,
/// `c`, `dims`; see [`GEMM_F32_WGSL`] for the binding layout.
pub fn gemm_f32(
    ctx: &Context,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    c: &wgpu::Buffer,
    dims: &wgpu::Buffer,
    m: u32,
    n: u32,
) {
    let pipeline = ComputePipeline::from_wgsl(ctx, GEMM_F32_WGSL, "main", 4);
    let groups = [(n + 15) / 16, (m + 15) / 16, 1];
    dispatch(ctx, &pipeline, &[a, b, c, dims], groups);
}
