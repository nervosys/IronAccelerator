//! Reference GLSL compute kernels. Mirrors the WebGPU / Vulkan SAXPY
//! — `y[i] = alpha * x[i] + y[i]` on `f32`.

use glow::HasContext;

use crate::compute::{dispatch, Program, Ssbo};

/// SAXPY source. Binding 0 = `x` (read), binding 1 = `y` (read-write).
/// `alpha` and `n` are uniforms.
pub const SAXPY_GLSL: &str = r#"#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer X { float x[]; };
layout(std430, binding = 1)          buffer Y { float y[]; };
uniform float uAlpha;
uniform uint  uN;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uN) return;
    y[i] = uAlpha * x[i] + y[i];
}
"#;

/// Compile + dispatch a SAXPY across `n` elements. The caller owns `x`
/// and `y`; they must already hold `n * 4` bytes each.
pub fn axpy_f32(gl: &glow::Context, x: &Ssbo, y: &Ssbo, alpha: f32, n: u32) -> Result<(), String> {
    let program = Program::from_glsl(gl, SAXPY_GLSL)?;
    unsafe {
        gl.use_program(Some(program.id));
        let loc_alpha = gl.get_uniform_location(program.id, "uAlpha");
        let loc_n = gl.get_uniform_location(program.id, "uN");
        gl.uniform_1_f32(loc_alpha.as_ref(), alpha);
        gl.uniform_1_u32(loc_n.as_ref(), n);
    }
    x.bind(gl, 0);
    y.bind(gl, 1);
    let groups = [n.div_ceil(64), 1, 1];
    dispatch(gl, &program, groups);
    program.destroy(gl);
    Ok(())
}
