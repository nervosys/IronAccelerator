//! OpenGL compute skeleton. A GL 4.3+ context must already be current
//! on the calling thread — hand it to [`crate::bind_current_context`]
//! before using anything here.
//!
//! The flow mirrors the Vulkan module: compile a GLSL compute shader
//! into a `Program`, allocate SSBOs with [`Ssbo::new`], bind them to
//! layout locations, and [`dispatch`]. Everything uses the cached
//! `glow::Context` stored by the driver module.

use glow::HasContext;

/// Upload + bind a shader-storage buffer object at `binding`.
pub struct Ssbo {
    pub id: <glow::Context as HasContext>::Buffer,
    pub size: u64,
}

impl Ssbo {
    /// Allocate an empty SSBO of `size` bytes with `STREAM_COPY` usage
    /// — the default for kernel-to-kernel intermediate tensors.
    pub fn new(gl: &glow::Context, size: u64) -> Result<Self, String> {
        unsafe {
            let id = gl.create_buffer().map_err(|e| e.to_string())?;
            gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(id));
            gl.buffer_data_size(
                glow::SHADER_STORAGE_BUFFER,
                size as i32,
                glow::STREAM_COPY,
            );
            gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, None);
            Ok(Ssbo { id, size })
        }
    }

    /// Upload `data` as the buffer's storage.
    pub fn with_data(gl: &glow::Context, data: &[u8]) -> Result<Self, String> {
        unsafe {
            let id = gl.create_buffer().map_err(|e| e.to_string())?;
            gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(id));
            gl.buffer_data_u8_slice(
                glow::SHADER_STORAGE_BUFFER,
                data,
                glow::STREAM_COPY,
            );
            gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, None);
            Ok(Ssbo {
                id,
                size: data.len() as u64,
            })
        }
    }

    pub fn bind(&self, gl: &glow::Context, binding: u32) {
        unsafe {
            gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, binding, Some(self.id));
        }
    }

    pub fn destroy(self, gl: &glow::Context) {
        unsafe { gl.delete_buffer(self.id) };
    }
}

/// Compiled + linked compute program.
pub struct Program {
    pub id: <glow::Context as HasContext>::Program,
}

impl Program {
    /// Compile `src` as a `#version 430` compute shader and link it
    /// into a program.
    pub fn from_glsl(gl: &glow::Context, src: &str) -> Result<Self, String> {
        unsafe {
            let shader = gl
                .create_shader(glow::COMPUTE_SHADER)
                .map_err(|e| e.to_string())?;
            gl.shader_source(shader, src);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                let log = gl.get_shader_info_log(shader);
                gl.delete_shader(shader);
                return Err(format!("compute shader compile: {log}"));
            }
            let prog = gl.create_program().map_err(|e| e.to_string())?;
            gl.attach_shader(prog, shader);
            gl.link_program(prog);
            gl.delete_shader(shader);
            if !gl.get_program_link_status(prog) {
                let log = gl.get_program_info_log(prog);
                gl.delete_program(prog);
                return Err(format!("program link: {log}"));
            }
            Ok(Program { id: prog })
        }
    }

    pub fn destroy(self, gl: &glow::Context) {
        unsafe { gl.delete_program(self.id) };
    }
}

/// Dispatch `num_groups` workgroups and issue a shader-storage memory
/// barrier so the following read sees the writes. The caller is
/// responsible for binding SSBOs via [`Ssbo::bind`] first.
pub fn dispatch(
    gl: &glow::Context,
    program: &Program,
    num_groups: [u32; 3],
) {
    unsafe {
        gl.use_program(Some(program.id));
        gl.dispatch_compute(num_groups[0], num_groups[1], num_groups[2]);
        gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT);
    }
}

/// Access the cached `glow::Context`. Returns `None` if no GL context
/// has been bound.
pub fn gl() -> Option<&'static glow::Context> {
    crate::drv::shared_context()
}
