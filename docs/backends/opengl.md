# OpenGL backend

Crate: `ironaccelerator-opengl`. Uses `glow 0.14`.

## Build time

No SDK. Glow loads GL symbols through a host-supplied loader closure.

## Runtime

- GL **4.3+** context **already current** on the calling thread.
  IronAccelerator does not spin up a window — hand your existing
  `glfw` / `winit` / `sdl` context in via:

  ```rust
  unsafe {
      ironaccelerator_opengl::bind_current_context(|s| {
          window.get_proc_address(s) as *const _
      });
  }
  ```

- Intended as a legacy / embedded-GPU fallback (older Mesa, locked-down
  integrated Intel kernels that don't expose Vulkan).

## Capabilities

- `compute::Program` (GLSL `#version 430` compute shader compile +
  link).
- `compute::Ssbo` (SSBO + `glBindBufferBase`).
- `compute::dispatch` with `SHADER_STORAGE_BARRIER_BIT`.
- `kernels::axpy_f32` — SAXPY reference kernel.
