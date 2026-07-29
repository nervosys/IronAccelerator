//! [`ComputeDevice`] — the one compute-submission surface shared by every
//! backend that owns a device handle.
//!
//! [`Backend`](crate::Backend) answers *what hardware is here and what can it
//! do*. `ComputeDevice` answers the next question — *get bytes onto it, run a
//! shader I compiled, get bytes back* — with a single trait the Vulkan, D3D12,
//! OpenGL, Metal, and Level Zero backends all implement. Code written against
//! `C: ComputeDevice` runs unchanged on any of them:
//!
//! ```no_run
//! use ironaccelerator_core::ComputeDevice;
//!
//! fn double_in_place<C: ComputeDevice>(dev: &C, code: &[u8]) -> Result<Vec<f32>, C::Error> {
//!     let input: Vec<u8> = (0..256u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
//!     let buf = dev.upload(&input)?;
//!     let pipe = dev.pipeline(code, 1)?;              // one storage buffer at slot 0
//!     dev.dispatch(&pipe, &[&buf], [256 / 64, 1, 1])?;
//!     let mut out = vec![0u8; input.len()];
//!     dev.download(&buf, &mut out)?;
//!     Ok(out.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
//! }
//! ```
//!
//! # It stays at the driver line
//!
//! The trait moves bytes and launches caller-supplied bytecode. It does not
//! compile shaders, choose work sizes, or describe workloads — those are
//! consumer concerns, exactly as with [`Backend`](crate::Backend). The shape is
//! deliberately the least-common-denominator of the three backends: flat
//! storage buffers bound at consecutive slots `0..bindings`, one entry point,
//! host-blocking submission.
//!
//! # Bytecode is backend-native
//!
//! [`pipeline`](ComputeDevice::pipeline) takes the format the backend's driver
//! consumes — there is no translation layer here:
//!
//! | backend | `code` is | entry point |
//! |---------|-----------|-------------|
//! | Vulkan  | SPIR-V (a little-endian `u32` word stream as bytes; length a multiple of 4) | `main` |
//! | D3D12   | a signed DXIL container (`dxc -T cs_6_0 …`) | shader's own |
//! | OpenGL  | GLSL compute-shader source, UTF-8 (`#version 430`+) | `main` |
//! | Metal   | a compiled `.metallib` (`xcrun metal … && xcrun metallib …`) | first kernel (MSL reserves `main`) |
//! | Level Zero | OpenCL/SYCL-flavored SPIR-V — `Kernel` model, pointer args (not Vulkan's `GLCompute` SPIR-V) | `main` |
//!
//! The `bindings` count is how many storage buffers the shader declares at
//! slots `0..bindings` (`binding = N` in SPIR-V/GLSL, `register(uN)` in HLSL,
//! `[[buffer(N)]]` in Metal); [`dispatch`](ComputeDevice::dispatch) binds
//! exactly that many in order.
//!
//! One wrinkle: Metal and Level Zero set the group size at dispatch, where the
//! others declare it in the shader (`local_size` / `numthreads`). The trait
//! carries only threadgroup counts, so those two impls assume a 1-D group of
//! 64; each exposes a native call taking an explicit size for other cases.
//!
//! # Why WebGPU is not here
//!
//! `ironaccelerator-webgpu` does not implement this trait. WebGPU's `GPUDevice`
//! is owned by the host page and its submission API is asynchronous and
//! JS-side; wrapping it would mean re-exporting a binding crate's types and
//! taking on the very dependency that backend was built to avoid. There is no
//! device handle for the crate to hang an impl on, so it stays a host-driven
//! adapter survey.

/// A device you can allocate on, submit a compute shader to, and read back
/// from — one uniform surface over Vulkan, D3D12, and OpenGL.
///
/// Implemented on the backend's owned device/context type (Vulkan `Context`,
/// D3D12 `Context`, OpenGL `GlDevice`, Metal `Context`, Level Zero `Context`).
/// Every method blocks until the GPU has finished the operation, matching the
/// one-shot submission model the backends already use; batching and async are
/// concrete-type concerns layered on top.
///
/// The associated types keep this zero-cost — there is no boxing and no
/// vtable. The trait is therefore not `dyn`-safe by design; use it as a generic
/// bound (`fn run<C: ComputeDevice>(dev: &C)`), not a trait object.
pub trait ComputeDevice {
    /// Backend-native buffer handle (device-local storage).
    type Buffer;

    /// Backend-native compiled compute pipeline.
    type Pipeline;

    /// Backend-native error. `Debug` so generic callers can surface it; the
    /// concrete backends carry richer error types on their inherent APIs.
    type Error: core::fmt::Debug;

    /// Allocate an uninitialized device-local buffer of `bytes`.
    fn device_buffer(&self, bytes: u64) -> Result<Self::Buffer, Self::Error>;

    /// Stage `data` into a fresh device-local buffer and wait for the copy.
    /// The buffer is sized to `data.len()`.
    fn upload(&self, data: &[u8]) -> Result<Self::Buffer, Self::Error>;

    /// Copy a device-local buffer back to host memory, reading
    /// `min(out.len(), len)` bytes.
    fn download(&self, buffer: &Self::Buffer, out: &mut [u8]) -> Result<(), Self::Error>;

    /// Compile `code` (see the module table for the per-backend format) into a
    /// pipeline that binds `bindings` storage buffers at slots `0..bindings`.
    fn pipeline(&self, code: &[u8], bindings: u32) -> Result<Self::Pipeline, Self::Error>;

    /// Bind `buffers` to slots `0..buffers.len()` and dispatch a
    /// `groups[0] × groups[1] × groups[2]` grid of workgroups, then block until
    /// it completes. `buffers.len()` should match the pipeline's `bindings`.
    fn dispatch(
        &self,
        pipeline: &Self::Pipeline,
        buffers: &[&Self::Buffer],
        groups: [u32; 3],
    ) -> Result<(), Self::Error>;

    /// Length of a buffer in bytes.
    fn buffer_len(&self, buffer: &Self::Buffer) -> u64;
}
