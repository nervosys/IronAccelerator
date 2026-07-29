//! Metal compute submission: buffers, a compute pipeline from a compiled
//! `.metallib`, and dispatch. Apple-only.
//!
//! This is the driver line, as everywhere else in the workspace: move bytes to
//! the device, run a shader you compiled, get bytes back. It does not compile
//! shaders — bring a `.metallib` (`xcrun metal -c k.metal … && xcrun metallib
//! …`), the same way the CUDA backend takes PTX and the Vulkan backend takes
//! SPIR-V.
//!
//! Buffers use `StorageModeShared`. On Apple silicon the CPU and GPU share
//! physical memory, so a shared buffer is device-resident and host-mappable at
//! once — [`upload`](ironaccelerator_core::ComputeDevice::upload) /
//! [`download`](ironaccelerator_core::ComputeDevice::download) are a `memcpy`
//! through the buffer's `contents()` pointer, with no staging copy.
//!
//! # Threadgroup size
//!
//! Metal sets the threadgroup size at dispatch, not in the shader — unlike the
//! `local_size` / `numthreads` that Vulkan, D3D12, and OpenGL bake into the
//! kernel. The unified [`ComputeDevice`] trait carries only threadgroup
//! *counts*, so its [`dispatch`](ComputeDevice::dispatch) assumes a 1-D group
//! of 64 threads, matching a `local_size_x = 64` kernel. For any other
//! geometry, call [`Context::dispatch_sized`] with an explicit
//! threads-per-threadgroup.

use std::sync::Arc;

use ironaccelerator_core::ComputeDevice;
use metal::{ComputePipelineState, MTLSize, NSUInteger};

use crate::drv::{Buffer, Device, Queue};

/// A compute context: one Metal device and a command queue on it.
pub struct Context {
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl Context {
    /// Bring up a context on the `ordinal`-th enumerated device.
    pub fn new(ordinal: u32) -> Option<Self> {
        let device = Device::all().into_iter().nth(ordinal as usize)?;
        let queue = device.new_queue();
        Some(Self { device, queue })
    }

    /// Bring up a context on the system-default device.
    pub fn system_default() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_queue();
        Some(Self { device, queue })
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    /// Build a compute pipeline from a compiled `.metallib`, taking the kernel
    /// named `entry`.
    pub fn pipeline_named(
        &self,
        metallib: &[u8],
        entry: &str,
    ) -> Result<ComputePipelineState, String> {
        let lib = self.device.raw().new_library_with_data(metallib)?;
        let func = lib.get_function(entry, None)?;
        self.device
            .raw()
            .new_compute_pipeline_state_with_function(&func)
    }

    /// Encode and dispatch `groups` threadgroups of `threads` each, then block
    /// until the GPU finishes. Buffers bind to indices `0..buffers.len()`.
    pub fn dispatch_sized(
        &self,
        pso: &ComputePipelineState,
        buffers: &[&Arc<Buffer>],
        groups: [u32; 3],
        threads: [u32; 3],
    ) {
        self.queue.scope(|cb| {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pso);
            for (i, b) in buffers.iter().enumerate() {
                enc.set_buffer(i as NSUInteger, Some(b.raw()), 0);
            }
            enc.dispatch_thread_groups(
                MTLSize {
                    width: groups[0] as NSUInteger,
                    height: groups[1] as NSUInteger,
                    depth: groups[2] as NSUInteger,
                },
                MTLSize {
                    width: threads[0] as NSUInteger,
                    height: threads[1] as NSUInteger,
                    depth: threads[2] as NSUInteger,
                },
            );
            enc.end_encoding();
        });
    }
}

/// Unified cross-backend compute surface. `code` is a compiled `.metallib`;
/// the kernel entry point is assumed to be `main`. See the module docs for the
/// threadgroup-size convention.
impl ComputeDevice for Context {
    type Buffer = Arc<Buffer>;
    type Pipeline = ComputePipelineState;
    type Error = String;

    fn device_buffer(&self, bytes: u64) -> Result<Arc<Buffer>, String> {
        self.device
            .new_buffer(bytes as usize, true)
            .map_err(|e| e.to_string())
    }

    fn upload(&self, data: &[u8]) -> Result<Arc<Buffer>, String> {
        let buf = self.device_buffer(data.len() as u64)?;
        // SAFETY: shared-storage buffer is host-visible; `contents()` is valid
        // for `data.len()` bytes (the buffer was sized to it).
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut u8, data.len());
        }
        Ok(buf)
    }

    fn download(&self, buffer: &Arc<Buffer>, out: &mut [u8]) -> Result<(), String> {
        let n = out.len().min(buffer.bytes());
        // SAFETY: shared-storage buffer is host-visible; reading `n` bytes stays
        // within the buffer's length.
        unsafe {
            std::ptr::copy_nonoverlapping(buffer.contents() as *const u8, out.as_mut_ptr(), n);
        }
        Ok(())
    }

    fn pipeline(&self, code: &[u8], _bindings: u32) -> Result<ComputePipelineState, String> {
        self.pipeline_named(code, "main")
    }

    fn dispatch(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[&Arc<Buffer>],
        groups: [u32; 3],
    ) -> Result<(), String> {
        self.dispatch_sized(pipeline, buffers, groups, [64, 1, 1]);
        Ok(())
    }

    fn buffer_len(&self, buffer: &Arc<Buffer>) -> u64 {
        buffer.bytes() as u64
    }
}
