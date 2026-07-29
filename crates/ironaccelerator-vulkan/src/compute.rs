//! Vulkan compute dispatch skeleton.
//!
//! This is the minimum path from "I have a SPIR-V binary and two storage
//! buffers" to "it ran on the GPU":
//!
//! 1. [`Context::new`] picks a device by ordinal from the enumerated set,
//!    creates an `ash::Device` + compute queue, and wires up a
//!    command-pool.
//! 2. [`Buffer::device_local`] / [`Buffer::host_visible`] allocate and
//!    bind storage buffers; [`Context::upload`] / [`Context::download`]
//!    stage host bytes to and from device-local memory in one call.
//! 3. [`ComputePipeline::new`] takes SPIR-V + entry-point name + storage
//!    buffer count, builds a descriptor-set layout, pipeline layout,
//!    and `vkPipeline`.
//! 4. [`Context::dispatch`] records + submits a one-shot command buffer
//!    that binds the pipeline + descriptor set and dispatches a
//!    workgroup grid, then waits.
//!
//! This is deliberately minimal: no descriptor-pool reuse, no pipeline
//! cache, no push constants. Higher layers cache those once they have a
//! reason to; the driver line only needs the one-shot path to be correct.

use core::ffi::c_void;

use ash::{vk, Device, Instance};

use crate::drv;

/// Compute-only Vulkan context bound to one physical device. The
/// underlying `Entry` + `Instance` are process-wide and outlive the
/// context; dropping a `Context` destroys only the logical device and
/// its pool.
pub struct Context {
    instance: &'static Instance,
    pub physical: vk::PhysicalDevice,
    pub device: Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub command_pool: vk::CommandPool,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
}

impl Context {
    /// Pick device `ordinal` from the enumerated list and bring up a
    /// compute queue on it. Returns `None` on any failure (missing ICD,
    /// ordinal out of range, queue creation failed). Prefer this over
    /// panicking — planners often probe.
    pub fn new(ordinal: u32) -> Option<Self> {
        let (_entry, instance) = drv::own_instance()?;
        let physicals = unsafe { instance.enumerate_physical_devices().ok()? };
        let physical = *physicals.get(ordinal as usize)?;

        let qfp = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let queue_family =
            qfp.iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))? as u32;

        let priorities = [1.0f32];
        let qci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities);
        let queue_infos = [qci];
        let dci = vk::DeviceCreateInfo::default().queue_create_infos(&queue_infos);
        let device = unsafe { instance.create_device(physical, &dci, None).ok()? };
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None).ok()? };

        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };

        Some(Context {
            instance,
            physical,
            device,
            queue,
            queue_family,
            command_pool,
            mem_props,
        })
    }

    fn find_memory_type(&self, bits: u32, required: vk::MemoryPropertyFlags) -> Option<u32> {
        for i in 0..self.mem_props.memory_type_count {
            if bits & (1 << i) != 0 {
                let props = self.mem_props.memory_types[i as usize].property_flags;
                if props.contains(required) {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Allocate a one-shot primary command buffer, let `record` fill it, then
    /// submit and block on `vkQueueWaitIdle`. The single choke-point every
    /// submitting method here goes through, so barrier and lifetime rules live
    /// in one place.
    fn run_commands(
        &self,
        record: impl FnOnce(&Device, vk::CommandBuffer),
    ) -> Result<(), vk::Result> {
        unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cbs = self.device.allocate_command_buffers(&alloc_info)?;
            let cb = cbs[0];
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cb, &begin)?;
            record(&self.device, cb);
            self.device.end_command_buffer(cb)?;

            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            self.device
                .queue_submit(self.queue, &[submit], vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;
            self.device.free_command_buffers(self.command_pool, &cbs);
        }
        Ok(())
    }

    /// Copy `bytes` from `src` to `dst` on the device and wait. Both buffers
    /// must be at least `bytes` long.
    pub fn copy_buffer(&self, src: &Buffer, dst: &Buffer, bytes: u64) -> Result<(), vk::Result> {
        self.run_commands(|device, cb| unsafe {
            let region = vk::BufferCopy::default().size(bytes);
            device.cmd_copy_buffer(cb, src.buffer, dst.buffer, &[region]);
        })
    }

    /// Stage `data` through a host-visible buffer into a fresh device-local
    /// buffer and wait for the copy. The mirror of [`Self::download`], and the
    /// device-side analogue of cudarc's `htod_sync_copy`.
    pub fn upload(&self, data: &[u8]) -> Result<Buffer, vk::Result> {
        let staging = Buffer::host_visible(self, data.len() as u64)?;
        staging.write_bytes(data)?;
        let dst = Buffer::device_local(self, data.len() as u64)?;
        self.copy_buffer(&staging, &dst, data.len() as u64)?;
        Ok(dst)
    }

    /// Read a device-local buffer back to host memory through a host-visible
    /// staging buffer. Copies `min(out.len(), src.size)` bytes.
    pub fn download(&self, src: &Buffer, out: &mut [u8]) -> Result<(), vk::Result> {
        let n = (out.len() as u64).min(src.size);
        let staging = Buffer::host_visible(self, n)?;
        self.copy_buffer(src, &staging, n)?;
        staging.read_bytes(out)
    }

    /// Record + submit a one-shot compute command buffer and wait.
    ///
    /// A buffer memory barrier up front makes any prior transfer or host writes
    /// visible to the shader — [`Self::upload`] followed by `dispatch` is the
    /// common path, and without the barrier the shader may read stale memory on
    /// a discrete GPU even though the copy's `vkQueueWaitIdle` has returned.
    pub fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        group_count: [u32; 3],
    ) -> Result<(), vk::Result> {
        self.run_commands(|device, cb| unsafe {
            let pre = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[pre],
                &[],
                &[],
            );
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &[pipeline.descriptor_set],
                &[],
            );
            device.cmd_dispatch(cb, group_count[0], group_count[1], group_count[2]);
            // Make shader writes available to the transfer that reads them back.
            let post = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ);
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[post],
                &[],
                &[],
            );
        })
    }

    pub fn instance(&self) -> &Instance {
        self.instance
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
        }
    }
}

// ── Buffers ───────────────────────────────────────────────────────────────

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    host_visible: bool,
    device: Device,
}

impl Buffer {
    pub fn device_local(ctx: &Context, size: u64) -> Result<Self, vk::Result> {
        Self::alloc(
            ctx,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
    }

    pub fn host_visible(ctx: &Context, size: u64) -> Result<Self, vk::Result> {
        Self::alloc(
            ctx,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }

    /// `true` when this buffer's memory is CPU-mappable — the precondition for
    /// [`Self::write_bytes`], [`Self::read_bytes`], and [`Self::map`].
    pub fn is_host_visible(&self) -> bool {
        self.host_visible
    }

    fn alloc(
        ctx: &Context,
        size: u64,
        usage: vk::BufferUsageFlags,
        props: vk::MemoryPropertyFlags,
    ) -> Result<Self, vk::Result> {
        unsafe {
            let bci = vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = ctx.device.create_buffer(&bci, None)?;
            let req = ctx.device.get_buffer_memory_requirements(buffer);
            let mem_type = ctx
                .find_memory_type(req.memory_type_bits, props)
                .ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
            let mai = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mem_type);
            let memory = ctx.device.allocate_memory(&mai, None)?;
            ctx.device.bind_buffer_memory(buffer, memory, 0)?;
            Ok(Buffer {
                buffer,
                memory,
                size,
                host_visible: props.contains(vk::MemoryPropertyFlags::HOST_VISIBLE),
                device: ctx.device.clone(),
            })
        }
    }

    /// Map a host-visible buffer; returns a raw pointer valid until
    /// [`Self::unmap`]. Caller upholds aliasing / lifetime invariants.
    ///
    /// # Safety
    /// Buffer must have been allocated with `HOST_VISIBLE`.
    pub unsafe fn map(&self) -> Result<*mut c_void, vk::Result> {
        self.device
            .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
    }

    pub unsafe fn unmap(&self) {
        self.device.unmap_memory(self.memory);
    }

    /// Copy `data` into a host-visible buffer. Bytes past `self.size` are
    /// dropped — the buffer is never grown. Errors if the buffer is
    /// `DEVICE_LOCAL`; use [`Context::upload`] to stage into device memory.
    ///
    /// Allocated with `HOST_COHERENT`, so no explicit flush is needed.
    pub fn write_bytes(&self, data: &[u8]) -> Result<(), vk::Result> {
        if !self.host_visible {
            return Err(vk::Result::ERROR_MEMORY_MAP_FAILED);
        }
        let n = data.len().min(self.size as usize);
        unsafe {
            let p = self.map()?;
            core::ptr::copy_nonoverlapping(data.as_ptr(), p as *mut u8, n);
            self.unmap();
        }
        Ok(())
    }

    /// Copy out of a host-visible buffer into `out`. Reads
    /// `min(out.len(), self.size)` bytes.
    pub fn read_bytes(&self, out: &mut [u8]) -> Result<(), vk::Result> {
        if !self.host_visible {
            return Err(vk::Result::ERROR_MEMORY_MAP_FAILED);
        }
        let n = out.len().min(self.size as usize);
        unsafe {
            let p = self.map()?;
            core::ptr::copy_nonoverlapping(p as *const u8, out.as_mut_ptr(), n);
            self.unmap();
        }
        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

// ── Compute pipeline ──────────────────────────────────────────────────────

pub struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub dsl: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    pub module: vk::ShaderModule,
    device: Device,
}

impl ComputePipeline {
    /// `spirv` is the raw SPIR-V binary (little-endian u32 stream cast to
    /// bytes is fine). `entry` is the shader entry point. `buffers` are
    /// the storage buffers bound at slots `0..buffers.len()`.
    ///
    /// A convenience over [`Self::with_bindings`] + [`Self::bind_buffers`] for
    /// the case where the buffers are known up front and never rebound.
    pub fn new(
        ctx: &Context,
        spirv: &[u32],
        entry: &std::ffi::CStr,
        buffers: &[&Buffer],
    ) -> Result<Self, vk::Result> {
        let pipeline = Self::with_bindings(ctx, spirv, entry, buffers.len() as u32)?;
        pipeline.bind_buffers(buffers);
        Ok(pipeline)
    }

    /// Build a pipeline whose descriptor set has `bindings` storage-buffer
    /// slots but no buffers written yet. Bind them later with
    /// [`Self::bind_buffers`] — this is the path the unified
    /// [`ComputeDevice`](ironaccelerator_core::ComputeDevice) trait takes, where
    /// the concrete buffers arrive at dispatch time rather than pipeline
    /// creation.
    pub fn with_bindings(
        ctx: &Context,
        spirv: &[u32],
        entry: &std::ffi::CStr,
        bindings: u32,
    ) -> Result<Self, vk::Result> {
        unsafe {
            let module_ci = vk::ShaderModuleCreateInfo::default().code(spirv);
            let module = ctx.device.create_shader_module(&module_ci, None)?;

            let layout_bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..bindings)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
            let dsl = ctx.device.create_descriptor_set_layout(&dsl_ci, None)?;

            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(bindings.max(1))];
            let pool_ci = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            let pool = ctx.device.create_descriptor_pool(&pool_ci, None)?;

            let dsls = [dsl];
            let ds_alloc = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&dsls);
            let descriptor_set = ctx.device.allocate_descriptor_sets(&ds_alloc)?[0];

            let pl_ci = vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls);
            let layout = ctx.device.create_pipeline_layout(&pl_ci, None)?;

            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(entry);
            let pipeline_ci = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout);
            let pipeline = ctx
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
                .map_err(|(_, r)| r)?[0];

            Ok(ComputePipeline {
                pipeline,
                layout,
                dsl,
                pool,
                descriptor_set,
                module,
                device: ctx.device.clone(),
            })
        }
    }

    /// Point the descriptor set at `buffers`, bound to slots
    /// `0..buffers.len()`. Overwrites any previous binding. Safe to call
    /// between dispatches provided the previous dispatch has completed — the
    /// one-shot submission path here always waits, so that holds.
    pub fn bind_buffers(&self, buffers: &[&Buffer]) {
        let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|b| {
                vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer)
                    .offset(0)
                    .range(b.size)
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device.destroy_descriptor_set_layout(self.dsl, None);
            self.device.destroy_shader_module(self.module, None);
        }
    }
}

/// Unified cross-backend compute surface. SPIR-V arrives as bytes (a `u32`
/// word stream); the entry point is assumed to be `main`.
impl ironaccelerator_core::ComputeDevice for Context {
    type Buffer = Buffer;
    type Pipeline = ComputePipeline;
    type Error = vk::Result;

    fn device_buffer(&self, bytes: u64) -> Result<Buffer, vk::Result> {
        Buffer::device_local(self, bytes)
    }

    fn upload(&self, data: &[u8]) -> Result<Buffer, vk::Result> {
        Context::upload(self, data)
    }

    fn download(&self, buffer: &Buffer, out: &mut [u8]) -> Result<(), vk::Result> {
        Context::download(self, buffer, out)
    }

    fn pipeline(&self, code: &[u8], bindings: u32) -> Result<ComputePipeline, vk::Result> {
        if code.len() % 4 != 0 {
            // Not a whole SPIR-V word stream.
            return Err(vk::Result::ERROR_INITIALIZATION_FAILED);
        }
        let words: Vec<u32> = code
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        ComputePipeline::with_bindings(self, &words, c"main", bindings)
    }

    fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[&Buffer],
        groups: [u32; 3],
    ) -> Result<(), vk::Result> {
        pipeline.bind_buffers(buffers);
        Context::dispatch(self, pipeline, groups)
    }

    fn buffer_len(&self, buffer: &Buffer) -> u64 {
        buffer.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A context on device 0, or `None` when the host has no Vulkan device —
    /// which is every CI runner without a GPU or an ICD. Tests then no-op
    /// rather than fail.
    fn ctx() -> Option<Context> {
        if crate::drv::enumerate().is_empty() {
            return None;
        }
        Context::new(0)
    }

    #[test]
    fn context_builds_on_every_enumerated_device() {
        for pd in crate::drv::enumerate() {
            // Not every physical device exposes a compute queue (some display
            // adapters do not); skip those rather than assert.
            if pd.compute_queue_family.is_none() {
                continue;
            }
            let c = Context::new(pd.ordinal).expect("compute queue but context failed");
            assert_ne!(c.queue, vk::Queue::null());
        }
    }

    #[test]
    fn host_visible_buffer_round_trips_bytes() {
        let Some(c) = ctx() else { return };
        let src: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        let buf = Buffer::host_visible(&c, src.len() as u64).expect("host-visible alloc");
        assert!(buf.is_host_visible());
        buf.write_bytes(&src).expect("write");
        let mut out = vec![0u8; src.len()];
        buf.read_bytes(&mut out).expect("read");
        assert_eq!(out, src, "host-visible round-trip corrupted data");
    }

    #[test]
    fn device_local_round_trips_via_staging() {
        let Some(c) = ctx() else { return };
        let src: Vec<u8> = (0..8192u32).map(|i| (i * 7 % 253) as u8).collect();
        let dev = c.upload(&src).expect("upload");
        assert_eq!(dev.size, src.len() as u64);
        assert!(!dev.is_host_visible(), "upload must return device-local memory");
        let mut out = vec![0u8; src.len()];
        c.download(&dev, &mut out).expect("download");
        assert_eq!(out, src, "device round-trip corrupted data");
    }

    #[test]
    fn device_local_rejects_direct_host_access() {
        let Some(c) = ctx() else { return };
        let dev = Buffer::device_local(&c, 64).expect("device-local alloc");
        assert!(dev.write_bytes(&[0u8; 64]).is_err());
        assert!(dev.read_bytes(&mut [0u8; 64]).is_err());
    }
}
