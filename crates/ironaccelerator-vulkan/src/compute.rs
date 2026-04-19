//! Vulkan compute dispatch skeleton.
//!
//! This is the minimum path from "I have a SPIR-V binary and two storage
//! buffers" to "it ran on the GPU":
//!
//! 1. [`Context::new`] picks a device by ordinal from the enumerated set,
//!    creates an `ash::Device` + compute queue, and wires up a
//!    command-pool.
//! 2. [`Buffer::device_local`] / [`Buffer::host_visible`] allocate and
//!    bind storage buffers.
//! 3. [`ComputePipeline::new`] takes SPIR-V + entry-point name + storage
//!    buffer count, builds a descriptor-set layout, pipeline layout,
//!    and `vkPipeline`.
//! 4. [`Context::dispatch`] records + submits a one-shot command buffer
//!    that binds the pipeline + descriptor set and dispatches a
//!    workgroup grid, then waits.
//!
//! This is deliberately scaffolding: no descriptor-pool reuse, no
//! pipeline cache, no push constants. Higher layers (a `GemmPlan`-style
//! object) will cache everything once real kernels land.

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
        let queue_family = qfp
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))?
            as u32;

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

    /// Record + submit a one-shot compute command buffer and wait.
    pub fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        group_count: [u32; 3],
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
            self.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &[pipeline.descriptor_set],
                &[],
            );
            self.device
                .cmd_dispatch(cb, group_count[0], group_count[1], group_count[2]);
            self.device.end_command_buffer(cb)?;

            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            self.device
                .queue_submit(self.queue, &[submit], vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;
            self.device.free_command_buffers(self.command_pool, &cbs);
        }
        Ok(())
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
                device: ctx.device.clone(),
            })
        }
    }

    /// Map a host-visible buffer; returns a raw pointer valid until
    /// [`unmap`]. Caller upholds aliasing / lifetime invariants.
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
    pub fn new(
        ctx: &Context,
        spirv: &[u32],
        entry: &std::ffi::CStr,
        buffers: &[&Buffer],
    ) -> Result<Self, vk::Result> {
        unsafe {
            let module_ci = vk::ShaderModuleCreateInfo::default().code(spirv);
            let module = ctx.device.create_shader_module(&module_ci, None)?;

            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..buffers.len() as u32)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            let dsl = ctx.device.create_descriptor_set_layout(&dsl_ci, None)?;

            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(buffers.len() as u32)];
            let pool_ci = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            let pool = ctx.device.create_descriptor_pool(&pool_ci, None)?;

            let dsls = [dsl];
            let ds_alloc = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&dsls);
            let descriptor_set = ctx.device.allocate_descriptor_sets(&ds_alloc)?[0];

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
                        .dst_set(descriptor_set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();
            ctx.device.update_descriptor_sets(&writes, &[]);

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
