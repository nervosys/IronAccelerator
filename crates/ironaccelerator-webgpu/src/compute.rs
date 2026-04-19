//! WebGPU compute skeleton: pick an adapter, request a device, expose
//! helpers for storage buffers + compute pipelines + dispatch.
//!
//! On WASM the browser hands us a device directly via
//! [`crate::drv::bind_device`]; [`Context::from_bound`] uses that path.
//! On native (or on WASM when no preselected device was bound) we run
//! the normal `pollster::block_on` adapter request.

use pollster::block_on;
use wgpu::util::DeviceExt;

/// Compute-only WebGPU context. Owns a `Device` + `Queue` pair.
pub struct Context {
    pub adapter_info: crate::drv::AdapterInfo,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Context {
    /// Bring up the `ordinal`-th enumerable adapter with minimal limits
    /// and no extra features.
    pub fn new(ordinal: u32) -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all());
        let adapter = adapters.into_iter().nth(ordinal as usize)?;
        let info = crate::drv::describe(&adapter);
        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ironaccelerator-webgpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;
        Some(Context {
            adapter_info: info,
            device,
            queue,
        })
    }

    /// Upload initial data to a new read-write storage buffer.
    pub fn storage_buffer_init(&self, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ia-storage"),
            contents: data,
            usage: usage | wgpu::BufferUsages::STORAGE,
        })
    }

    /// Create an empty storage buffer of `size` bytes.
    pub fn storage_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ia-storage"),
            size,
            usage: usage | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }
}

// ── Compute pipeline ──────────────────────────────────────────────────────

pub struct ComputePipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl ComputePipeline {
    /// Build a pipeline from WGSL source with `buffer_count` storage
    /// bindings at slots `0..buffer_count`. `entry` is the shader entry
    /// point name.
    pub fn from_wgsl(
        ctx: &Context,
        wgsl: &str,
        entry: &str,
        buffer_count: u32,
    ) -> Self {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ia-wgsl"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..buffer_count)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("ia-bgl"),
                    entries: &entries,
                });
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ia-pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ia-cp"),
                layout: Some(&layout),
                module: &module,
                entry_point: entry,
                compilation_options: Default::default(),
                cache: None,
            });
        ComputePipeline {
            pipeline,
            bind_group_layout,
        }
    }
}

/// Record + submit a single dispatch. `buffers` bind to slots
/// `0..buffers.len()`.
pub fn dispatch(
    ctx: &Context,
    pipeline: &ComputePipeline,
    buffers: &[&wgpu::Buffer],
    group_count: [u32; 3],
) {
    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ia-bg"),
        layout: &pipeline.bind_group_layout,
        entries: &entries,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ia-enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ia-cp"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(group_count[0], group_count[1], group_count[2]);
    }
    ctx.queue.submit(Some(encoder.finish()));
}
