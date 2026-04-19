//! WebGPU `Backend` impl via `wgpu`. One descriptor per adapter.

use ironaccelerator_core::{
    Backend, BackendKind, Capability, CapabilityFlags, ComputeTier, DeviceDescriptor,
    DeviceId, Result, Strategy, Vendor, Workload,
};

use crate::drv::AdapterInfo;

pub struct WebGpuBackend;
pub static WEBGPU_BACKEND: WebGpuBackend = WebGpuBackend;

impl Backend for WebGpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::WebGpu
    }

    fn is_available(&self) -> bool {
        !crate::drv::enumerate().is_empty()
    }

    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>> {
        Ok(crate::drv::enumerate()
            .into_iter()
            .enumerate()
            .map(|(i, info)| describe(i as u32, info))
            .collect())
    }

    fn capabilities(&self, _device: u32) -> Result<CapabilityFlags> {
        Ok(CapabilityFlags::FP32 | CapabilityFlags::FP16 | CapabilityFlags::MULTI_STREAM)
    }

    fn plan(&self, _device: u32, _w: &Workload) -> Result<Strategy> {
        Ok(Strategy::Wgsl { workgroup: (64, 1, 1) })
    }
}

fn describe(ordinal: u32, info: AdapterInfo) -> DeviceDescriptor {
    let vendor = match info.vendor {
        0x10DE => Vendor::Nvidia,
        0x1002 | 0x1022 => Vendor::Amd,
        0x8086 => Vendor::Intel,
        0x106B => Vendor::Apple,
        0x5143 => Vendor::Qualcomm,
        _ => Vendor::Other,
    };
    let tier = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => ComputeTier::Consumer,
        wgpu::DeviceType::IntegratedGpu => ComputeTier::Mobile,
        wgpu::DeviceType::VirtualGpu => ComputeTier::Mobile,
        wgpu::DeviceType::Cpu => ComputeTier::Baseline,
        wgpu::DeviceType::Other => ComputeTier::Baseline,
    };
    let mut flags = CapabilityFlags::FP32 | CapabilityFlags::FP16 | CapabilityFlags::MULTI_STREAM;
    if info.subgroup_support {
        flags |= CapabilityFlags::WMMA;
    }
    let arch = format!(
        "wgpu-{}",
        match info.backend {
            wgpu::Backend::Vulkan => "vk",
            wgpu::Backend::Metal => "mtl",
            wgpu::Backend::Dx12 => "dx12",
            wgpu::Backend::Gl => "gl",
            wgpu::Backend::BrowserWebGpu => "web",
            wgpu::Backend::Empty => "none",
        }
    );
    DeviceDescriptor {
        id: DeviceId {
            backend: BackendKind::WebGpu,
            ordinal,
        },
        vendor,
        name: info.name,
        arch,
        total_memory_bytes: info.max_buffer_size,
        multiprocessor_count: 0,
        clock_khz: 0,
        capability: Capability {
            flags,
            tier,
            fp16_tflops: None,
            fp8_tflops: None,
            mem_bandwidth_gbs: None,
        },
    }
}
