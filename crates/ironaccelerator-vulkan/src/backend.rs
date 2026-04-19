//! Vulkan `Backend` impl. Enumerates physical devices via a process-wide
//! Vulkan 1.3 instance; reports unavailable when `libvulkan` cannot be loaded.

use ironaccelerator_core::{
    Backend, BackendKind, Capability, CapabilityFlags, ComputeTier, DeviceDescriptor,
    DeviceId, Result, Strategy, Vendor, Workload,
};

pub struct VulkanBackend;
pub static VULKAN_BACKEND: VulkanBackend = VulkanBackend;

impl Backend for VulkanBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Vulkan
    }

    fn is_available(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            !crate::drv::enumerate().is_empty()
        }
        #[cfg(target_arch = "wasm32")]
        {
            false
        }
    }

    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let pds = crate::drv::enumerate();
            Ok(pds.into_iter().map(describe).collect())
        }
        #[cfg(target_arch = "wasm32")]
        {
            Ok(Vec::new())
        }
    }

    fn capabilities(&self, _device: u32) -> Result<CapabilityFlags> {
        Ok(CapabilityFlags::FP32 | CapabilityFlags::FP16 | CapabilityFlags::MULTI_STREAM)
    }

    fn plan(&self, _device: u32, _w: &Workload) -> Result<Strategy> {
        Ok(Strategy::SpirvCompute { workgroup: (64, 1, 1) })
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn describe(pd: crate::drv::PhysicalDevice) -> DeviceDescriptor {
    use ash::vk;

    let vendor = match pd.vendor_id {
        0x10DE => Vendor::Nvidia,
        0x1002 | 0x1022 => Vendor::Amd,
        0x8086 => Vendor::Intel,
        0x106B => Vendor::Apple,
        0x5143 => Vendor::Qualcomm,
        _ => Vendor::Other,
    };

    let tier = match (pd.device_type, vendor) {
        (vk::PhysicalDeviceType::DISCRETE_GPU, _) => ComputeTier::Consumer,
        (vk::PhysicalDeviceType::INTEGRATED_GPU, _) => ComputeTier::Mobile,
        _ => ComputeTier::Baseline,
    };

    let mut flags = CapabilityFlags::FP32 | CapabilityFlags::MULTI_STREAM;
    if pd.shader_float16 {
        flags |= CapabilityFlags::FP16;
    }
    if pd.shader_int8 {
        flags |= CapabilityFlags::INT8;
    }
    if pd.cooperative_matrix {
        flags |= CapabilityFlags::WMMA | CapabilityFlags::TENSOR_CORES;
    }

    let arch = format!(
        "vk{}.{}",
        vk::api_version_major(pd.api_version),
        vk::api_version_minor(pd.api_version),
    );

    DeviceDescriptor {
        id: DeviceId {
            backend: BackendKind::Vulkan,
            ordinal: pd.ordinal,
        },
        vendor,
        name: pd.name,
        arch,
        total_memory_bytes: pd.heap_size_bytes,
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
