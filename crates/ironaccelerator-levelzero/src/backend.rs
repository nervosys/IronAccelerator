//! Level Zero `Backend` impl. One entry per GPU or NPU device the loader
//! returns; `ze_device_type_t` distinguishes the two.

use ironaccelerator_core::{
    Backend, BackendKind, Capability, CapabilityFlags, ComputeTier, DeviceDescriptor,
    DeviceId, Result, Strategy, Vendor, Workload,
};

use crate::drv::{EnumeratedDevice, ZE_DEVICE_TYPE_GPU, ZE_DEVICE_TYPE_VPU};

pub struct LevelZeroBackend;
pub static LEVELZERO_BACKEND: LevelZeroBackend = LevelZeroBackend;

impl Backend for LevelZeroBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::LevelZero
    }

    fn is_available(&self) -> bool {
        crate::drv::is_available() && !crate::drv::enumerate().is_empty()
    }

    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>> {
        Ok(crate::drv::enumerate().into_iter().map(describe).collect())
    }

    fn capabilities(&self, _device: u32) -> Result<CapabilityFlags> {
        Ok(CapabilityFlags::FP32 | CapabilityFlags::FP16 | CapabilityFlags::INT8
            | CapabilityFlags::MULTI_STREAM)
    }

    fn plan(&self, _device: u32, _w: &Workload) -> Result<Strategy> {
        Ok(Strategy::LevelZero { device_type: "auto" })
    }
}

fn describe(d: EnumeratedDevice) -> DeviceDescriptor {
    let vendor = match d.vendor_id {
        0x8086 => Vendor::Intel,
        0x10DE => Vendor::Nvidia,
        0x1002 => Vendor::Amd,
        _ => Vendor::Other,
    };

    // Level Zero GPUs expose BF16 from Xe-HPG/HPC on; NPU (VPU) advertises
    // BF16 + INT8 from Meteor Lake onward. Expose the common superset.
    let mut flags = CapabilityFlags::FP32
        | CapabilityFlags::FP16
        | CapabilityFlags::BF16
        | CapabilityFlags::INT8
        | CapabilityFlags::MULTI_STREAM;
    let (tier, arch_prefix) = match d.type_ {
        ZE_DEVICE_TYPE_GPU => {
            flags |= CapabilityFlags::WMMA | CapabilityFlags::TENSOR_CORES | CapabilityFlags::INT4;
            (ComputeTier::Consumer, "xe")
        }
        ZE_DEVICE_TYPE_VPU => (ComputeTier::Mobile, "vpu"),
        _ => (ComputeTier::Baseline, "ze"),
    };

    DeviceDescriptor {
        id: DeviceId {
            backend: BackendKind::LevelZero,
            ordinal: d.ordinal,
        },
        vendor,
        name: d.name,
        arch: format!("{arch_prefix}-{:04x}", d.device_id),
        total_memory_bytes: d.max_mem_alloc_size,
        multiprocessor_count: d
            .num_slices
            .saturating_mul(d.num_subslices_per_slice)
            .saturating_mul(d.num_eus_per_subslice),
        clock_khz: d.core_clock_khz,
        capability: Capability {
            flags,
            tier,
            fp16_tflops: None,
            fp8_tflops: None,
            mem_bandwidth_gbs: None,
        },
    }
}
