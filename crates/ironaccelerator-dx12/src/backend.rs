//! Direct3D 12 `Backend` impl. One descriptor per hardware adapter that
//! creates a device at feature level 11_0 or better.

use ironaccelerator_core::{
    Backend, BackendKind, Capability, CapabilityFlags, ComputeTier, DeviceDescriptor, DeviceId,
    Error, Result, Vendor,
};

use crate::drv::{
    EnumeratedAdapter, D3D_FEATURE_LEVEL_12_0, D3D_FEATURE_LEVEL_12_1, D3D_FEATURE_LEVEL_12_2,
};

pub struct Dx12Backend;
pub static DX12_BACKEND: Dx12Backend = Dx12Backend;

impl Backend for Dx12Backend {
    fn kind(&self) -> BackendKind {
        BackendKind::Dx12
    }

    fn is_available(&self) -> bool {
        crate::drv::is_available()
    }

    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>> {
        Ok(crate::drv::enumerate().into_iter().map(describe).collect())
    }

    fn capabilities(&self, device: u32) -> Result<CapabilityFlags> {
        crate::drv::enumerate()
            .into_iter()
            .find(|a| a.ordinal == device)
            .map(|a| flags_for(&a))
            .ok_or(Error::InvalidArgument("d3d12 adapter ordinal out of range"))
    }
}

/// Translate probed D3D12 feature support into the common flag space.
///
/// Deliberately conservative. D3D12 has no query for INT8 dot-product or for
/// matrix/tensor units, so those bits stay clear even on hardware that has
/// them — a false negative is recoverable, a false positive is not. `WMMA` in
/// particular is *not* implied by wave ops: wave intrinsics are cross-lane
/// shuffles, not a matrix engine.
fn flags_for(a: &EnumeratedAdapter) -> CapabilityFlags {
    let mut flags = CapabilityFlags::FP32 | CapabilityFlags::MULTI_STREAM;
    if a.fp64 {
        flags |= CapabilityFlags::FP64;
    }
    // Only native 16-bit math counts as FP16; min-precision is a storage hint
    // the driver may satisfy at FP32, so it would be a lie here.
    if a.native_16bit_ops {
        flags |= CapabilityFlags::FP16;
    }
    if a.uma {
        flags |= CapabilityFlags::UNIFIED_MEMORY;
    }
    flags
}

fn describe(a: EnumeratedAdapter) -> DeviceDescriptor {
    let vendor = match a.vendor_id {
        0x10DE => Vendor::Nvidia,
        0x1002 | 0x1022 => Vendor::Amd,
        0x8086 => Vendor::Intel,
        0x5143 => Vendor::Qualcomm,
        _ => Vendor::Other,
    };

    // Integrated parts report UMA; everything else is treated as a consumer
    // dGPU. D3D12 exposes nothing that distinguishes workstation or
    // datacenter silicon, so we do not guess at those tiers.
    let tier = if a.uma {
        ComputeTier::Mobile
    } else {
        ComputeTier::Consumer
    };

    let level = match a.feature_level {
        D3D_FEATURE_LEVEL_12_2 => "12_2",
        D3D_FEATURE_LEVEL_12_1 => "12_1",
        D3D_FEATURE_LEVEL_12_0 => "12_0",
        _ => "11_x",
    };

    let flags = flags_for(&a);
    DeviceDescriptor {
        id: DeviceId {
            backend: BackendKind::Dx12,
            ordinal: a.ordinal,
        },
        vendor,
        name: a.name,
        arch: format!("d3d{level}-{:04x}", a.device_id),
        // UMA parts report no dedicated VRAM; fall back to the shared budget
        // so the descriptor is not simply zero.
        total_memory_bytes: if a.dedicated_video_memory > 0 {
            a.dedicated_video_memory
        } else {
            a.shared_system_memory
        },
        // D3D12 exposes no SM/CU count. Wave lanes are the closest signal, but
        // they measure width, not count, so leave this unknown rather than
        // report a number that means something else.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_match_enumerated_descriptors() {
        let b = &DX12_BACKEND;
        for d in b.enumerate().unwrap() {
            let flags = b.capabilities(d.id.ordinal).unwrap();
            assert_eq!(flags, d.capability.flags);
            assert!(flags.contains(CapabilityFlags::FP32));
        }
    }

    #[test]
    fn unknown_ordinal_is_not_available() {
        assert!(DX12_BACKEND.capabilities(u32::MAX).is_err());
    }
}
