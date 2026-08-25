//! Metal `Backend` impl. Enumerates live `MTLDevice`s on Apple hosts; on
//! non-Apple hosts the backend reports unavailable and enumerates nothing.

use ironaccelerator_core::{Backend, BackendKind, CapabilityFlags, DeviceDescriptor, Result};
#[cfg(target_vendor = "apple")]
use ironaccelerator_core::{Capability, ComputeTier, DeviceId, Vendor};

pub struct MetalBackend;
pub static METAL_BACKEND: MetalBackend = MetalBackend;

impl Backend for MetalBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Metal
    }

    fn is_available(&self) -> bool {
        #[cfg(target_vendor = "apple")]
        {
            !crate::drv::Device::all().is_empty()
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            false
        }
    }

    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>> {
        #[cfg(target_vendor = "apple")]
        {
            let devices = crate::drv::Device::all();
            let mut out = Vec::with_capacity(devices.len());
            for (ord, d) in devices.iter().enumerate() {
                let fam = d.apple_family();
                let (flags, tier) = capability_for(fam);
                let arch = if fam > 0 {
                    format!("apple-g{fam}")
                } else {
                    "metal-discrete".to_string()
                };
                out.push(DeviceDescriptor {
                    id: DeviceId {
                        backend: BackendKind::Metal,
                        ordinal: ord as u32,
                    },
                    vendor: Vendor::Apple,
                    name: d.name(),
                    arch,
                    total_memory_bytes: d.recommended_max_working_set_size(),
                    multiprocessor_count: 0,
                    clock_khz: 0,
                    capability: Capability {
                        flags,
                        tier,
                        fp16_tflops: None,
                        fp8_tflops: None,
                        mem_bandwidth_gbs: None,
                    },
                });
            }
            Ok(out)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            Ok(Vec::new())
        }
    }

    fn capabilities(&self, device: u32) -> Result<CapabilityFlags> {
        // Capabilities are per-family (BF16/INT8/INT4 vary), so this must derive
        // from the same `capability_for` path `enumerate()` uses for this exact
        // ordinal — a hardcoded set would disagree with the descriptor on any
        // device whose family differs from the constant. An out-of-range ordinal
        // is a typed error rather than a plausible-looking answer.
        #[cfg(target_vendor = "apple")]
        {
            let devices = crate::drv::Device::all();
            let d = devices.get(device as usize).ok_or(
                ironaccelerator_core::Error::InvalidArgument("metal device ordinal out of range"),
            )?;
            Ok(capability_for(d.apple_family()).0)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = device;
            Err(ironaccelerator_core::Error::InvalidArgument(
                "metal device ordinal out of range",
            ))
        }
    }
}

/// Capability bits + compute tier for a given Apple GPU family ordinal.
/// Family 9 is M3/M4, 8 is M2, 7 is M1/A14+, 6 is A13.
#[cfg(target_vendor = "apple")]
fn capability_for(family: u32) -> (CapabilityFlags, ComputeTier) {
    let base = CapabilityFlags::FP32
        | CapabilityFlags::FP16
        | CapabilityFlags::UNIFIED_MEMORY
        | CapabilityFlags::ANE
        | CapabilityFlags::MULTI_STREAM;
    match family {
        9 | 8 => (
            base | CapabilityFlags::BF16 | CapabilityFlags::INT8 | CapabilityFlags::INT4,
            ComputeTier::Workstation,
        ),
        7 => (
            base | CapabilityFlags::BF16 | CapabilityFlags::INT8,
            ComputeTier::Consumer,
        ),
        _ => (base, ComputeTier::Mobile),
    }
}
