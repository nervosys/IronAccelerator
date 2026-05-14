//! ROCm `Backend` impl. Live device enumeration through HIP.

use crate::drv::{Device, Result as DrvResult};
use iron_rocm_sys::hip::HipDeviceAttribute as Attr;
use ironaccelerator_core::{
    strategy::FlashVariant, Backend, BackendKind, Capability, CapabilityFlags, ComputeTier,
    DeviceDescriptor, DeviceId, Result, Strategy, Vendor, Workload, WorkloadKind,
};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub struct RocmBackend {
    devices: RwLock<HashMap<u32, Arc<Device>>>,
}

pub static ROCM_BACKEND: Lazy<RocmBackend> = Lazy::new(|| RocmBackend {
    devices: RwLock::new(HashMap::new()),
});

impl RocmBackend {
    pub fn device(&self, ordinal: u32) -> DrvResult<Arc<Device>> {
        if let Some(d) = self.devices.read().get(&ordinal) {
            return Ok(d.clone());
        }
        let d = Device::open(ordinal)?;
        self.devices.write().insert(ordinal, d.clone());
        Ok(d)
    }

    pub fn capability(&self, ordinal: u32) -> Result<Capability> {
        let d = self.device(ordinal)?;
        let (maj, min) = d.compute_capability()?;
        let total = d.total_mem()? as u64;
        Ok(capability_from_arch(maj, min, total))
    }
}

impl Backend for RocmBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Rocm
    }

    fn is_available(&self) -> bool {
        Device::count().unwrap_or(0) > 0
    }

    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>> {
        let n = Device::count()?;
        let mut out = Vec::with_capacity(n as usize);
        for ord in 0..n {
            let d = match self.device(ord) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let name = d.name().unwrap_or_else(|_| "unknown".into());
            let (maj, min) = d.compute_capability().unwrap_or((0, 0));
            let arch = format!("gfx{maj}{min:02}");
            let total = d.total_mem().unwrap_or(0) as u64;
            let mp = d.attribute(Attr::MultiprocessorCount).unwrap_or(0) as u32;
            let clock = d.attribute(Attr::ClockRate).unwrap_or(0) as u32;
            let cap = self.capability(ord).unwrap_or_else(|_| empty_cap());

            out.push(DeviceDescriptor {
                id: DeviceId {
                    backend: BackendKind::Rocm,
                    ordinal: ord,
                },
                vendor: Vendor::Amd,
                name,
                arch,
                total_memory_bytes: total,
                multiprocessor_count: mp,
                clock_khz: clock,
                capability: cap,
            });
        }
        Ok(out)
    }

    fn capabilities(&self, device: u32) -> Result<CapabilityFlags> {
        Ok(self.capability(device)?.flags)
    }

    fn plan(&self, _device: u32, w: &Workload) -> Result<Strategy> {
        Ok(match w.kind {
            WorkloadKind::Gemm | WorkloadKind::BatchedGemm => Strategy::BlasLt {
                epilogue: "bias-gelu",
            },
            WorkloadKind::FlashAttention | WorkloadKind::Attention => Strategy::FusedAttention {
                variant: FlashVariant::V2,
            },
            _ => Strategy::CutlassTemplate {
                tile: (256, 128, 32),
                stages: 2,
            },
        })
    }
}

// ─── capability mapping ────────────────────────────────────────────────────

fn capability_from_arch(maj: u32, min: u32, total: u64) -> Capability {
    // CDNA family (MI100=gfx908, MI200=gfx90a, MI300=gfx942).
    // RDNA3 (gfx1100+) covers consumer Radeon 7000-series with WMMA.
    let code = maj * 100 + min;
    let (tier, mut flags) = match code {
        // MI300X / MI300A — full FP8 + matrix cores
        942 => (
            ComputeTier::Datacenter,
            CapabilityFlags::FP64
                | CapabilityFlags::FP32
                | CapabilityFlags::FP16
                | CapabilityFlags::BF16
                | CapabilityFlags::FP8_E4M3
                | CapabilityFlags::FP8_E5M2
                | CapabilityFlags::INT8
                | CapabilityFlags::TENSOR_CORES
                | CapabilityFlags::WMMA
                | CapabilityFlags::INFINITY_FABRIC
                | CapabilityFlags::RCCL,
        ),
        // MI250 / MI210
        910 => (
            ComputeTier::Datacenter,
            CapabilityFlags::FP64
                | CapabilityFlags::FP32
                | CapabilityFlags::FP16
                | CapabilityFlags::BF16
                | CapabilityFlags::INT8
                | CapabilityFlags::TENSOR_CORES
                | CapabilityFlags::INFINITY_FABRIC
                | CapabilityFlags::RCCL,
        ),
        // MI100
        908 => (
            ComputeTier::Datacenter,
            CapabilityFlags::FP64
                | CapabilityFlags::FP32
                | CapabilityFlags::FP16
                | CapabilityFlags::BF16
                | CapabilityFlags::TENSOR_CORES,
        ),
        // RDNA3 (Radeon 7000) — WMMA, no matrix cores
        1100..=1199 => (
            ComputeTier::Consumer,
            CapabilityFlags::FP32
                | CapabilityFlags::FP16
                | CapabilityFlags::BF16
                | CapabilityFlags::WMMA,
        ),
        // RDNA2 (Radeon 6000)
        1000..=1099 => (
            ComputeTier::Consumer,
            CapabilityFlags::FP32 | CapabilityFlags::FP16,
        ),
        _ => (
            ComputeTier::Consumer,
            CapabilityFlags::FP32 | CapabilityFlags::FP16,
        ),
    };
    // Large HBM → datacenter class regardless of arch decoding gaps.
    if total >= 40 * (1u64 << 30) {
        flags |= CapabilityFlags::INFINITY_FABRIC | CapabilityFlags::RCCL;
    }
    Capability {
        flags,
        tier,
        fp16_tflops: None,
        fp8_tflops: None,
        mem_bandwidth_gbs: None,
    }
}

fn empty_cap() -> Capability {
    Capability {
        flags: CapabilityFlags::FP32 | CapabilityFlags::FP16,
        tier: ComputeTier::Consumer,
        fp16_tflops: None,
        fp8_tflops: None,
        mem_bandwidth_gbs: None,
    }
}
