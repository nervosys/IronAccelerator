//! TPU `Backend` impl. One `DeviceDescriptor` per TPU chip visible to the
//! host, driven by `TPU_ACCELERATOR_TYPE` + `TPU_NUM_DEVICES`.

use ironaccelerator_core::{
    Backend, BackendKind, Capability, CapabilityFlags, ComputeTier, DeviceDescriptor, DeviceId,
    Result, Vendor,
};

pub struct TpuBackend;
pub static TPU_BACKEND: TpuBackend = TpuBackend;

impl Backend for TpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Tpu
    }

    fn is_available(&self) -> bool {
        crate::drv::is_plugin_available() && crate::drv::topology().is_some()
    }

    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>> {
        let Some(topo) = crate::drv::topology() else {
            return Ok(Vec::new());
        };
        let generation = detect_generation(&topo.accelerator_type);
        let flags = capability_flags(generation);
        let tier = ComputeTier::Datacenter;
        Ok((0..topo.chips_per_host.min(topo.num_devices))
            .map(|ord| DeviceDescriptor {
                id: DeviceId {
                    backend: BackendKind::Tpu,
                    ordinal: ord,
                },
                vendor: Vendor::Google,
                name: format!("Google TPU {}", topo.accelerator_type),
                arch: generation_arch(generation).to_string(),
                total_memory_bytes: hbm_bytes_per_chip(generation),
                multiprocessor_count: 0,
                clock_khz: 0,
                capability: Capability {
                    flags,
                    tier,
                    fp16_tflops: None,
                    fp8_tflops: None,
                    mem_bandwidth_gbs: None,
                },
            })
            .collect())
    }

    fn capabilities(&self, device: u32) -> Result<CapabilityFlags> {
        // All chips in a TPU slice are the same generation; validate the
        // ordinal so this agrees with `enumerate` on what exists.
        if !self.enumerate()?.iter().any(|d| d.id.ordinal == device) {
            return Err(ironaccelerator_core::Error::InvalidArgument(
                "tpu chip ordinal out of range",
            ));
        }
        let gen = crate::drv::topology()
            .map(|t| detect_generation(&t.accelerator_type))
            .unwrap_or(TpuGen::Unknown);
        Ok(capability_flags(gen))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TpuGen {
    V4,
    V5,
    V5e,
    V5p,
    V6e,
    Unknown,
}

fn detect_generation(accel: &str) -> TpuGen {
    let s = accel.to_ascii_lowercase();
    if s.starts_with("v6e") {
        TpuGen::V6e
    } else if s.starts_with("v5p") {
        TpuGen::V5p
    } else if s.starts_with("v5litepod") || s.starts_with("v5e") {
        TpuGen::V5e
    } else if s.starts_with("v5") {
        TpuGen::V5
    } else if s.starts_with("v4") {
        TpuGen::V4
    } else {
        TpuGen::Unknown
    }
}

fn generation_arch(gen: TpuGen) -> &'static str {
    match gen {
        TpuGen::V4 => "tpu-v4",
        TpuGen::V5 => "tpu-v5",
        TpuGen::V5e => "tpu-v5e",
        TpuGen::V5p => "tpu-v5p",
        TpuGen::V6e => "tpu-v6e-trillium",
        TpuGen::Unknown => "tpu",
    }
}

fn capability_flags(gen: TpuGen) -> CapabilityFlags {
    let base = CapabilityFlags::FP32
        | CapabilityFlags::BF16
        | CapabilityFlags::INT8
        | CapabilityFlags::TENSOR_CORES
        | CapabilityFlags::HBM
        | CapabilityFlags::MULTI_STREAM;
    match gen {
        TpuGen::V6e | TpuGen::V5p | TpuGen::V5 => base | CapabilityFlags::INT4,
        _ => base,
    }
}

fn hbm_bytes_per_chip(gen: TpuGen) -> u64 {
    // Vendor-published numbers; used as a sizing hint only.
    const GIB: u64 = 1024 * 1024 * 1024;
    match gen {
        TpuGen::V4 => 32 * GIB,
        TpuGen::V5 | TpuGen::V5e => 16 * GIB,
        TpuGen::V5p => 95 * GIB,
        TpuGen::V6e => 32 * GIB,
        TpuGen::Unknown => 0,
    }
}
