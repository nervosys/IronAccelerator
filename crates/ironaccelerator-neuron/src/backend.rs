//! AWS Neuron `Backend` impl. One descriptor per NeuronCore.

use ironaccelerator_core::{
    Backend, BackendKind, Capability, CapabilityFlags, ComputeTier, DeviceDescriptor, DeviceId,
    Result, Strategy, Vendor, Workload,
};

use crate::drv::NeuronGen;

pub struct NeuronBackend;
pub static NEURON_BACKEND: NeuronBackend = NeuronBackend;

impl Backend for NeuronBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Neuron
    }

    fn is_available(&self) -> bool {
        crate::drv::is_available() && crate::drv::total_cores() > 0
    }

    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>> {
        let cores = crate::drv::total_cores();
        if cores == 0 {
            return Ok(Vec::new());
        }
        let gen = crate::drv::detect_generation();
        let flags = capability_flags(gen);
        let arch = arch_string(gen);
        let hbm = hbm_bytes_per_core(gen);
        Ok((0..cores)
            .map(|ord| DeviceDescriptor {
                id: DeviceId {
                    backend: BackendKind::Neuron,
                    ordinal: ord,
                },
                vendor: Vendor::Aws,
                name: format!("AWS NeuronCore ({})", arch),
                arch: arch.to_string(),
                total_memory_bytes: hbm,
                multiprocessor_count: 0,
                clock_khz: 0,
                capability: Capability {
                    flags,
                    tier: ComputeTier::Datacenter,
                    fp16_tflops: None,
                    fp8_tflops: None,
                    mem_bandwidth_gbs: None,
                },
            })
            .collect())
    }

    fn capabilities(&self, _device: u32) -> Result<CapabilityFlags> {
        Ok(capability_flags(crate::drv::detect_generation()))
    }

    fn plan(&self, _device: u32, _w: &Workload) -> Result<Strategy> {
        Ok(Strategy::Neuron {
            num_cores: crate::drv::total_cores(),
        })
    }
}

fn capability_flags(gen: NeuronGen) -> CapabilityFlags {
    let base = CapabilityFlags::FP32
        | CapabilityFlags::BF16
        | CapabilityFlags::INT8
        | CapabilityFlags::TENSOR_CORES
        | CapabilityFlags::HBM
        | CapabilityFlags::MULTI_STREAM;
    match gen {
        NeuronGen::Trn2 => {
            base | CapabilityFlags::FP8_E4M3 | CapabilityFlags::FP8_E5M2 | CapabilityFlags::INT4
        }
        NeuronGen::Trn1 => base | CapabilityFlags::FP8_E4M3 | CapabilityFlags::FP8_E5M2,
        NeuronGen::Inf1 => {
            CapabilityFlags::FP32
                | CapabilityFlags::BF16
                | CapabilityFlags::INT8
                | CapabilityFlags::TENSOR_CORES
                | CapabilityFlags::MULTI_STREAM
        }
        NeuronGen::Unknown => base,
    }
}

fn arch_string(gen: NeuronGen) -> &'static str {
    match gen {
        NeuronGen::Inf1 => "neuron-v1-inferentia",
        NeuronGen::Trn1 => "neuron-v2-trainium",
        NeuronGen::Trn2 => "neuron-v3-trainium2",
        NeuronGen::Unknown => "neuron",
    }
}

fn hbm_bytes_per_core(gen: NeuronGen) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;
    match gen {
        NeuronGen::Inf1 => 8 * GIB,
        NeuronGen::Trn1 => 16 * GIB,
        NeuronGen::Trn2 => 24 * GIB,
        NeuronGen::Unknown => 0,
    }
}
