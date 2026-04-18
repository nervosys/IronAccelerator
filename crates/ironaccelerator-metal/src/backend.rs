use ironaccelerator_core::{
    Backend, BackendKind, CapabilityFlags, DeviceDescriptor, Result, Strategy, Workload,
    WorkloadKind,
};

pub struct MetalBackend;
pub static METAL_BACKEND: MetalBackend = MetalBackend;

impl Backend for MetalBackend {
    fn kind(&self) -> BackendKind { BackendKind::Metal }

    fn is_available(&self) -> bool { cfg!(target_vendor = "apple") }

    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>> { Ok(Vec::new()) }

    fn capabilities(&self, _device: u32) -> Result<CapabilityFlags> {
        Ok(CapabilityFlags::FP32 | CapabilityFlags::FP16 | CapabilityFlags::BF16
            | CapabilityFlags::UNIFIED_MEMORY | CapabilityFlags::ANE
            | CapabilityFlags::MULTI_STREAM)
    }

    fn plan(&self, _device: u32, w: &Workload) -> Result<Strategy> {
        Ok(match w.kind {
            WorkloadKind::Gemm | WorkloadKind::BatchedGemm => Strategy::MpsGraph,
            WorkloadKind::FlashAttention | WorkloadKind::Attention => Strategy::MpsGraph,
            _ => Strategy::MpsGraph,
        })
    }
}
