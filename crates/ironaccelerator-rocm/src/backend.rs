//! ROCm `Backend` impl. Currently a probe-only scaffold; once the HIP FFI
//! bindings land, `enumerate` will walk `hipGetDeviceCount` / `hipGetDeviceProperties`.

use ironaccelerator_core::{
    Backend, BackendKind, CapabilityFlags, DeviceDescriptor, Result, Strategy, Workload,
    WorkloadKind, strategy::FlashVariant,
};
use once_cell::sync::Lazy;

pub struct RocmBackend {
    available: bool,
}

pub static ROCM_BACKEND: Lazy<RocmBackend> = Lazy::new(|| RocmBackend {
    available: probe_runtime(),
});

fn probe_runtime() -> bool {
    use libloading::Library;
    let candidates: &[&str] = if cfg!(target_os = "linux") {
        &["libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"]
    } else if cfg!(target_os = "windows") {
        &["amdhip64.dll", "amdhip64_6.dll"]
    } else {
        &[]
    };
    candidates.iter().any(|n| unsafe { Library::new(*n) }.is_ok())
}

impl Backend for RocmBackend {
    fn kind(&self) -> BackendKind { BackendKind::Rocm }
    fn is_available(&self) -> bool { self.available }

    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>> { Ok(Vec::new()) }

    fn capabilities(&self, _device: u32) -> Result<CapabilityFlags> {
        Ok(CapabilityFlags::FP32 | CapabilityFlags::FP16 | CapabilityFlags::BF16
            | CapabilityFlags::TENSOR_CORES | CapabilityFlags::WMMA
            | CapabilityFlags::INFINITY_FABRIC | CapabilityFlags::RCCL)
    }

    fn plan(&self, _device: u32, w: &Workload) -> Result<Strategy> {
        Ok(match w.kind {
            WorkloadKind::Gemm | WorkloadKind::BatchedGemm =>
                Strategy::BlasLt { epilogue: "bias-gelu" },
            WorkloadKind::FlashAttention | WorkloadKind::Attention =>
                Strategy::FusedAttention { variant: FlashVariant::V2 },
            _ => Strategy::CutlassTemplate { tile: (256, 128, 32), stages: 2 },
        })
    }
}
