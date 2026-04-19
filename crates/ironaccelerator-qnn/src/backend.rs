//! QNN `Backend` impl. Enumerate-by-probing: try HTP / GPU / CPU / DSP and
//! emit a `DeviceDescriptor` for each target whose library loads.

use crate::drv;
use iron_qnn_sys::qnn::Target;
use ironaccelerator_core::{
    Backend, BackendKind, Capability, CapabilityFlags, ComputeTier, DType, DeviceDescriptor,
    DeviceId, Result, Strategy, Vendor, Workload, WorkloadKind,
};
use once_cell::sync::Lazy;

pub struct QnnBackend {
    /// Bitmask of available targets at discovery time. Cached; library state
    /// doesn't change across a single process invocation.
    available: u8,
}

pub static QNN_BACKEND: Lazy<QnnBackend> = Lazy::new(|| QnnBackend {
    available: probe_all(),
});

fn probe_all() -> u8 {
    let mut m = 0u8;
    if drv::is_available(Target::Htp) { m |= 1 << 0; }
    if drv::is_available(Target::Gpu) { m |= 1 << 1; }
    if drv::is_available(Target::Cpu) { m |= 1 << 2; }
    if drv::is_available(Target::Dsp) { m |= 1 << 3; }
    m
}

fn target_of(ordinal: u32) -> Option<Target> {
    match ordinal {
        0 => Some(Target::Htp),
        1 => Some(Target::Gpu),
        2 => Some(Target::Cpu),
        3 => Some(Target::Dsp),
        _ => None,
    }
}

impl QnnBackend {
    /// Which targets were loadable at discovery time.
    pub fn available_targets(&self) -> Vec<Target> {
        let mut v = Vec::new();
        if self.available & 1 != 0 { v.push(Target::Htp); }
        if self.available & 2 != 0 { v.push(Target::Gpu); }
        if self.available & 4 != 0 { v.push(Target::Cpu); }
        if self.available & 8 != 0 { v.push(Target::Dsp); }
        v
    }
}

impl Backend for QnnBackend {
    fn kind(&self) -> BackendKind { BackendKind::QualcommNpu }
    fn is_available(&self) -> bool { self.available != 0 }

    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>> {
        let mut out = Vec::new();
        for (ord, target) in self.available_targets().into_iter().enumerate() {
            let ord = ord as u32;
            let name = format!("QNN {:?}", target);
            let arch = match target {
                Target::Htp => "hexagon-v75",
                Target::Gpu => "adreno",
                Target::Cpu => "qnn-cpu",
                Target::Dsp => "hexagon-dsp",
                Target::Saver => "saver",
            }.to_string();
            out.push(DeviceDescriptor {
                id: DeviceId { backend: BackendKind::QualcommNpu, ordinal: ord },
                vendor: Vendor::Qualcomm,
                name, arch,
                total_memory_bytes: 0,
                multiprocessor_count: 0,
                clock_khz: 0,
                capability: capability_for(target),
            });
        }
        Ok(out)
    }

    fn capabilities(&self, device: u32) -> Result<CapabilityFlags> {
        let t = target_of(device).ok_or(ironaccelerator_core::Error::InvalidArgument("unknown QNN ordinal"))?;
        Ok(capability_for(t).flags)
    }

    fn plan(&self, _device: u32, w: &Workload) -> Result<Strategy> {
        Ok(match (w.kind, w.input_dtype) {
            (WorkloadKind::Gemm | WorkloadKind::Conv2d | WorkloadKind::Attention, DType::I8) =>
                Strategy::QnnHtpGraph { precision: DType::I8 },
            (WorkloadKind::Gemm | WorkloadKind::Conv2d | WorkloadKind::Attention, _) =>
                Strategy::QnnHtpGraph { precision: DType::F16 },
            _ => Strategy::QnnHtpGraph { precision: DType::F16 },
        })
    }
}

fn capability_for(target: Target) -> Capability {
    let (flags, tier) = match target {
        Target::Htp => (
            CapabilityFlags::FP16 | CapabilityFlags::INT8 | CapabilityFlags::INT4
            | CapabilityFlags::HMX | CapabilityFlags::HVX | CapabilityFlags::UNIFIED_MEMORY,
            ComputeTier::Mobile),
        Target::Gpu => (
            CapabilityFlags::FP32 | CapabilityFlags::FP16 | CapabilityFlags::UNIFIED_MEMORY,
            ComputeTier::Mobile),
        Target::Cpu => (
            CapabilityFlags::FP32 | CapabilityFlags::FP16 | CapabilityFlags::INT8,
            ComputeTier::Baseline),
        Target::Dsp => (
            CapabilityFlags::FP16 | CapabilityFlags::INT8 | CapabilityFlags::HVX,
            ComputeTier::Mobile),
        Target::Saver => (CapabilityFlags::FP32, ComputeTier::Baseline),
    };
    Capability { flags, tier, fp16_tflops: None, fp8_tflops: None, mem_bandwidth_gbs: None }
}
