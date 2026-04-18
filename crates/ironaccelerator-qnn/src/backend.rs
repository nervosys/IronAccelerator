use ironaccelerator_core::{
    Backend, BackendKind, CapabilityFlags, DType, DeviceDescriptor, Result, Strategy, Workload,
    WorkloadKind,
};
use once_cell::sync::Lazy;

pub struct QnnBackend { available: bool }

pub static QNN_BACKEND: Lazy<QnnBackend> = Lazy::new(|| QnnBackend { available: probe() });

fn probe() -> bool {
    use libloading::Library;
    let candidates: &[&str] = if cfg!(target_os = "linux") {
        &["libQnnHtp.so", "libQnnSystem.so"]
    } else if cfg!(target_os = "windows") {
        &["QnnHtp.dll", "QnnSystem.dll"]
    } else if cfg!(target_os = "android") {
        &["libQnnHtp.so"]
    } else { &[] };
    candidates.iter().any(|n| unsafe { Library::new(*n) }.is_ok())
}

impl Backend for QnnBackend {
    fn kind(&self) -> BackendKind { BackendKind::QualcommNpu }
    fn is_available(&self) -> bool { self.available }
    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>> { Ok(Vec::new()) }

    fn capabilities(&self, _device: u32) -> Result<CapabilityFlags> {
        Ok(CapabilityFlags::FP16 | CapabilityFlags::INT8 | CapabilityFlags::INT4
            | CapabilityFlags::HMX | CapabilityFlags::HVX)
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
