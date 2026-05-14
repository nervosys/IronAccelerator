//! OpenGL `Backend` impl. Reports unavailable until a GL 4.3+ compute-capable
//! context has been bound via [`crate::bind_current_context`].

use ironaccelerator_core::{
    Backend, BackendKind, Capability, CapabilityFlags, ComputeTier, DeviceDescriptor, DeviceId,
    Result, Strategy, Vendor, Workload,
};

pub struct OpenGlBackend;
pub static OPENGL_BACKEND: OpenGlBackend = OpenGlBackend;

impl Backend for OpenGlBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::OpenGl
    }

    fn is_available(&self) -> bool {
        crate::drv::info()
            .map(|i| i.supports_compute)
            .unwrap_or(false)
    }

    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>> {
        let Some(info) = crate::drv::info() else {
            return Ok(Vec::new());
        };
        if !info.supports_compute {
            return Ok(Vec::new());
        }
        let vendor = detect_vendor(&info.vendor);
        let tier = match vendor {
            Vendor::Nvidia | Vendor::Amd => ComputeTier::Consumer,
            Vendor::Intel | Vendor::Apple => ComputeTier::Mobile,
            _ => ComputeTier::Baseline,
        };
        Ok(vec![DeviceDescriptor {
            id: DeviceId {
                backend: BackendKind::OpenGl,
                ordinal: 0,
            },
            vendor,
            name: info.renderer,
            arch: format!("gl{}.{}", info.major, info.minor),
            total_memory_bytes: 0,
            multiprocessor_count: 0,
            clock_khz: 0,
            capability: Capability {
                flags: CapabilityFlags::FP32 | CapabilityFlags::FP16,
                tier,
                fp16_tflops: None,
                fp8_tflops: None,
                mem_bandwidth_gbs: None,
            },
        }])
    }

    fn capabilities(&self, _device: u32) -> Result<CapabilityFlags> {
        Ok(CapabilityFlags::FP32 | CapabilityFlags::FP16)
    }

    fn plan(&self, _device: u32, _w: &Workload) -> Result<Strategy> {
        Ok(Strategy::GlslCompute {
            workgroup: (64, 1, 1),
        })
    }
}

fn detect_vendor(s: &str) -> Vendor {
    let s = s.to_ascii_lowercase();
    if s.contains("nvidia") {
        Vendor::Nvidia
    } else if s.contains("amd") || s.contains("ati") || s.contains("radeon") {
        Vendor::Amd
    } else if s.contains("intel") {
        Vendor::Intel
    } else if s.contains("apple") {
        Vendor::Apple
    } else if s.contains("qualcomm") || s.contains("adreno") {
        Vendor::Qualcomm
    } else {
        Vendor::Other
    }
}
