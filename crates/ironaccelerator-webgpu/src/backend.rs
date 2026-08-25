//! WebGPU `Backend` impl over the host-bound adapter.

use ironaccelerator_core::{
    Backend, BackendKind, Capability, CapabilityFlags, ComputeTier, DeviceDescriptor, DeviceId,
    Error, Result, Vendor,
};

use crate::drv::AdapterInfo;

pub struct WebGpuBackend;
pub static WEBGPU_BACKEND: WebGpuBackend = WebGpuBackend;

impl Backend for WebGpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::WebGpu
    }

    fn is_available(&self) -> bool {
        crate::drv::is_available()
    }

    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>> {
        Ok(crate::drv::enumerate()
            .into_iter()
            .enumerate()
            .map(|(i, info)| describe(i as u32, info))
            .collect())
    }

    fn capabilities(&self, device: u32) -> Result<CapabilityFlags> {
        if device != 0 {
            return Err(Error::InvalidArgument(
                "webgpu exposes a single bound adapter at ordinal 0",
            ));
        }
        crate::drv::bound_adapter()
            .map(|a| flags_for(&a))
            .ok_or(Error::BackendUnavailable("webgpu: no adapter bound"))
    }
}

/// Translate the granted WebGPU feature set into the common flag space.
///
/// WebGPU's compute surface is small by design, so this stays minimal: FP32 is
/// guaranteed, FP16 only with `"shader-f16"`, and nothing above that is
/// expressible. There is no INT8 dot product, no matrix engine, and no
/// query for either.
fn flags_for(a: &AdapterInfo) -> CapabilityFlags {
    let mut flags = CapabilityFlags::FP32;
    if a.shader_f16 {
        flags |= CapabilityFlags::FP16;
    }
    flags
}

fn describe(ordinal: u32, a: AdapterInfo) -> DeviceDescriptor {
    // WebGPU reports a lowercase vendor string, not a PCI ID.
    let vendor = match a.vendor.as_str() {
        "nvidia" => Vendor::Nvidia,
        "amd" => Vendor::Amd,
        "intel" => Vendor::Intel,
        "apple" => Vendor::Apple,
        "qualcomm" => Vendor::Qualcomm,
        _ => Vendor::Other,
    };

    // The browser tells us nothing about discrete-vs-integrated or silicon
    // class, and guessing from the vendor string would be fiction.
    let tier = ComputeTier::Baseline;

    let name = if !a.description.is_empty() {
        a.description.clone()
    } else if !a.device.is_empty() {
        a.device.clone()
    } else if !a.vendor.is_empty() {
        a.vendor.clone()
    } else {
        "webgpu adapter".to_string()
    };

    let arch = if a.architecture.is_empty() {
        "webgpu".to_string()
    } else {
        format!("webgpu-{}", a.architecture)
    };

    DeviceDescriptor {
        id: DeviceId {
            backend: BackendKind::WebGpu,
            ordinal,
        },
        vendor,
        name,
        arch,
        // Not a memory size: WebGPU never reports one. This is the largest
        // single allocation the adapter will honour, which is the only
        // capacity figure available.
        total_memory_bytes: a.max_buffer_size,
        multiprocessor_count: 0,
        clock_khz: 0,
        capability: Capability {
            flags: flags_for(&a),
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
    use crate::drv::{bind_adapter, unbind_adapter, TEST_BINDING_LOCK};

    #[test]
    fn unbound_backend_reports_unavailable() {
        let _g = TEST_BINDING_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unbind_adapter();
        assert!(!WEBGPU_BACKEND.is_available());
        assert!(WEBGPU_BACKEND.enumerate().unwrap().is_empty());
        assert!(WEBGPU_BACKEND.capabilities(0).is_err());
    }

    #[test]
    fn bound_adapter_is_described_and_capable() {
        let _g = TEST_BINDING_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        bind_adapter(AdapterInfo {
            vendor: "amd".into(),
            architecture: "rdna-3".into(),
            shader_f16: true,
            max_buffer_size: 4096,
            ..Default::default()
        });

        let devices = WEBGPU_BACKEND.enumerate().unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].vendor, Vendor::Amd);
        assert_eq!(devices[0].arch, "webgpu-rdna-3");
        assert_eq!(devices[0].name, "amd");
        assert_eq!(devices[0].total_memory_bytes, 4096);

        let flags = WEBGPU_BACKEND.capabilities(0).unwrap();
        assert_eq!(flags, CapabilityFlags::FP32 | CapabilityFlags::FP16);
        assert_eq!(flags, devices[0].capability.flags);

        assert!(
            WEBGPU_BACKEND.capabilities(1).is_err(),
            "only ordinal 0 exists"
        );
        unbind_adapter();
    }

    #[test]
    fn without_shader_f16_only_fp32() {
        let _g = TEST_BINDING_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        bind_adapter(AdapterInfo {
            vendor: "intel".into(),
            ..Default::default()
        });
        assert_eq!(
            WEBGPU_BACKEND.capabilities(0).unwrap(),
            CapabilityFlags::FP32
        );
        unbind_adapter();
    }
}
