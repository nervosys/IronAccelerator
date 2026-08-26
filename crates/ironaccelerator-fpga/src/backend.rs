//! FPGA `Backend` impl over the Xilinx Runtime. One descriptor per XRT device.

use ironaccelerator_core::{
    Backend, BackendKind, Capability, CapabilityFlags, ComputeTier, DeviceDescriptor, DeviceId,
    Result, Vendor,
};

pub struct FpgaBackend;
pub static FPGA_BACKEND: FpgaBackend = FpgaBackend;

impl Backend for FpgaBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Fpga
    }

    fn is_available(&self) -> bool {
        crate::drv::is_available() && crate::drv::device_count() > 0
    }

    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>> {
        let n = crate::drv::device_count();
        Ok((0..n)
            .map(|ord| DeviceDescriptor {
                id: DeviceId {
                    backend: BackendKind::Fpga,
                    ordinal: ord,
                },
                // XRT is the AMD/Xilinx stack (Alveo, Versal). Silicon vendor,
                // not API — the Fpga BackendKind is what marks the API family.
                vendor: Vendor::Amd,
                name: format!("AMD/Xilinx FPGA (XRT device {ord})"),
                arch: "xrt".to_string(),
                // XRT does report board memory via xclGetDeviceInfo2, but that
                // needs a versioned struct we don't bind at this scaffold level;
                // 0 is honest ("unknown") rather than a guessed figure.
                total_memory_bytes: 0,
                multiprocessor_count: 0,
                clock_khz: 0,
                capability: Capability {
                    flags: fpga_flags(),
                    // Alveo/Versal are datacenter accelerator cards.
                    tier: ComputeTier::Datacenter,
                    fp16_tflops: None,
                    fp8_tflops: None,
                    mem_bandwidth_gbs: None,
                },
            })
            .collect())
    }

    fn capabilities(&self, device: u32) -> Result<CapabilityFlags> {
        // Validate the ordinal so this agrees with `enumerate` on what exists.
        if device >= crate::drv::device_count() {
            return Err(ironaccelerator_core::Error::InvalidArgument(
                "fpga device ordinal out of range",
            ));
        }
        Ok(fpga_flags())
    }
}

/// An FPGA's numeric capability is defined by the *loaded bitstream*, not the
/// silicon — the same card runs an INT8 CNN accelerator or an FP32 solver
/// depending on what was flashed. Reporting fixed numeric flags here would be
/// fiction, so the fabric advertises an empty set; consumers learn the real
/// capability from the `.xclbin` they load, above the driver line.
fn fpga_flags() -> CapabilityFlags {
    CapabilityFlags::empty()
}
