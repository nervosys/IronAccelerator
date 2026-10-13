//! Vendor-neutral device descriptors.

use crate::{backend::BackendKind, capability::Capability};

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(feature = "std")]
use std::string::String;

/// Stable identifier for a device: owning backend plus backend-local ordinal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeviceId {
    /// Backend that owns the device.
    pub backend: BackendKind,
    /// Backend-local ordinal (driver enumeration index).
    pub ordinal: u32,
}

/// Hardware vendor of a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Vendor {
    /// NVIDIA.
    Nvidia,
    /// AMD.
    Amd,
    /// Apple.
    Apple,
    /// Qualcomm.
    Qualcomm,
    /// Intel.
    Intel,
    /// Google (TPU).
    Google,
    /// AWS (Trainium / Inferentia).
    Aws,
    /// Any vendor not listed above.
    Other,
}

/// Static description of a device, as reported by enumeration.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeviceDescriptor {
    /// Identifier of the device.
    pub id: DeviceId,
    /// Hardware vendor.
    pub vendor: Vendor,
    /// Marketing name (e.g. "NVIDIA H100 80GB HBM3").
    pub name: String,
    /// Vendor architecture string (e.g. "sm_90a", "gfx942", "Apple9", "v75").
    pub arch: String,
    /// Total device memory in bytes.
    pub total_memory_bytes: u64,
    /// Number of multiprocessors (SMs, CUs, or equivalent).
    pub multiprocessor_count: u32,
    /// Core clock in kHz.
    pub clock_khz: u32,
    /// Capability description of the device.
    pub capability: Capability,
}

/// Live device handle. Concrete backends extend this trait with launch APIs.
pub trait Device: Send + Sync {
    /// Returns the device's descriptor.
    fn descriptor(&self) -> &DeviceDescriptor;

    /// Returns the device's identifier.
    fn id(&self) -> DeviceId {
        self.descriptor().id
    }
}
