//! Vendor-neutral device descriptors.

use crate::{backend::BackendKind, capability::Capability};

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(feature = "std")]
use std::string::String;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeviceId {
    pub backend: BackendKind,
    /// Backend-local ordinal (driver enumeration index).
    pub ordinal: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Vendor {
    Nvidia,
    Amd,
    Apple,
    Qualcomm,
    Intel,
    Google,
    Aws,
    Other,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeviceDescriptor {
    pub id: DeviceId,
    pub vendor: Vendor,
    /// Marketing name (e.g. "NVIDIA H100 80GB HBM3").
    pub name: String,
    /// Vendor architecture string (e.g. "sm_90a", "gfx942", "Apple9", "v75").
    pub arch: String,
    pub total_memory_bytes: u64,
    pub multiprocessor_count: u32,
    pub clock_khz: u32,
    pub capability: Capability,
}

/// Live device handle. Concrete backends extend this trait with launch APIs.
pub trait Device: Send + Sync {
    fn descriptor(&self) -> &DeviceDescriptor;

    fn id(&self) -> DeviceId {
        self.descriptor().id
    }
}
