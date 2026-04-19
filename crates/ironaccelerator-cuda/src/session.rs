//! `Session` — the canonical entry point for executing IronAccelerator
//! workloads on CUDA.
//!
//! Built on the in-crate [`crate::drv`] safe layer (which sits directly on
//! `iron_cuda_sys`). A `Session` bundles:
//!
//! - a retained primary [`crate::drv::Device`] for an ordinal
//! - the active [`crate::drv::Stream`] (`Arc`-shared so it can be cloned cheaply
//!   into tensor lifetimes when needed)
//! - the device's `Capability` profile
//! - per-session [`Metrics`] and [`Profiler`]
//!
//! Library handles (cuBLASLt, cuDNN, cuFFT, …) are **not** held by the
//! `Session` — they live in their own per-domain cache modules so sessions
//! stay cheap to create.
//!
//! ```no_run
//! use ironaccelerator_cuda::Session;
//! use ironaccelerator_core::{DType, Workload};
//!
//! let s = Session::new(0)?;
//! let plan = s.plan(&Workload::gemm(4096, 4096, 4096, DType::Bf16))?;
//! println!("planner picked {plan:?}");
//! # Ok::<(), ironaccelerator_core::Error>(())
//! ```

use crate::backend::{capability_from_arch, plan_strategy};
use crate::drv::{Device, Priority, Stream};
use crate::observe::Metrics;
use crate::profile::Profiler;
use iron_cuda_sys::driver::CUdevice_attribute as Attr;
use ironaccelerator_core::{Capability, CapabilityFlags, ComputeTier, Result, Strategy, Workload};
use std::sync::Arc;

pub struct Session {
    ordinal: u32,
    device: Arc<Device>,
    stream: Arc<Stream>,
    capability: Capability,
    metrics: Arc<Metrics>,
    profiler: Arc<Profiler>,
}

impl Session {
    /// Open (or attach to) the primary context for `ordinal` and create a
    /// dedicated default-priority stream.
    pub fn new(ordinal: u32) -> Result<Self> {
        let device = Device::open(ordinal)?;
        let stream = Stream::new(device.clone())?;
        let capability = derive_capability(&device)?;
        Ok(Self {
            ordinal, device, stream, capability,
            metrics: Arc::new(Metrics::default()),
            profiler: Arc::new(Profiler::new()),
        })
    }

    /// Open with a specific stream priority.
    pub fn with_priority(ordinal: u32, priority: Priority) -> Result<Self> {
        let device = Device::open(ordinal)?;
        let stream = Stream::with_priority(device.clone(), priority)?;
        let capability = derive_capability(&device)?;
        Ok(Self {
            ordinal, device, stream, capability,
            metrics: Arc::new(Metrics::default()),
            profiler: Arc::new(Profiler::new()),
        })
    }

    /// Open with a caller-supplied stream (e.g. shared between sessions for
    /// fine-grained pipelining). The stream must be on the same device.
    pub fn with_stream(ordinal: u32, stream: Arc<Stream>) -> Result<Self> {
        let device = Device::open(ordinal)?;
        let capability = derive_capability(&device)?;
        Ok(Self {
            ordinal, device, stream, capability,
            metrics: Arc::new(Metrics::default()),
            profiler: Arc::new(Profiler::new()),
        })
    }

    #[inline] pub fn ordinal(&self) -> u32 { self.ordinal }
    #[inline] pub fn device(&self) -> &Arc<Device> { &self.device }
    #[inline] pub fn stream(&self) -> &Arc<Stream> { &self.stream }
    #[inline] pub fn capability(&self) -> &Capability { &self.capability }
    #[inline] pub fn metrics(&self) -> &Arc<Metrics> { &self.metrics }
    #[inline] pub fn profiler(&self) -> &Arc<Profiler> { &self.profiler }

    /// Plan a workload on the session's device using the pure planner.
    #[inline]
    pub fn plan(&self, workload: &Workload) -> Result<Strategy> {
        Ok(plan_strategy(&self.capability, workload))
    }

    /// Fence: wait until every command previously enqueued on this stream
    /// has completed, then resolve deferred profiler GPU spans.
    #[inline]
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()?;
        self.profiler.flush_gpu();
        Ok(())
    }
}

fn derive_capability(device: &Device) -> Result<Capability> {
    let (maj, min) = device.compute_capability()?;
    let total = device.total_mem()? as u64;
    let mem_clock = device.attribute(Attr::MemoryClockRate).unwrap_or(0) as u32;
    let bus_w = device.attribute(Attr::GlobalMemoryBusWidth).unwrap_or(0) as u32;
    let mut cap = capability_from_arch(maj as i32, min as i32, total, mem_clock, bus_w);
    if matches!(cap.tier, ComputeTier::Datacenter) {
        cap.flags |= CapabilityFlags::NCCL;
    }
    Ok(cap)
}
