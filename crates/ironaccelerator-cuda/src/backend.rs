//! `Backend` trait implementation for CUDA.
//!
//! Uses the in-crate [`crate::drv::Device`] layer to enumerate devices and
//! read capability bits, then derives an IronAccelerator `Strategy` from
//! the workload + capability flags.

use crate::drv::Device;
use iron_cuda_sys::driver::CUdevice_attribute as Attr;
use ironaccelerator_core::{
    Backend, BackendKind, Capability, CapabilityFlags, ComputeTier, DType, DeviceDescriptor,
    DeviceId, Result, Strategy, Vendor, Workload, WorkloadKind,
    strategy::FlashVariant,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

pub struct CudaBackend {
    /// Cache of `(ordinal -> Arc<Device>)` so repeated planner calls don't
    /// re-retain the primary context.
    devices: RwLock<HashMap<u32, Arc<Device>>>,
}

pub static CUDA_BACKEND: LazyLock<CudaBackend> = LazyLock::new(|| CudaBackend {
    devices: RwLock::new(HashMap::new()),
});

impl CudaBackend {
    /// Get-or-create the cached primary device handle for an ordinal.
    pub fn device(&self, ordinal: u32) -> Result<Arc<Device>> {
        if let Some(d) = self.devices.read().get(&ordinal) {
            return Ok(d.clone());
        }
        let d = Device::open(ordinal)?;
        self.devices.write().insert(ordinal, d.clone());
        Ok(d)
    }

    pub fn capability(&self, ordinal: u32) -> Result<Capability> {
        let d = self.device(ordinal)?;
        let (maj, min) = d.compute_capability()?;
        let total = d.total_mem()? as u64;
        let mem_clock = d.attribute(Attr::MemoryClockRate).unwrap_or(0) as u32;
        let bus_w = d.attribute(Attr::GlobalMemoryBusWidth).unwrap_or(0) as u32;
        Ok(capability_from_arch(maj as i32, min as i32, total, mem_clock, bus_w))
    }
}

impl Backend for CudaBackend {
    fn kind(&self) -> BackendKind { BackendKind::Cuda }

    fn is_available(&self) -> bool {
        Device::count().unwrap_or(0) > 0
    }

    fn enumerate(&self) -> Result<Vec<DeviceDescriptor>> {
        let n = Device::count()?;
        let mut out = Vec::with_capacity(n as usize);
        for ord in 0..n {
            let d = match self.device(ord) { Ok(c) => c, Err(_) => continue };
            let name = d.name().unwrap_or_else(|_| "unknown".into());
            let (maj, min) = d.compute_capability().unwrap_or((0, 0));
            let arch = format!("sm_{maj}{min}");
            let total = d.total_mem().unwrap_or(0) as u64;
            let mp = d.attribute(Attr::MultiprocessorCount).unwrap_or(0) as u32;
            let clock = d.attribute(Attr::ClockRate).unwrap_or(0) as u32;
            let cap = self.capability(ord).unwrap_or_else(|_| empty_cap());

            out.push(DeviceDescriptor {
                id: DeviceId { backend: BackendKind::Cuda, ordinal: ord },
                vendor: Vendor::Nvidia,
                name,
                arch,
                total_memory_bytes: total,
                multiprocessor_count: mp,
                clock_khz: clock,
                capability: cap,
            });
        }
        Ok(out)
    }

    fn capabilities(&self, device: u32) -> Result<CapabilityFlags> {
        Ok(self.capability(device)?.flags)
    }

    fn score(&self, device: u32, w: &Workload) -> f32 {
        let cap = match self.capability(device) { Ok(c) => c, Err(_) => return 0.0 };
        heuristic_score(&cap, w)
    }

    fn plan(&self, device: u32, w: &Workload) -> Result<Strategy> {
        let cap = self.capability(device)?;
        Ok(plan_strategy(&cap, w))
    }
}

// ─── pure helpers (unit-testable without a live device) ─────────────────────

fn empty_cap() -> Capability {
    Capability {
        flags: CapabilityFlags::FP32 | CapabilityFlags::FP16,
        tier: ComputeTier::Consumer,
        fp16_tflops: None, fp8_tflops: None, mem_bandwidth_gbs: None,
    }
}

/// Translate (sm_major, sm_minor) and a few attributes into the IronAccelerator
/// capability descriptor. Pure function — directly unit-tested below.
pub fn capability_from_arch(
    major: i32, minor: i32, total_mem: u64, mem_clock_khz: u32, bus_width_bits: u32,
) -> Capability {
    use CapabilityFlags as F;
    let mut flags = F::FP64 | F::FP32 | F::FP16
        | F::HOST_PINNED | F::PEER_ACCESS | F::ASYNC_ALLOC
        | F::MULTI_STREAM | F::GRAPHS | F::COOPERATIVE_LAUNCH | F::NCCL;

    if major >= 7 { flags |= F::TENSOR_CORES | F::WMMA; }
    if major >= 8 {
        flags |= F::TF32 | F::BF16 | F::SPARSE_2_4 | F::FLASH_ATTN | F::UNIFIED_MEMORY;
    }
    if major >= 9 {
        flags |= F::FP8_E4M3 | F::FP8_E5M2 | F::TRANSFORMER_ENGINE | F::HBM | F::NVLINK;
    }
    if major >= 10 { flags |= F::FP4; }

    let tier = match (major, total_mem) {
        (m, _)              if m <= 6 => ComputeTier::Consumer,
        (7, _)                        => ComputeTier::Workstation,
        (8, t) if t > 40 * (1 << 30)  => ComputeTier::Datacenter,
        (8, _)                        => ComputeTier::Workstation,
        (9, _) | (10, _) | _          => ComputeTier::Datacenter,
    };

    // Effective HBM/GDDR bandwidth ≈ 2 (DDR) × mem_clock_khz × bus_width_bits / 8.
    let bw_gbs = if mem_clock_khz > 0 && bus_width_bits > 0 {
        Some(2.0 * (mem_clock_khz as f32 / 1.0e6) * (bus_width_bits as f32 / 8.0))
    } else { None };

    let _ = minor;
    Capability { flags, tier, fp16_tflops: None, fp8_tflops: None, mem_bandwidth_gbs: bw_gbs }
}

/// Score a workload against a capability. Higher = more preferred.
pub fn heuristic_score(cap: &Capability, w: &Workload) -> f32 {
    let mut s = 1.0;
    match cap.tier {
        ComputeTier::Datacenter   => s *= 4.0,
        ComputeTier::Workstation  => s *= 2.5,
        ComputeTier::Consumer     => s *= 1.5,
        ComputeTier::Mobile       => s *= 1.0,
        ComputeTier::Baseline     => s *= 0.25,
    }
    if w.input_dtype == DType::F8E4M3 && cap.flags.contains(CapabilityFlags::FP8_E4M3) { s *= 4.0; }
    if w.input_dtype == DType::Bf16   && cap.flags.contains(CapabilityFlags::BF16)     { s *= 2.0; }
    if matches!(w.kind, WorkloadKind::Gemm | WorkloadKind::BatchedGemm | WorkloadKind::Conv2d)
        && cap.flags.contains(CapabilityFlags::TENSOR_CORES) { s *= 2.0; }
    if matches!(w.kind, WorkloadKind::FlashAttention | WorkloadKind::PagedAttention)
        && cap.flags.contains(CapabilityFlags::FLASH_ATTN) { s *= 2.0; }
    s
}

/// Pure planner: capability + workload → Strategy. No driver calls.
pub fn plan_strategy(cap: &Capability, w: &Workload) -> Strategy {
    let f = cap.flags;
    match w.kind {
        WorkloadKind::Gemm | WorkloadKind::BatchedGemm => {
            if f.contains(CapabilityFlags::TRANSFORMER_ENGINE)
                && matches!(w.input_dtype, DType::F8E4M3 | DType::F8E5M2)
            {
                Strategy::TransformerEngine { recipe: "delayed-scaling-e4m3" }
            } else if f.contains(CapabilityFlags::TENSOR_CORES) {
                Strategy::BlasLt { epilogue: "bias-gelu" }
            } else {
                Strategy::VendorBlas { name: "cublas-sgemm" }
            }
        }
        WorkloadKind::Gemv => Strategy::VendorBlas { name: "cublas-gemv" },
        WorkloadKind::FlashAttention | WorkloadKind::Attention => {
            let v = if f.contains(CapabilityFlags::TRANSFORMER_ENGINE) {
                FlashVariant::V3
            } else if f.contains(CapabilityFlags::FLASH_ATTN) {
                FlashVariant::V2
            } else {
                FlashVariant::V2
            };
            Strategy::FusedAttention { variant: v }
        }
        WorkloadKind::PagedAttention => Strategy::FusedAttention { variant: FlashVariant::Paged },
        WorkloadKind::Conv2d | WorkloadKind::Conv3d | WorkloadKind::DepthwiseConv =>
            Strategy::VendorBlas { name: "cudnn-conv" },
        WorkloadKind::Fft   => Strategy::VendorBlas { name: "cufft" },
        WorkloadKind::SpMM  => Strategy::VendorBlas { name: "cusparse-spmm" },
        WorkloadKind::Sddmm => Strategy::VendorBlas { name: "cusparse-sddmm" },
        WorkloadKind::SampleTopK | WorkloadKind::SampleTopP =>
            Strategy::CutlassTemplate { tile: (256, 1, 1), stages: 2 },
        WorkloadKind::Softmax | WorkloadKind::LayerNorm | WorkloadKind::RmsNorm =>
            Strategy::CutlassTemplate { tile: (1024, 1, 1), stages: 1 },
        WorkloadKind::Elementwise | WorkloadKind::Activation =>
            Strategy::CutlassTemplate { tile: (1024, 1, 1), stages: 1 },
        WorkloadKind::Quantize | WorkloadKind::Dequantize =>
            Strategy::CutlassTemplate { tile: (256, 1, 1), stages: 1 },
        WorkloadKind::Mamba   => Strategy::TritonJit { signature: "mamba-scan-bf16".into() },
        WorkloadKind::Reduce  => Strategy::CutlassTemplate { tile: (1024, 1, 1), stages: 1 },
        WorkloadKind::Custom  => Strategy::Custom { name: "user".into() },
        // `WorkloadKind` is non_exhaustive — fall back to a JIT path for any
        // future variant the planner doesn't yet recognise.
        _ => Strategy::TritonJit { signature: "fallback".into() },
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ironaccelerator_core::DType;

    #[test]
    fn hopper_picks_transformer_engine_for_fp8_gemm() {
        let cap = capability_from_arch(9, 0, 80 * (1 << 30), 0, 0);
        assert!(cap.flags.contains(CapabilityFlags::FP8_E4M3));
        assert!(cap.flags.contains(CapabilityFlags::TRANSFORMER_ENGINE));
        let w = ironaccelerator_core::Workload::gemm(8192, 8192, 8192, DType::F8E4M3);
        match plan_strategy(&cap, &w) {
            Strategy::TransformerEngine { .. } => {}
            other => panic!("expected TE on Hopper FP8, got {other:?}"),
        }
    }

    #[test]
    fn ampere_picks_blaslt_for_bf16_gemm() {
        let cap = capability_from_arch(8, 0, 80 * (1 << 30), 0, 0);
        let w = ironaccelerator_core::Workload::gemm(4096, 4096, 4096, DType::Bf16);
        assert!(matches!(plan_strategy(&cap, &w), Strategy::BlasLt { .. }));
    }

    #[test]
    fn hopper_fa_picks_v3() {
        let cap = capability_from_arch(9, 0, 0, 0, 0);
        let w = ironaccelerator_core::Workload {
            kind: WorkloadKind::FlashAttention,
            ..ironaccelerator_core::Workload::gemm(1, 1, 1, DType::Bf16)
        };
        match plan_strategy(&cap, &w) {
            Strategy::FusedAttention { variant: FlashVariant::V3 } => {}
            other => panic!("expected FA-v3, got {other:?}"),
        }
    }

    #[test]
    fn datacenter_tier_outscores_consumer() {
        let dc = capability_from_arch(9, 0, 80 * (1 << 30), 0, 0);
        let cn = capability_from_arch(8, 6, 24 * (1 << 30), 0, 0);
        let w = ironaccelerator_core::Workload::gemm(4096, 4096, 4096, DType::F8E4M3);
        assert!(heuristic_score(&dc, &w) > heuristic_score(&cn, &w));
    }

    #[test]
    fn bandwidth_derived_from_attributes() {
        // H100 SXM: 1593 MHz × 5120-bit = ~2039 GB/s
        let cap = capability_from_arch(9, 0, 0, 1_593_000, 5120);
        let bw = cap.mem_bandwidth_gbs.expect("bw computed");
        assert!(bw > 1500.0 && bw < 2300.0, "bw={bw}");
    }
}
