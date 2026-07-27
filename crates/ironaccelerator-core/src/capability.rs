//! Coarse capability flags. Backends translate vendor-specific capability
//! tables (CUDA compute capability, ROCm gfx, Metal feature sets, QNN HTP
//! version) into this common space so callers can reason about hardware
//! uniformly without parsing vendor version strings.

use bitflags::bitflags;

bitflags! {
    /// Hardware capability flags. Stable across backends; new bits append.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct CapabilityFlags: u64 {
        // Numeric formats --------------------------------------------------
        const FP64           = 1 << 0;
        const FP32           = 1 << 1;
        const TF32           = 1 << 2;
        const FP16           = 1 << 3;
        const BF16           = 1 << 4;
        const FP8_E4M3       = 1 << 5;
        const FP8_E5M2       = 1 << 6;
        const FP4            = 1 << 7;
        const INT8           = 1 << 8;
        const INT4           = 1 << 9;

        // Tensor / matrix engines ------------------------------------------
        const TENSOR_CORES   = 1 << 16;     // Volta+ / CDNA / AMX-style MMA
        const WMMA           = 1 << 17;     // wavefront matrix-multiply ops
        const SPARSE_2_4     = 1 << 18;     // Ampere-style 2:4 sparsity
        const TRANSFORMER_ENGINE = 1 << 19; // Hopper+ TE / equivalent

        // Memory ------------------------------------------------------------
        const UNIFIED_MEMORY = 1 << 24;
        const HOST_PINNED    = 1 << 25;
        const PEER_ACCESS    = 1 << 26;
        const ASYNC_ALLOC    = 1 << 27;     // cudaMallocAsync etc.
        const HBM            = 1 << 28;
        const NVLINK         = 1 << 29;
        const INFINITY_FABRIC = 1 << 30;

        // Compute features -------------------------------------------------
        const COOPERATIVE_LAUNCH = 1 << 32;
        const DYNAMIC_PARALLELISM = 1 << 33;
        const GRAPHS         = 1 << 34;     // CUDA graphs / equivalent
        const MULTI_STREAM   = 1 << 35;
        const FLASH_ATTN     = 1 << 36;     // hardware-friendly fused attn
        const RAYTRACING     = 1 << 37;

        // Distribution -----------------------------------------------------
        const NCCL           = 1 << 40;
        const RCCL           = 1 << 41;
        const MPI_AWARE      = 1 << 42;

        // NPU / mobile -----------------------------------------------------
        const HVX            = 1 << 48;     // Hexagon Vector eXtensions
        const HMX            = 1 << 49;     // Hexagon Matrix eXtensions
        const ANE            = 1 << 50;     // Apple Neural Engine bridge
    }
}

/// A coarse "compute tier" used by strategy heuristics when capability bits
/// are insufficient. Backends map their micro-arch to one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ComputeTier {
    /// CPU SIMD baseline.
    Baseline,
    /// Mobile NPU / integrated GPU.
    Mobile,
    /// Consumer dGPU (e.g. RTX 4060, RX 7600, M-series Pro).
    Consumer,
    /// Workstation GPU (RTX 6000 Ada, Radeon Pro, M-series Max).
    Workstation,
    /// Datacenter GPU (H100, MI300, GB200).
    Datacenter,
}

/// Aggregated capability description that wraps flags + tier + numeric extras.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Capability {
    pub flags: CapabilityFlags,
    pub tier: ComputeTier,
    /// Peak FP16 TFLOPS, vendor-published. `None` if unknown.
    pub fp16_tflops: Option<f32>,
    /// Peak FP8 TFLOPS, vendor-published.
    pub fp8_tflops: Option<f32>,
    /// HBM/GDDR bandwidth in GB/s.
    pub mem_bandwidth_gbs: Option<f32>,
}
