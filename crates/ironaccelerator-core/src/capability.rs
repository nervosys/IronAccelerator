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
        /// 64-bit IEEE floating point.
        const FP64           = 1 << 0;
        /// 32-bit IEEE floating point.
        const FP32           = 1 << 1;
        /// TensorFloat-32 matrix math.
        const TF32           = 1 << 2;
        /// 16-bit IEEE half-precision floating point.
        const FP16           = 1 << 3;
        /// bfloat16 floating point.
        const BF16           = 1 << 4;
        /// 8-bit float with 4 exponent and 3 mantissa bits.
        const FP8_E4M3       = 1 << 5;
        /// 8-bit float with 5 exponent and 2 mantissa bits.
        const FP8_E5M2       = 1 << 6;
        /// 4-bit floating point.
        const FP4            = 1 << 7;
        /// 8-bit integer arithmetic.
        const INT8           = 1 << 8;
        /// 4-bit integer arithmetic.
        const INT4           = 1 << 9;

        // Tensor / matrix engines ------------------------------------------
        /// Dedicated matrix-multiply-accumulate engines (Volta+ / CDNA / AMX-style).
        const TENSOR_CORES   = 1 << 16;     // Volta+ / CDNA / AMX-style MMA
        /// Wavefront matrix-multiply operations.
        const WMMA           = 1 << 17;     // wavefront matrix-multiply ops
        /// Hardware 2:4 structured sparsity (Ampere-style).
        const SPARSE_2_4     = 1 << 18;     // Ampere-style 2:4 sparsity
        /// Hopper-or-later transformer engine, or an equivalent.
        const TRANSFORMER_ENGINE = 1 << 19; // Hopper+ TE / equivalent

        // Memory ------------------------------------------------------------
        /// Unified memory addressable from host and device.
        const UNIFIED_MEMORY = 1 << 24;
        /// Page-locked host memory accessible from the device.
        const HOST_PINNED    = 1 << 25;
        /// Direct access between peer devices.
        const PEER_ACCESS    = 1 << 26;
        /// Stream-ordered asynchronous allocation (e.g. `cudaMallocAsync`).
        const ASYNC_ALLOC    = 1 << 27;     // cudaMallocAsync etc.
        /// High-bandwidth stacked memory.
        const HBM            = 1 << 28;
        /// NVIDIA NVLink peer interconnect.
        const NVLINK         = 1 << 29;
        /// AMD Infinity Fabric peer interconnect.
        const INFINITY_FABRIC = 1 << 30;

        // Compute features -------------------------------------------------
        /// Cooperative kernel launch with grid-wide synchronisation.
        const COOPERATIVE_LAUNCH = 1 << 32;
        /// Kernels can launch kernels from the device.
        const DYNAMIC_PARALLELISM = 1 << 33;
        /// Captured work graphs (CUDA graphs or equivalent).
        const GRAPHS         = 1 << 34;     // CUDA graphs / equivalent
        /// Multiple concurrent streams or queues.
        const MULTI_STREAM   = 1 << 35;
        /// Hardware-friendly fused attention kernels.
        const FLASH_ATTN     = 1 << 36;     // hardware-friendly fused attn
        /// Hardware ray-tracing units.
        const RAYTRACING     = 1 << 37;

        // Distribution -----------------------------------------------------
        /// NCCL collective communication is available.
        const NCCL           = 1 << 40;
        /// RCCL collective communication is available.
        const RCCL           = 1 << 41;
        /// MPI-aware device buffers.
        const MPI_AWARE      = 1 << 42;

        // NPU / mobile -----------------------------------------------------
        /// Hexagon Vector eXtensions.
        const HVX            = 1 << 48;     // Hexagon Vector eXtensions
        /// Hexagon Matrix eXtensions.
        const HMX            = 1 << 49;     // Hexagon Matrix eXtensions
        /// Apple Neural Engine bridge.
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
    /// Capability bits for the device.
    pub flags: CapabilityFlags,
    /// Coarse compute tier used by strategy heuristics.
    pub tier: ComputeTier,
    /// Peak FP16 TFLOPS, vendor-published. `None` if unknown.
    pub fp16_tflops: Option<f32>,
    /// Peak FP8 TFLOPS, vendor-published.
    pub fp8_tflops: Option<f32>,
    /// HBM/GDDR bandwidth in GB/s.
    pub mem_bandwidth_gbs: Option<f32>,
}
