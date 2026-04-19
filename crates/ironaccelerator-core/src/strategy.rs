//! Execution strategy: the planner's *answer* for a workload.
//!
//! Strategies are designed to be readable by both humans and agents — every
//! variant carries enough metadata for the ontology layer to explain *why*
//! the planner picked it.

use crate::{backend::BackendKind, dtype::DType};

#[cfg(feature = "std")]
use std::{string::String, vec::Vec};
#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StrategyScore {
    /// Predicted throughput (TFLOPS, samples/s, tokens/s — kind-dependent).
    pub throughput: f32,
    /// Predicted memory traffic in GB/s.
    pub bandwidth: f32,
    /// Predicted peak memory in MiB.
    pub memory_mib: f32,
    /// Confidence in [0, 1].
    pub confidence: f32,
}

/// Family of kernel implementations that can satisfy a workload. Each backend
/// supplies its own concrete kernel pipeline behind the matching variant.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Strategy {
    /// Pure cuBLAS / rocBLAS / MPSMatrix call.
    VendorBlas { name: &'static str },
    /// cuBLASLt / hipBLASLt epilogue-fused matmul.
    BlasLt { epilogue: &'static str },
    /// CUTLASS / Composable Kernel template instantiation.
    CutlassTemplate { tile: (u32, u32, u32), stages: u32 },
    /// Triton-generated kernel cached in JIT store.
    TritonJit { signature: String },
    /// FlashAttention-style fused kernel (vendor-specific impl).
    FusedAttention { variant: FlashVariant },
    /// Tensor Cores via WMMA / MFMA (architecture-specific).
    MatrixCore { mma_shape: (u32, u32, u32) },
    /// Transformer Engine (Hopper+) microscaled FP8.
    TransformerEngine { recipe: &'static str },
    /// MPSGraph (Apple) compiled subgraph.
    MpsGraph,
    /// QNN HTP graph (Qualcomm).
    QnnHtpGraph { precision: DType },
    /// Naive reference kernel — used for correctness/oracle, not perf.
    Reference,
    /// SPIR-V compute shader dispatched through Vulkan or Level Zero.
    SpirvCompute { workgroup: (u32, u32, u32) },
    /// WGSL compute shader dispatched through WebGPU.
    Wgsl { workgroup: (u32, u32, u32) },
    /// GLSL 4.3+ compute shader via OpenGL.
    GlslCompute { workgroup: (u32, u32, u32) },
    /// Google TPU via the PJRT plugin interface.
    Pjrt { accelerator: &'static str },
    /// AWS Neuron NEFF executing on one or more NeuronCores.
    Neuron { num_cores: u32 },
    /// Intel Level Zero compute kernel (GPU or NPU).
    LevelZero { device_type: &'static str },
    /// User-supplied custom kernel.
    Custom { name: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FlashVariant {
    V2,
    V3,
    Paged,
    Mqa,
    Gqa,
}

/// Hint provided by the agent / caller to bias the planner. `None` means the
/// planner is free to choose.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StrategyHint {
    /// Restrict to these backends, in priority order.
    pub prefer_backends: Vec<BackendKind>,
    /// Cap the predicted memory in MiB.
    pub memory_budget_mib: Option<f32>,
    /// Minimum acceptable confidence.
    pub min_confidence: Option<f32>,
    /// Disallow JIT compilation (cold-start sensitive).
    pub forbid_jit: bool,
    /// Disallow vendor closed-source paths.
    pub require_open_source: bool,
}
