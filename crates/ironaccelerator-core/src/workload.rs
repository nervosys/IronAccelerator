//! Workload descriptions used by the planner / ontology layer.
//!
//! A [`Workload`] is the input to backend [`Backend::plan`](crate::Backend::plan).
//! It is a structured, vendor-neutral statement of *what* the caller wants to
//! compute — the planner returns a [`Strategy`](crate::Strategy) with the
//! *how*.

use crate::dtype::DType;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};
#[cfg(feature = "std")]
use std::{string::String, vec::Vec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Precision {
    /// Use the highest precision the hardware supports natively.
    Highest,
    /// Use the precision implied by the input/output dtype (no internal
    /// upcast).
    Native,
    /// Mixed — accumulate in higher precision than inputs.
    Mixed,
    /// Lowest precision that meets `tolerance_bits` accuracy.
    Lowest { tolerance_bits: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum WorkloadKind {
    // ---- Linear algebra ---------------------------------------------------
    Gemm,
    Gemv,
    BatchedGemm,
    Conv2d,
    Conv3d,
    DepthwiseConv,
    // ---- Attention ------------------------------------------------------
    Attention,      // standard MHA
    FlashAttention, // memory-efficient
    PagedAttention, // vLLM-style KV-cache
    Mamba,          // SSM
    // ---- Reductions -----------------------------------------------------
    Reduce,
    Softmax,
    LayerNorm,
    RmsNorm,
    // ---- Element-wise ---------------------------------------------------
    Elementwise,
    Activation,
    // ---- Sampling / random ----------------------------------------------
    SampleTopK,
    SampleTopP,
    // ---- FFT / signal ---------------------------------------------------
    Fft,
    // ---- Sparse ---------------------------------------------------------
    SpMM,
    Sddmm,
    // ---- Quantisation ----------------------------------------------------
    Quantize,
    Dequantize,
    // ---- MoE / routing --------------------------------------------------
    MoEDispatch,
    MoECombine,
    // ---- Embedding ------------------------------------------------------
    Embedding,
    EmbeddingBag,
    // ---- Misc -----------------------------------------------------------
    Dropout,
    RotaryEmbedding,
    AllReduce,
    AllGather,
    ReduceScatter,
    // ---- Custom (user kernel) -------------------------------------------
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WorkloadShape {
    /// Batch size (B).
    pub batch: u32,
    /// Sequence / spatial extent 1 (M, H, T).
    pub m: u32,
    /// Inner extent (N, W).
    pub n: u32,
    /// Reduction extent (K, C).
    pub k: u32,
}

impl WorkloadShape {
    pub const fn matmul(m: u32, n: u32, k: u32) -> Self {
        Self { batch: 1, m, n, k }
    }
}

/// Inference vs training distinguishes the optimal kernel family on most
/// vendors (e.g. fused FlashAttention forward vs forward+backward).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Phase {
    Inference,
    Training,
    Calibration,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Workload {
    pub kind: WorkloadKind,
    pub shape: WorkloadShape,
    pub input_dtype: DType,
    pub output_dtype: DType,
    pub accum_dtype: DType,
    pub precision: Precision,
    pub phase: Phase,
    /// Free-form tags consumed by the ontology (e.g. "llm", "decode",
    /// "prefill", "vit", "diffusion").
    pub tags: Vec<String>,
}

impl Workload {
    /// Convenience constructor for a GEMM workload.
    pub fn gemm(m: u32, n: u32, k: u32, dt: DType) -> Self {
        Self {
            kind: WorkloadKind::Gemm,
            shape: WorkloadShape::matmul(m, n, k),
            input_dtype: dt,
            output_dtype: dt,
            accum_dtype: if matches!(dt, DType::F16 | DType::Bf16 | DType::F8E4M3 | DType::F8E5M2) {
                DType::F32
            } else {
                dt
            },
            precision: Precision::Mixed,
            phase: Phase::Inference,
            tags: Vec::new(),
        }
    }
}
