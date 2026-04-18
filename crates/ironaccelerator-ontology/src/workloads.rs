//! Workload-class sub-graph.

use crate::{Id, Ontology, WorkloadClass};
use ironaccelerator_core::WorkloadKind;

fn w(id: &str, kind: WorkloadKind, bound: &str, desc: &str, tags: &[&str]) -> WorkloadClass {
    WorkloadClass {
        id: Id(id.into()),
        kind,
        bound_by: bound.into(),
        description: desc.into(),
        tags: tags.iter().map(|s| s.to_string()).collect(),
    }
}

pub fn populate(o: &mut Ontology) {
    let xs = [
        w("workload.gemm", WorkloadKind::Gemm, "compute",
          "Dense matrix multiply C = αAB + βC.", &["dense", "blas-3"]),
        w("workload.gemv", WorkloadKind::Gemv, "memory",
          "Matrix-vector product; bandwidth bound.", &["dense", "blas-2"]),
        w("workload.batched_gemm", WorkloadKind::BatchedGemm, "compute",
          "Batched dense matmul, common in attention QKV projections.", &["dense", "batched"]),
        w("workload.conv2d", WorkloadKind::Conv2d, "compute",
          "2D convolution forward/backward.", &["cnn", "vision"]),
        w("workload.attention", WorkloadKind::Attention, "memory",
          "Standard multi-head attention. Memory-bound on long sequences.",
          &["transformer", "llm"]),
        w("workload.flash_attention", WorkloadKind::FlashAttention, "compute",
          "Tiled, fused attention with O(N) memory.", &["transformer", "llm", "fused"]),
        w("workload.paged_attention", WorkloadKind::PagedAttention, "memory",
          "vLLM-style paged KV cache attention; decode phase.",
          &["transformer", "llm", "decode", "kv-cache"]),
        w("workload.mamba", WorkloadKind::Mamba, "memory",
          "State-space model (Mamba/Mamba2) selective scan.", &["ssm", "llm"]),
        w("workload.softmax", WorkloadKind::Softmax, "memory",
          "Numerically-stable row softmax.", &["reduction"]),
        w("workload.layernorm", WorkloadKind::LayerNorm, "memory",
          "LayerNorm with optional fused residual + bias.", &["norm"]),
        w("workload.rmsnorm", WorkloadKind::RmsNorm, "memory",
          "RMSNorm — preferred in LLaMA-family models.", &["norm", "llm"]),
        w("workload.elementwise", WorkloadKind::Elementwise, "memory",
          "Element-wise unary/binary op.", &["ew"]),
        w("workload.activation", WorkloadKind::Activation, "memory",
          "GELU/SiLU/SwiGLU activation.", &["ew", "llm"]),
        w("workload.sample_topk", WorkloadKind::SampleTopK, "compute",
          "Top-K logit sampling for autoregressive decoding.",
          &["llm", "decode", "sampling"]),
        w("workload.sample_topp", WorkloadKind::SampleTopP, "compute",
          "Top-P (nucleus) sampling.", &["llm", "decode", "sampling"]),
        w("workload.fft", WorkloadKind::Fft, "memory",
          "Fast Fourier transform (1D/2D/3D).", &["dsp"]),
        w("workload.spmm", WorkloadKind::SpMM, "memory",
          "Sparse matrix × dense matrix.", &["sparse"]),
        w("workload.quantize", WorkloadKind::Quantize, "memory",
          "Quantise FP→INT/FP8/FP4 with calibration.", &["quant"]),
        w("workload.dequantize", WorkloadKind::Dequantize, "memory",
          "Inverse of quantise; usually fused into matmul.", &["quant"]),

        // ── MoE ──────────────────────────────────────────────────────────
        w("workload.moe_dispatch", WorkloadKind::MoEDispatch, "memory",
          "Token-to-expert dispatch (permute by routing weights).",
          &["moe", "llm", "permute"]),
        w("workload.moe_combine", WorkloadKind::MoECombine, "memory",
          "Inverse permute / weighted sum after expert FFN.", &["moe", "llm"]),

        // ── Embeddings ──────────────────────────────────────────────────
        w("workload.embedding", WorkloadKind::Embedding, "memory",
          "Index-into-table lookup; gather op.", &["llm", "gather"]),
        w("workload.embedding_bag", WorkloadKind::EmbeddingBag, "memory",
          "Embedding + reduce; common in recsys.", &["recsys", "gather"]),

        // ── Auxiliary ----------------------------------------------------
        w("workload.dropout", WorkloadKind::Dropout, "memory",
          "Bernoulli-mask multiplication during training.",
          &["train", "stochastic"]),
        w("workload.rotary", WorkloadKind::RotaryEmbedding, "memory",
          "RoPE / rotary positional embeddings.", &["llm", "fused"]),

        // ── Collective comms (multi-device) ─────────────────────────────
        w("workload.allreduce", WorkloadKind::AllReduce, "memory",
          "All-reduce across N ranks (NCCL/RCCL).", &["dist", "comms"]),
        w("workload.allgather", WorkloadKind::AllGather, "memory",
          "All-gather across N ranks.", &["dist", "comms"]),
        w("workload.reducescatter", WorkloadKind::ReduceScatter, "memory",
          "Reduce-scatter across N ranks.", &["dist", "comms"]),
    ];

    for x in xs {
        o.workloads.insert(x.id.clone(), x);
    }
}
