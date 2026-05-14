//! Edges encode "this strategy is preferred on this hardware" with a weight.
//! The planner sums weights to break ties; default is `1.0`.

use crate::{Edge, Id, Ontology, Relation};

fn e(from: &str, to: &str, rel: Relation, w: f32, note: &str) -> Edge {
    Edge {
        from: Id(from.into()),
        to: Id(to.into()),
        relation: rel,
        weight: w,
        note: note.into(),
    }
}

pub fn populate(o: &mut Ontology) {
    let xs = [
        // ── NVIDIA preferences ───────────────────────────────────────────
        e(
            "strategy.cublaslt.fp8_te",
            "hardware.nvidia.h100",
            Relation::Prefers,
            3.0,
            "Hopper FP8 TE recipe: fastest GEMM.",
        ),
        e(
            "strategy.cublaslt.fp8_te",
            "hardware.nvidia.gb200",
            Relation::Prefers,
            3.5,
            "Blackwell extends FP8 to FP4.",
        ),
        e(
            "strategy.flashattn.v3",
            "hardware.nvidia.h100",
            Relation::Prefers,
            3.0,
            "FA3 needs sm_90a warp specialisation.",
        ),
        e(
            "strategy.flashattn.v3",
            "hardware.nvidia.gb200",
            Relation::Prefers,
            3.0,
            "",
        ),
        e(
            "strategy.flashattn.v2",
            "hardware.nvidia.a100",
            Relation::Prefers,
            2.5,
            "",
        ),
        e(
            "strategy.flashattn.v2",
            "hardware.nvidia.rtx4090",
            Relation::Prefers,
            2.0,
            "",
        ),
        e(
            "strategy.cudnn.flash",
            "hardware.nvidia.l40s",
            Relation::Prefers,
            2.0,
            "",
        ),
        e(
            "strategy.tensorrt_llm.engine",
            "hardware.nvidia.h100",
            Relation::Prefers,
            2.5,
            "Best end-to-end LLM serving when the graph is static.",
        ),
        e(
            "strategy.vllm.paged",
            "hardware.nvidia.h100",
            Relation::Prefers,
            2.0,
            "",
        ),
        e(
            "strategy.vllm.paged",
            "hardware.nvidia.a100",
            Relation::Prefers,
            2.0,
            "",
        ),
        // ── AMD preferences ──────────────────────────────────────────────
        e(
            "strategy.hipblaslt.fp8",
            "hardware.amd.mi300x",
            Relation::Prefers,
            3.0,
            "",
        ),
        e(
            "strategy.hipblaslt.fp8",
            "hardware.amd.mi325x",
            Relation::Prefers,
            3.0,
            "",
        ),
        e(
            "strategy.composable_kernel.gemm",
            "hardware.amd.mi300x",
            Relation::Prefers,
            2.5,
            "CK templates beat rocBLAS on attention shapes.",
        ),
        e(
            "strategy.flashattn.v2",
            "hardware.amd.mi300x",
            Relation::Prefers,
            2.0,
            "FA2 has good ROCm port via CK.",
        ),
        e(
            "strategy.vllm.paged",
            "hardware.amd.mi300x",
            Relation::Prefers,
            2.0,
            "",
        ),
        e(
            "strategy.rocblas.bf16",
            "hardware.amd.rx7900xtx",
            Relation::Prefers,
            1.5,
            "",
        ),
        // ── Apple preferences ────────────────────────────────────────────
        e(
            "strategy.mlx.attention",
            "hardware.apple.m3-max",
            Relation::Prefers,
            3.0,
            "",
        ),
        e(
            "strategy.mlx.attention",
            "hardware.apple.m4-max",
            Relation::Prefers,
            3.0,
            "",
        ),
        e(
            "strategy.mps.matmul",
            "hardware.apple.m2-ultra",
            Relation::Prefers,
            2.0,
            "",
        ),
        e(
            "strategy.coreml.ane",
            "hardware.apple.m4-max",
            Relation::Prefers,
            1.5,
            "ANE wins on perf-per-watt for compatible subgraphs.",
        ),
        // ── Qualcomm preferences ─────────────────────────────────────────
        e(
            "strategy.qnn.htp_int8",
            "hardware.qualcomm.snapdragon-x-elite",
            Relation::Prefers,
            3.0,
            "INT8 + HMX is the sweet spot.",
        ),
        e(
            "strategy.qnn.htp_int8",
            "hardware.qualcomm.snapdragon-8gen4",
            Relation::Prefers,
            2.5,
            "",
        ),
        e(
            "strategy.qnn.htp_fp16",
            "hardware.qualcomm.cloud-ai-100-ultra",
            Relation::Prefers,
            2.0,
            "",
        ),
        e(
            "strategy.qnn.htp_fp16",
            "hardware.qualcomm.cloud-ai-100",
            Relation::Prefers,
            1.5,
            "",
        ),
        // ── Extra NVIDIA part edges ─────────────────────────────────────
        e(
            "strategy.cublaslt.fp8_te",
            "hardware.nvidia.h200",
            Relation::Prefers,
            3.0,
            "",
        ),
        e(
            "strategy.cublaslt.fp8_te",
            "hardware.nvidia.b100",
            Relation::Prefers,
            3.5,
            "",
        ),
        e(
            "strategy.flashattn.v3",
            "hardware.nvidia.h200",
            Relation::Prefers,
            3.0,
            "",
        ),
        e(
            "strategy.flashattn.v3",
            "hardware.nvidia.b100",
            Relation::Prefers,
            3.5,
            "Blackwell extends FA-3 with FP4 GEMM in the matmul stage.",
        ),
        // ── Collective comms ────────────────────────────────────────────
        e(
            "strategy.nccl.allreduce",
            "hardware.nvidia.h100",
            Relation::Prefers,
            3.0,
            "NVLink + NCCL is the canonical multi-GPU path.",
        ),
        e(
            "strategy.nccl.allreduce",
            "hardware.nvidia.gb200",
            Relation::Prefers,
            3.5,
            "NVL72 fabric — tree collectives win at unusually large N.",
        ),
        e(
            "strategy.rccl.allreduce",
            "hardware.amd.mi300x",
            Relation::Prefers,
            3.0,
            "",
        ),
        e(
            "strategy.rccl.allreduce",
            "hardware.amd.mi355x",
            Relation::Prefers,
            3.0,
            "",
        ),
        // ── MoE ─────────────────────────────────────────────────────────
        e(
            "strategy.megablocks.moe",
            "hardware.nvidia.h100",
            Relation::Prefers,
            2.5,
            "",
        ),
        e(
            "strategy.megablocks.moe",
            "hardware.amd.mi300x",
            Relation::Prefers,
            2.0,
            "",
        ),
        e(
            "strategy.deepspeed.moe",
            "hardware.nvidia.a100",
            Relation::Prefers,
            2.0,
            "",
        ),
        // ── Embeddings / RoPE / Dropout ─────────────────────────────────
        e(
            "strategy.cuda.embedding_gather",
            "hardware.nvidia.h100",
            Relation::Prefers,
            2.0,
            "",
        ),
        e(
            "strategy.cuda.rope_fused",
            "hardware.nvidia.h100",
            Relation::Prefers,
            2.0,
            "",
        ),
        e(
            "strategy.cuda.rope_fused",
            "hardware.amd.mi300x",
            Relation::Prefers,
            1.5,
            "",
        ),
        e(
            "strategy.cuda.dropout_fused",
            "hardware.nvidia.a100",
            Relation::Prefers,
            1.5,
            "",
        ),
        // ── FFT ─────────────────────────────────────────────────────────
        e(
            "strategy.cufft.multi",
            "hardware.nvidia.h100",
            Relation::Prefers,
            1.5,
            "",
        ),
        e(
            "strategy.cufft.multi",
            "hardware.nvidia.a100",
            Relation::Prefers,
            1.5,
            "",
        ),
    ];

    for x in xs {
        o.edges.push(x);
    }
}
