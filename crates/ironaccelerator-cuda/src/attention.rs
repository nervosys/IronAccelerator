//! Attention strategy dispatch.
//!
//! The cuDNN frontend / Flash-Attention operator is wired up in
//! [`crate::flash_attention`]; this module is a **planner shim** that maps
//! an attention `Workload` onto the `Strategy` the CUDA backend would
//! execute (FA-v3, FA-v2, cuDNN fused MHA, or fall-back) and exposes the
//! pieces each path needs (qkv layout, scale, causal flag, head dim).
//!
//! The actual kernel lives either in a vendored NVRTC source (see
//! [`crate::kernel`]) or in the TE/cuDNN backends added later; this module
//! is the single decision point so the rest of the crate stays agnostic.

use crate::Session;
use ironaccelerator_core::strategy::FlashVariant;
use ironaccelerator_core::{CapabilityFlags, DType, Result, Strategy};

#[derive(Debug, Clone, Copy)]
pub struct AttentionParams {
    pub batch: u32,
    pub heads: u32,
    pub seq_q: u32,
    pub seq_k: u32,
    pub head_dim: u32,
    pub dtype: DType,
    pub causal: bool,
    pub softmax_scale: f32,
}

impl AttentionParams {
    /// `1 / sqrt(head_dim)` — the textbook default.
    #[inline] pub fn default_scale(head_dim: u32) -> f32 {
        1.0 / (head_dim as f32).sqrt()
    }

    #[inline] pub fn elements_qkv(&self) -> u64 {
        3 * self.batch as u64 * self.heads as u64 * self.seq_q as u64 * self.head_dim as u64
    }

    /// QK^T FLOPs (single-pass online softmax avoids the materialised pass).
    #[inline] pub fn flops(&self) -> u64 {
        // 2·B·H·Sq·Sk·D (QK^T) + 2·B·H·Sq·Sk·D (·V) = 4·B·H·Sq·Sk·D
        4 * self.batch as u64 * self.heads as u64
          * self.seq_q as u64 * self.seq_k as u64 * self.head_dim as u64
    }
}

/// Recommend a [`Strategy`] for this session + shape.
///
/// Rules of thumb:
/// - Hopper (`sm_90`) + FP8/BF16 + head_dim ≤ 256 → FA-v3
/// - Ampere (`sm_80`) + BF16/FP16 + head_dim ≤ 128 → FA-v2
/// - Anything else with a compatible cuDNN → cuDNN fused MHA
/// - Fallback → Triton JIT (signalled; caller must compile)
pub fn recommend(session: &Session, p: &AttentionParams) -> Result<Strategy> {
    let cap = session.capability();
    let has_flash = cap.flags.contains(CapabilityFlags::FLASH_ATTN);
    let has_fp8   = cap.flags.contains(CapabilityFlags::FP8_E4M3);

    let dtype_ok_v3 = matches!(p.dtype, DType::Bf16 | DType::F16 | DType::F8E4M3 | DType::F8E5M2);
    let dtype_ok_v2 = matches!(p.dtype, DType::Bf16 | DType::F16);

    // Hopper+ proxy: flash-attn + fp8 flags both present.
    if has_flash && has_fp8 && dtype_ok_v3 && p.head_dim <= 256 {
        return Ok(Strategy::FusedAttention { variant: FlashVariant::V3 });
    }
    // Ampere+ proxy: flash-attn present.
    if has_flash && dtype_ok_v2 && p.head_dim <= 128 {
        return Ok(Strategy::FusedAttention { variant: FlashVariant::V2 });
    }
    // GQA / MQA as fallback when head_dim is unusual.
    if dtype_ok_v2 {
        return Ok(Strategy::FusedAttention { variant: FlashVariant::Gqa });
    }
    Ok(Strategy::TritonJit { signature: format!(
        "attention:{:?}:{}x{}x{}:{}",
        p.dtype, p.heads, p.seq_q, p.head_dim, if p.causal { "causal" } else { "full" }
    ) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_scale_is_inv_sqrt_d() {
        let s = AttentionParams::default_scale(64);
        assert!((s - 0.125).abs() < 1e-6);
    }

    #[test]
    fn flops_scale_with_seq_squared() {
        let mut p = AttentionParams {
            batch: 1, heads: 1, seq_q: 128, seq_k: 128, head_dim: 64,
            dtype: DType::Bf16, causal: false, softmax_scale: 0.1,
        };
        let a = p.flops();
        p.seq_q *= 2; p.seq_k *= 2;
        let b = p.flops();
        assert_eq!(b, a * 4);
    }
}
