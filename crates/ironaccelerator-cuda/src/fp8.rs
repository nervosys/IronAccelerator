//! FP8 recipe builder — the "how do I actually drive a Hopper+/Blackwell FP8
//! GEMM" knob surface, decoupled from the cuBLASLt descriptor plumbing.
//!
//! A [`Fp8Recipe`] describes:
//! - operand dtype (`E4M3` or `E5M2`) per tensor
//! - scaling strategy (delayed per-tensor, block, or vec-16 sub-channel)
//! - fast-accumulate mode (BF16/FP32 accumulator on Hopper; FP32 on Blackwell)
//! - amax history depth for delayed scaling (Transformer-Engine convention)
//!
//! The recipe is consumed by the `TransformerEngine` strategy path; it is
//! independent from cuBLASLt's descriptor type so the same recipe can be
//! executed either via cuBLASLt directly or via a CUTLASS / Transformer
//! Engine kernel.

use ironaccelerator_core::DType;

/// Which FP8 format a tensor uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Dtype {
    /// E4M3 — larger max (~448), smaller subnormals. Default for forward
    /// activations and weights per the NVIDIA FP8 recipe.
    E4M3,
    /// E5M2 — larger dynamic range, coarser precision. Default for
    /// gradients / backward pass.
    E5M2,
}

impl Fp8Dtype {
    #[inline] pub fn as_dtype(self) -> DType {
        match self {
            Self::E4M3 => DType::F8E4M3,
            Self::E5M2 => DType::F8E5M2,
        }
    }
    /// Maximum representable magnitude — reference for amax clipping.
    #[inline] pub fn max_value(self) -> f32 {
        match self {
            Self::E4M3 => 448.0,
            Self::E5M2 => 57344.0,
        }
    }
}

/// How scales are produced / consumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleMode {
    /// One scale per tensor, updated from an amax history after every step.
    /// Transformer Engine's default. Cheapest; works for well-behaved
    /// activations.
    DelayedPerTensor,
    /// One scale per 1×16 block along the K dimension (Blackwell MX path).
    /// Preserves dynamic range on spiky tensors.
    Vec16,
    /// Fully static scales — caller owns the amax history and writes scales
    /// out of band. Useful for deployment with quantised weights.
    Static,
}

/// How the tensor-core MMA accumulates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumMode {
    /// FP32 accumulate — safest, standard.
    Fp32,
    /// BF16 accumulate ("fast accum"). ~10% throughput on Hopper, loses
    /// precision on long K. Use only when you've validated the model.
    Bf16Fast,
}

/// Amax history length — Transformer Engine's running window of absolute
/// maxima used to derive the next step's scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmaxHistoryLen {
    /// TE default.
    Len1024,
    Len16,
    Custom(u32),
}

impl AmaxHistoryLen {
    #[inline] pub fn as_u32(self) -> u32 {
        match self {
            Self::Len1024 => 1024,
            Self::Len16   => 16,
            Self::Custom(n) => n,
        }
    }
}

/// A full FP8 GEMM recipe, ready to hand to a launcher.
#[derive(Debug, Clone, Copy)]
pub struct Fp8Recipe {
    pub a: Fp8Dtype,
    pub b: Fp8Dtype,
    pub out: DType,       // typically Bf16 or Fp16
    pub scale: ScaleMode,
    pub accum: AccumMode,
    pub amax_history: AmaxHistoryLen,
    /// Whether to expose an amax tensor on the output (required for
    /// downstream delayed-scaling consumers).
    pub produce_output_amax: bool,
}

impl Default for Fp8Recipe {
    fn default() -> Self { Self::hopper_default() }
}

impl Fp8Recipe {
    /// NVIDIA's canonical Hopper forward-pass recipe: E4M3 × E4M3 → BF16,
    /// delayed per-tensor scaling, FP32 accumulate, 1024-deep amax history.
    pub fn hopper_default() -> Self {
        Self {
            a: Fp8Dtype::E4M3, b: Fp8Dtype::E4M3,
            out: DType::Bf16,
            scale: ScaleMode::DelayedPerTensor,
            accum: AccumMode::Fp32,
            amax_history: AmaxHistoryLen::Len1024,
            produce_output_amax: true,
        }
    }

    /// Backward pass on Hopper: E5M2 grads, larger dynamic range.
    pub fn hopper_backward() -> Self {
        Self {
            a: Fp8Dtype::E5M2, b: Fp8Dtype::E4M3,
            out: DType::Bf16,
            scale: ScaleMode::DelayedPerTensor,
            accum: AccumMode::Fp32,
            amax_history: AmaxHistoryLen::Len1024,
            produce_output_amax: true,
        }
    }

    /// Blackwell MX-style recipe: per-16-element sub-channel scales.
    pub fn blackwell_mx() -> Self {
        Self {
            a: Fp8Dtype::E4M3, b: Fp8Dtype::E4M3,
            out: DType::Bf16,
            scale: ScaleMode::Vec16,
            accum: AccumMode::Fp32,
            amax_history: AmaxHistoryLen::Len16,
            produce_output_amax: true,
        }
    }

    /// Aggressive serving recipe: BF16 accumulate, static scales, no amax.
    pub fn serving_fast() -> Self {
        Self {
            a: Fp8Dtype::E4M3, b: Fp8Dtype::E4M3,
            out: DType::Bf16,
            scale: ScaleMode::Static,
            accum: AccumMode::Bf16Fast,
            amax_history: AmaxHistoryLen::Custom(1),
            produce_output_amax: false,
        }
    }

    /// Quick correctness check: catch obviously nonsensical recipes early.
    pub fn validate(&self) -> Result<(), &'static str> {
        if !matches!(self.out, DType::Bf16 | DType::F16 | DType::F32) {
            return Err("FP8 output dtype must be Bf16/Fp16/Fp32");
        }
        if matches!(self.scale, ScaleMode::Static) && self.produce_output_amax {
            return Err("Static scale recipes cannot produce output amax");
        }
        if self.amax_history.as_u32() == 0 {
            return Err("amax history length must be >= 1");
        }
        Ok(())
    }

    /// Bytes per element of the output — handy for sizing output buffers.
    #[inline] pub fn out_elem_bytes(&self) -> u32 {
        match self.out {
            DType::Bf16 | DType::F16 => 2,
            DType::F32 => 4,
            _ => 2,
        }
    }

    /// Human-readable tag, useful for NVTX ranges and log lines.
    pub fn tag(&self) -> String {
        format!(
            "fp8[{a:?}x{b:?}->{out:?}:{scale:?}:{accum:?}]",
            a = self.a, b = self.b, out = self.out,
            scale = self.scale, accum = self.accum,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hopper_default_validates() {
        assert!(Fp8Recipe::hopper_default().validate().is_ok());
        assert!(Fp8Recipe::hopper_backward().validate().is_ok());
        assert!(Fp8Recipe::blackwell_mx().validate().is_ok());
        assert!(Fp8Recipe::serving_fast().validate().is_ok());
    }

    #[test]
    fn static_scales_forbid_output_amax() {
        let mut r = Fp8Recipe::serving_fast();
        r.produce_output_amax = true;
        assert!(r.validate().is_err());
    }

    #[test]
    fn zero_history_is_rejected() {
        let mut r = Fp8Recipe::hopper_default();
        r.amax_history = AmaxHistoryLen::Custom(0);
        assert!(r.validate().is_err());
    }

    #[test]
    fn max_values_match_ieee_fp8() {
        assert_eq!(Fp8Dtype::E4M3.max_value(), 448.0);
        assert_eq!(Fp8Dtype::E5M2.max_value(), 57344.0);
    }

    #[test]
    fn tags_are_informative() {
        let t = Fp8Recipe::hopper_default().tag();
        assert!(t.contains("E4M3"));
        assert!(t.contains("Bf16"));
    }
}
