//! CPU reference activations. Backends validate fused activation kernels
//! against these. Not intended to be fast — intended to be obviously
//! correct.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// SiLU / swish: `x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Apply SiLU element-wise in place.
pub fn silu_inplace(xs: &mut [f32]) {
    for v in xs {
        *v = silu(*v);
    }
}

/// SwiGLU gated activation as used by LLaMA-family FFNs and most
/// modern MoE expert blocks:
///
/// ```text
/// up   = x @ W_up
/// gate = x @ W_gate
/// y    = silu(gate) * up  // element-wise
/// ```
///
/// This helper assumes the caller has already produced `up` and `gate`
/// (the two halves of the gated-linear projection) and writes
/// `out[i] = silu(gate[i]) * up[i]`.
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    for i in 0..gate.len() {
        out[i] = silu(gate[i]) * up[i];
    }
}

/// Variant that reads gate + up from a single interleaved buffer laid
/// out as `[gate_0, up_0, gate_1, up_1, ...]` — the default memory
/// layout produced by fused up+gate projections in LLaMA-style FFNs.
pub fn swiglu_interleaved(gate_up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate_up.len(), out.len() * 2);
    for i in 0..out.len() {
        let g = gate_up[2 * i];
        let u = gate_up[2 * i + 1];
        out[i] = silu(g) * u;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silu_matches_reference_points() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        // silu(1) ≈ 0.7310586
        assert!((silu(1.0) - 0.7310586).abs() < 1e-5);
        // silu(-1) ≈ -0.26894143
        assert!((silu(-1.0) - (-0.26894143)).abs() < 1e-5);
    }

    #[test]
    fn swiglu_matches_manual() {
        let gate = vec![0.5, -0.5, 2.0];
        let up = vec![1.0, 1.0, 0.5];
        let mut out = vec![0.0; 3];
        swiglu(&gate, &up, &mut out);
        for i in 0..3 {
            let expect = silu(gate[i]) * up[i];
            assert!((out[i] - expect).abs() < 1e-6);
        }
    }

    #[test]
    fn swiglu_interleaved_matches_split() {
        let gate = vec![0.1, 0.3, -0.7, 1.2];
        let up = vec![0.9, 2.1, -0.4, 0.8];
        let interleaved: Vec<f32> = gate.iter().zip(&up).flat_map(|(g, u)| [*g, *u]).collect();
        let mut a = vec![0.0; 4];
        let mut b = vec![0.0; 4];
        swiglu(&gate, &up, &mut a);
        swiglu_interleaved(&interleaved, &mut b);
        for i in 0..4 {
            assert!((a[i] - b[i]).abs() < 1e-6);
        }
    }
}
