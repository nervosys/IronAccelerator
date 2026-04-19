//! Post-training quantization primitives.
//!
//! Scope: the *scheme* description, the *parameters* produced by
//! calibration, and CPU reference quant / dequant. Backends then consume
//! these parameters to dispatch hardware INT8 / INT4 / FP8 kernels.
//!
//! We distinguish three axes:
//! - **Granularity** — per-tensor, per-channel, or per-group (block).
//! - **Symmetry** — symmetric (zero-point = 0) or asymmetric.
//! - **Storage** — INT8, INT4 (packed two per byte), FP8.
//!
//! Everything in this module is backend-agnostic. GPU dequant kernels live
//! with the backends; this module is the oracle they are tested against.

use crate::dtype::DType;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ── Scheme ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum QuantGranularity {
    /// One scale (and optional zero-point) for the whole tensor.
    PerTensor,
    /// One scale per output channel (axis 0). The classic weight-only
    /// INT8 / INT4 GPTQ layout.
    PerChannel,
    /// One scale per `group_size`-wide block along axis 1 (input dim).
    /// Used by AWQ, GPTQ-group, and GGUF K-quants.
    PerGroup { group_size: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum QuantSymmetry { Symmetric, Asymmetric }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QuantScheme {
    pub storage: DType,
    pub granularity: QuantGranularity,
    pub symmetry: QuantSymmetry,
}

impl QuantScheme {
    pub const fn int8_per_channel_sym() -> Self {
        Self {
            storage: DType::I8,
            granularity: QuantGranularity::PerChannel,
            symmetry: QuantSymmetry::Symmetric,
        }
    }
    pub const fn int4_per_group_sym(group_size: u32) -> Self {
        Self {
            storage: DType::U4,
            granularity: QuantGranularity::PerGroup { group_size },
            symmetry: QuantSymmetry::Symmetric,
        }
    }
    pub const fn int8_per_tensor_asym() -> Self {
        Self {
            storage: DType::I8,
            granularity: QuantGranularity::PerTensor,
            symmetry: QuantSymmetry::Asymmetric,
        }
    }

    /// Range `[qmin, qmax]` of the stored integer type for this scheme.
    pub const fn qrange(&self) -> (i32, i32) {
        match (self.storage, self.symmetry) {
            (DType::I8, QuantSymmetry::Symmetric)  => (-127, 127),
            (DType::I8, QuantSymmetry::Asymmetric) => (-128, 127),
            (DType::U8, _)                         => (0, 255),
            (DType::U4, _)                         => (0, 15),
            (DType::I16, QuantSymmetry::Symmetric) => (-32767, 32767),
            _ => (0, 0),
        }
    }
}

// ── Calibration stats → parameters ─────────────────────────────────────────

/// Per-axis min / max statistics collected during calibration.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CalibStats {
    pub min: Vec<f32>,
    pub max: Vec<f32>,
}

impl CalibStats {
    pub fn new_scalar(min: f32, max: f32) -> Self {
        Self { min: vec![min], max: vec![max] }
    }

    pub fn per_channel(c: usize) -> Self {
        Self { min: vec![f32::INFINITY; c], max: vec![f32::NEG_INFINITY; c] }
    }

    /// Merge a row's observations into channel-wise stats (axis 0 = rows,
    /// axis 1 = channels).
    pub fn absorb_row(&mut self, row: &[f32]) {
        assert_eq!(row.len(), self.min.len());
        for (i, &v) in row.iter().enumerate() {
            if v < self.min[i] { self.min[i] = v; }
            if v > self.max[i] { self.max[i] = v; }
        }
    }
}

/// Concrete scale / zero-point table produced by calibration.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QuantParams {
    pub scheme: QuantScheme,
    /// One scale per channel / group — length depends on `scheme.granularity`.
    pub scales: Vec<f32>,
    /// Zero-points. Empty for symmetric schemes.
    pub zero_points: Vec<i32>,
}

impl QuantParams {
    /// Derive per-channel symmetric INT8 params from calibration stats.
    pub fn int8_per_channel_sym(stats: &CalibStats) -> Self {
        let mut scales = Vec::with_capacity(stats.min.len());
        for (lo, hi) in stats.min.iter().zip(&stats.max) {
            let amax = lo.abs().max(hi.abs()).max(f32::EPSILON);
            scales.push(amax / 127.0);
        }
        Self {
            scheme: QuantScheme::int8_per_channel_sym(),
            scales, zero_points: Vec::new(),
        }
    }

    /// Derive per-tensor asymmetric INT8 params from scalar min/max.
    pub fn int8_per_tensor_asym(min: f32, max: f32) -> Self {
        let (qmin, qmax) = QuantScheme::int8_per_tensor_asym().qrange();
        let qrange = (qmax - qmin) as f32;
        let scale = (max - min) / qrange;
        let zp = (qmin as f32 - min / scale).round() as i32;
        Self {
            scheme: QuantScheme::int8_per_tensor_asym(),
            scales: vec![scale.max(f32::EPSILON)],
            zero_points: vec![zp.clamp(qmin, qmax)],
        }
    }
}

// ── CPU reference quant / dequant (row-major matrices) ────────────────────

fn sat_i8(v: f32) -> i8 {
    v.round().clamp(-128.0, 127.0) as i8
}
fn sat_u4(v: f32) -> u8 {
    v.round().clamp(0.0, 15.0) as u8
}

/// Quantize an FP32 row-major matrix `[rows, cols]` to INT8 with per-channel
/// symmetric scales (channels along `cols`). Output is row-major INT8.
pub fn quant_i8_per_channel_sym(
    src: &[f32], rows: usize, cols: usize, params: &QuantParams, dst: &mut [i8],
) {
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert_eq!(dst.len(), rows * cols);
    debug_assert_eq!(params.scales.len(), cols);
    for r in 0..rows {
        let base = r * cols;
        crate::simd::quant_i8_row(
            &src[base..base + cols],
            &params.scales[..cols],
            &mut dst[base..base + cols],
        );
    }
    let _ = sat_i8;
}

/// Dequantize INT8 (per-channel symmetric) back to FP32.
pub fn dequant_i8_per_channel_sym(
    src: &[i8], rows: usize, cols: usize, params: &QuantParams, dst: &mut [f32],
) {
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert_eq!(dst.len(), rows * cols);
    debug_assert_eq!(params.scales.len(), cols);
    for r in 0..rows {
        let base = r * cols;
        crate::simd::dequant_i8_row(
            &src[base..base + cols],
            &params.scales[..cols],
            &mut dst[base..base + cols],
        );
    }
}

/// Quantize a per-group symmetric INT4 weight `[rows, cols]`, packing two
/// nibbles per output byte along `cols`. `cols` must be a multiple of 2;
/// `group_size` must divide `cols`. Output layout: `[rows, cols / 2]` bytes,
/// lower nibble = column `2i`, upper nibble = column `2i + 1`. Scales: one
/// per `(row, group)` in row-major `[rows, cols / group_size]`.
pub fn quant_u4_per_group_sym(
    src: &[f32], rows: usize, cols: usize, group_size: usize,
    packed_out: &mut [u8], scales_out: &mut [f32],
) {
    assert!(cols % 2 == 0, "quant_u4: cols must be even");
    assert!(cols % group_size == 0, "quant_u4: group_size must divide cols");
    let gpr = cols / group_size;
    assert_eq!(scales_out.len(), rows * gpr);
    assert_eq!(packed_out.len(), rows * cols / 2);

    for r in 0..rows {
        let row = &src[r * cols..(r + 1) * cols];
        let packed_row = &mut packed_out[r * cols / 2..(r + 1) * cols / 2];
        let scales_row = &mut scales_out[r * gpr..(r + 1) * gpr];
        for g in 0..gpr {
            let group = &row[g * group_size..(g + 1) * group_size];
            // Symmetric: centre on 7.5; scale = amax / 7.5
            let amax = group.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(f32::EPSILON);
            let scale = amax / 7.5;
            scales_row[g] = scale;
            let inv = 1.0 / scale;
            for (k, &v) in group.iter().enumerate() {
                let q = sat_u4(v * inv + 7.5);
                let col = g * group_size + k;
                let byte = &mut packed_row[col / 2];
                if col % 2 == 0 {
                    *byte = (*byte & 0xF0) | (q & 0x0F);
                } else {
                    *byte = (*byte & 0x0F) | ((q & 0x0F) << 4);
                }
            }
        }
    }
}

/// Quantize FP32 to INT8 with per-tensor asymmetric params
/// (one scale + one zero-point). `dst[i] = clip(round(src[i] / s) + zp)`.
pub fn quant_i8_per_tensor_asym(src: &[f32], params: &QuantParams, dst: &mut [i8]) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(params.scales.len(), 1);
    debug_assert_eq!(params.zero_points.len(), 1);
    let s = params.scales[0].max(f32::EPSILON);
    let zp = params.zero_points[0];
    let (qmin, qmax) = params.scheme.qrange();
    let inv = 1.0 / s;
    for (d, &v) in dst.iter_mut().zip(src) {
        let q = (v * inv).round() as i32 + zp;
        *d = q.clamp(qmin, qmax) as i8;
    }
}

/// Dequantize INT8 with per-tensor asymmetric params back to FP32.
pub fn dequant_i8_per_tensor_asym(src: &[i8], params: &QuantParams, dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    let s = params.scales[0];
    let zp = params.zero_points[0];
    for (d, &q) in dst.iter_mut().zip(src) {
        *d = (q as i32 - zp) as f32 * s;
    }
}

// ── FP8 scale calibration ──────────────────────────────────────────────────

/// Which FP8 variant a scale is being calibrated for. The two IEEE
/// sub-formats supported by CUDA 11.8+ / H100 / MI300 / transformer-engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Fp8Format {
    /// 4-bit exponent, 3-bit mantissa. Max finite ≈ 448. Default for
    /// weight / activation tensors in TE.
    E4M3,
    /// 5-bit exponent, 2-bit mantissa. Max finite ≈ 57344. Default for
    /// gradient tensors.
    E5M2,
}

impl Fp8Format {
    /// Largest finite representable magnitude.
    pub const fn absmax(self) -> f32 {
        match self {
            Fp8Format::E4M3 => 448.0,
            Fp8Format::E5M2 => 57344.0,
        }
    }
}

/// Calibrate a per-tensor FP8 scale from observed `absmax`. Returns the
/// multiplier that maps tensor values into the FP8 finite range —
/// `fp8 = clip(x * scale)` on the way in, `fp = fp8 / scale` on the way
/// out. Matches the transformer-engine `amax_history → scale` recipe:
/// `scale = fp8_max / max(absmax, eps)`.
pub fn fp8_scale_from_absmax(absmax: f32, format: Fp8Format) -> f32 {
    let absmax = absmax.max(f32::EPSILON);
    format.absmax() / absmax
}

/// Delayed-scaling helper: given a history window of observed `absmax`
/// values, return the scale to use for the *next* step. Transformer-engine
/// uses the max over the history; we mirror that policy.
pub fn fp8_scale_from_history(history: &[f32], format: Fp8Format) -> f32 {
    let amax = history.iter().copied().fold(0.0f32, f32::max);
    fp8_scale_from_absmax(amax, format)
}

/// Dequantize U4 packed weights (per-group symmetric) back to FP32.
pub fn dequant_u4_per_group_sym(
    packed: &[u8], rows: usize, cols: usize, group_size: usize,
    scales: &[f32], dst: &mut [f32],
) {
    let gpr = cols / group_size;
    for r in 0..rows {
        let packed_row = &packed[r * cols / 2..(r + 1) * cols / 2];
        let scales_row = &scales[r * gpr..(r + 1) * gpr];
        let dst_row = &mut dst[r * cols..(r + 1) * cols];
        for c in 0..cols {
            let byte = packed_row[c / 2];
            let q = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            let g = c / group_size;
            dst_row[c] = (q as f32 - 7.5) * scales_row[g];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_roundtrip_matches_within_scale() {
        let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let mut stats = CalibStats::per_channel(8);
        for row in src.chunks(8) { stats.absorb_row(row); }
        let params = QuantParams::int8_per_channel_sym(&stats);
        let mut q = vec![0i8; 32];
        quant_i8_per_channel_sym(&src, 4, 8, &params, &mut q);
        let mut back = vec![0f32; 32];
        dequant_i8_per_channel_sym(&q, 4, 8, &params, &mut back);
        for (a, b) in src.iter().zip(&back) {
            assert!((a - b).abs() < 0.02, "i8 round-trip too lossy: {a} vs {b}");
        }
    }

    #[test]
    fn int8_asym_roundtrip_reasonable() {
        let src: Vec<f32> = (0..32).map(|i| i as f32 * 0.05).collect(); // [0, 1.55]
        let params = QuantParams::int8_per_tensor_asym(0.0, 1.6);
        let mut q = vec![0i8; 32];
        quant_i8_per_tensor_asym(&src, &params, &mut q);
        let mut back = vec![0f32; 32];
        dequant_i8_per_tensor_asym(&q, &params, &mut back);
        for (a, b) in src.iter().zip(&back) {
            assert!((a - b).abs() < 0.02, "i8-asym lossy: {a} vs {b}");
        }
    }

    #[test]
    fn fp8_scale_inverts_absmax() {
        let s = fp8_scale_from_absmax(1.5, Fp8Format::E4M3);
        assert!((1.5_f32 * s - 448.0).abs() < 1e-3);
        let s5 = fp8_scale_from_absmax(200.0, Fp8Format::E5M2);
        assert!((200.0_f32 * s5 - 57344.0).abs() < 1e-1);
        let hist = vec![0.9, 1.2, 0.7, 1.5, 1.0];
        let sh = fp8_scale_from_history(&hist, Fp8Format::E4M3);
        assert!((sh - s).abs() < 1e-4);
    }

    #[test]
    fn u4_roundtrip_reasonable() {
        let src: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let mut packed = vec![0u8; 32];
        let mut scales = vec![0f32; 8]; // 2 rows × 4 groups(8)
        quant_u4_per_group_sym(&src, 2, 32, 8, &mut packed, &mut scales);
        let mut back = vec![0f32; 64];
        dequant_u4_per_group_sym(&packed, 2, 32, 8, &scales, &mut back);
        let max_err = src.iter().zip(&back).map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.2, "u4 error {max_err} too large");
    }
}
