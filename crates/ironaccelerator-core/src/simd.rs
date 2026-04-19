//! Runtime-dispatched SIMD primitives for CPU fallback paths.
//!
//! These are the oracle that backends validate their fused quant/dequant
//! kernels against. AVX2 on x86_64 when available; scalar elsewhere. Kept
//! simple — a handful of row-wise kernels, not a general SIMD library.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Per-channel INT8 symmetric quantization of a row.
/// `dst[i] = round(src[i] / scales[i])` saturated to `[-128, 127]`.
pub fn quant_i8_row(src: &[f32], scales: &[f32], dst: &mut [i8]) {
    debug_assert_eq!(src.len(), scales.len());
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { quant_i8_row_avx2(src, scales, dst) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { quant_i8_row_neon(src, scales, dst) };
            return;
        }
    }
    quant_i8_row_scalar(src, scales, dst);
}

/// Per-channel INT8 symmetric dequantization of a row.
/// `dst[i] = src[i] as f32 * scales[i]`.
pub fn dequant_i8_row(src: &[i8], scales: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), scales.len());
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            unsafe { dequant_i8_row_avx2(src, scales, dst) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { dequant_i8_row_neon(src, scales, dst) };
            return;
        }
    }
    dequant_i8_row_scalar(src, scales, dst);
}

fn quant_i8_row_scalar(src: &[f32], scales: &[f32], dst: &mut [i8]) {
    for i in 0..src.len() {
        let s = scales[i];
        let inv = if s > f32::EPSILON { 1.0 / s } else { 0.0 };
        let q = (src[i] * inv).round().clamp(-128.0, 127.0);
        dst[i] = q as i8;
    }
}

fn dequant_i8_row_scalar(src: &[i8], scales: &[f32], dst: &mut [f32]) {
    for i in 0..src.len() {
        dst[i] = src[i] as f32 * scales[i];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn quant_i8_row_avx2(src: &[f32], scales: &[f32], dst: &mut [i8]) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;

    let n = src.len();
    let mut i = 0;
    let eps = _mm256_set1_ps(f32::EPSILON);
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    while i + 8 <= n {
        let v = _mm256_loadu_ps(src.as_ptr().add(i));
        let s = _mm256_loadu_ps(scales.as_ptr().add(i));
        // inv = s > eps ? 1/s : 0
        let gt = _mm256_cmp_ps(s, eps, _CMP_GT_OS);
        let inv_raw = _mm256_div_ps(one, s);
        let inv = _mm256_blendv_ps(zero, inv_raw, gt);
        let scaled = _mm256_mul_ps(v, inv);
        let rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let qi = _mm256_cvtps_epi32(rounded);
        // Pack 8×i32 → 8×i8 with signed saturation.
        let lo = _mm256_castsi256_si128(qi);
        let hi = _mm256_extracti128_si256(qi, 1);
        let packed16 = _mm_packs_epi32(lo, hi);
        let packed8 = _mm_packs_epi16(packed16, packed16);
        let mut buf = [0i8; 16];
        _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, packed8);
        std::ptr::copy_nonoverlapping(buf.as_ptr(), dst.as_mut_ptr().add(i), 8);
        i += 8;
    }
    if i < n {
        quant_i8_row_scalar(&src[i..], &scales[i..], &mut dst[i..]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dequant_i8_row_avx2(src: &[i8], scales: &[f32], dst: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;

    let n = src.len();
    let mut i = 0;
    while i + 8 <= n {
        // Load 8 i8 → sign-extend to i32 → convert to f32.
        let mut buf = [0i8; 16];
        std::ptr::copy_nonoverlapping(src.as_ptr().add(i), buf.as_mut_ptr(), 8);
        let packed = _mm_loadu_si128(buf.as_ptr() as *const __m128i);
        let q16 = _mm_cvtepi8_epi16(packed);
        let q32 = _mm256_cvtepi16_epi32(q16);
        let f = _mm256_cvtepi32_ps(q32);
        let s = _mm256_loadu_ps(scales.as_ptr().add(i));
        let out = _mm256_mul_ps(f, s);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), out);
        i += 8;
    }
    if i < n {
        dequant_i8_row_scalar(&src[i..], &scales[i..], &mut dst[i..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn quant_i8_row_neon(src: &[f32], scales: &[f32], dst: &mut [i8]) {
    use std::arch::aarch64::*;
    let n = src.len();
    let mut i = 0;
    let eps = vdupq_n_f32(f32::EPSILON);
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    while i + 8 <= n {
        let v_lo = vld1q_f32(src.as_ptr().add(i));
        let v_hi = vld1q_f32(src.as_ptr().add(i + 4));
        let s_lo = vld1q_f32(scales.as_ptr().add(i));
        let s_hi = vld1q_f32(scales.as_ptr().add(i + 4));
        let gt_lo = vcgtq_f32(s_lo, eps);
        let gt_hi = vcgtq_f32(s_hi, eps);
        let inv_lo = vbslq_f32(gt_lo, vdivq_f32(one, s_lo), zero);
        let inv_hi = vbslq_f32(gt_hi, vdivq_f32(one, s_hi), zero);
        let r_lo = vrndnq_f32(vmulq_f32(v_lo, inv_lo));
        let r_hi = vrndnq_f32(vmulq_f32(v_hi, inv_hi));
        let i32_lo = vcvtq_s32_f32(r_lo);
        let i32_hi = vcvtq_s32_f32(r_hi);
        let i16x4_lo = vqmovn_s32(i32_lo);
        let i16x4_hi = vqmovn_s32(i32_hi);
        let i16x8 = vcombine_s16(i16x4_lo, i16x4_hi);
        let i8x8 = vqmovn_s16(i16x8);
        vst1_s8(dst.as_mut_ptr().add(i), i8x8);
        i += 8;
    }
    if i < n {
        quant_i8_row_scalar(&src[i..], &scales[i..], &mut dst[i..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dequant_i8_row_neon(src: &[i8], scales: &[f32], dst: &mut [f32]) {
    use std::arch::aarch64::*;
    let n = src.len();
    let mut i = 0;
    while i + 8 <= n {
        let q8 = vld1_s8(src.as_ptr().add(i));
        let q16 = vmovl_s8(q8);
        let lo32 = vmovl_s16(vget_low_s16(q16));
        let hi32 = vmovl_s16(vget_high_s16(q16));
        let lo_f = vcvtq_f32_s32(lo32);
        let hi_f = vcvtq_f32_s32(hi32);
        let s_lo = vld1q_f32(scales.as_ptr().add(i));
        let s_hi = vld1q_f32(scales.as_ptr().add(i + 4));
        vst1q_f32(dst.as_mut_ptr().add(i), vmulq_f32(lo_f, s_lo));
        vst1q_f32(dst.as_mut_ptr().add(i + 4), vmulq_f32(hi_f, s_hi));
        i += 8;
    }
    if i < n {
        dequant_i8_row_scalar(&src[i..], &scales[i..], &mut dst[i..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_dequant_row_roundtrip() {
        let src: Vec<f32> = (0..17).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let scales: Vec<f32> = vec![0.1; 17];
        let mut q = vec![0i8; 17];
        quant_i8_row(&src, &scales, &mut q);
        let mut back = vec![0f32; 17];
        dequant_i8_row(&q, &scales, &mut back);
        for (a, b) in src.iter().zip(&back) {
            assert!((a - b).abs() < 0.06, "{a} vs {b}");
        }
    }
}
