//! FP8 GEMM launcher.
//!
//! Turns an [`Fp8Recipe`] + shape into a concrete cuBLASLt call. The recipe
//! is pre-validated; the builder configures:
//!
//! - per-operand [`CudaDataType`] (E4M3/E5M2 inputs, Bf16/Fp16/Fp32 output)
//! - compute type (FP32 accumulate, or FP32-fast-BF16 for [`AccumMode::Bf16Fast`])
//! - delayed-scaling pointers (A / B / D)
//! - optional output-amax pointer
//! - optional fused bias + epilogue
//! - optional fast-accumulate flag
//!
//! Static-scale and Vec16 (Blackwell MX) paths still route through here; the
//! difference is just which scale pointers the caller hands in.

use crate::blas::{self, BlasLt, MatmulDesc, MatrixLayout, Preference, ScaleTensor};
use crate::drv::{DeviceBuf, Stream};
use crate::fp8::{AccumMode, Fp8Dtype, Fp8Recipe};
use iron_cuda_sys::cublas_lt as sys;
use iron_cuda_sys::driver::CUdeviceptr;
use ironaccelerator_core::{DType, Error, Result};

/// Raw `CUBLASLT_MATMUL_DESC_FAST_ACCUM` (attr = 11).
const DESC_ATTR_FAST_ACCUM: u32 = 11;
/// Raw epilogue values (cuBLASLt `cublasLtEpilogue_t`).
const EPI_DEFAULT: u32 = 1;
const EPI_BIAS: u32 = 4;
const EPI_GELU: u32 = 32;
const EPI_GELU_BIAS: u32 = 36;
const EPI_RELU: u32 = 8;
const EPI_RELU_BIAS: u32 = 12;

#[derive(Copy, Clone, Debug, Default)]
pub struct Fp8Shape {
    pub m: u64, pub n: u64, pub k: u64,
    /// Leading dims; 0 means "packed" (lda=k for N-op A, etc.).
    pub lda: i64, pub ldb: i64, pub ldc: i64, pub ldd: i64,
    pub trans_a: bool, pub trans_b: bool,
}

impl Fp8Shape {
    pub fn square(m: u64, n: u64, k: u64) -> Self {
        Self { m, n, k, lda: 0, ldb: 0, ldc: 0, ldd: 0, trans_a: false, trans_b: false }
    }
}

/// Per-tensor delayed-scaling pointers. Each `*_scale` is a single-element
/// FP32 device buffer in Transformer-Engine convention.
#[derive(Copy, Clone, Debug, Default)]
pub struct Fp8Scales {
    pub a_scale: Option<CUdeviceptr>,
    pub b_scale: Option<CUdeviceptr>,
    pub d_scale: Option<CUdeviceptr>,
    pub amax_d:  Option<CUdeviceptr>,
}

/// Optional epilogue fusion (bias / activation).
#[derive(Copy, Clone, Debug, Default)]
pub enum Fp8Epilogue {
    #[default]
    None,
    Bias { ptr: CUdeviceptr },
    Gelu,
    GeluBias { ptr: CUdeviceptr },
    Relu,
    ReluBias { ptr: CUdeviceptr },
}

impl Fp8Epilogue {
    fn code(&self) -> u32 {
        match self {
            Self::None => EPI_DEFAULT,
            Self::Bias { .. } => EPI_BIAS,
            Self::Gelu => EPI_GELU,
            Self::GeluBias { .. } => EPI_GELU_BIAS,
            Self::Relu => EPI_RELU,
            Self::ReluBias { .. } => EPI_RELU_BIAS,
        }
    }
    fn bias_ptr(&self) -> Option<CUdeviceptr> {
        match self {
            Self::Bias { ptr } | Self::GeluBias { ptr } | Self::ReluBias { ptr } => Some(*ptr),
            _ => None,
        }
    }
}

fn fp8_input(d: Fp8Dtype) -> sys::CudaDataType {
    match d {
        Fp8Dtype::E4M3 => sys::CudaDataType::R8F_E4M3,
        Fp8Dtype::E5M2 => sys::CudaDataType::R8F_E5M2,
    }
}

fn output_dtype(d: DType) -> Result<sys::CudaDataType> {
    Ok(match d {
        DType::Bf16 => sys::CudaDataType::R16BF,
        DType::F16  => sys::CudaDataType::R16F,
        DType::F32  => sys::CudaDataType::R32F,
        _ => return Err(Error::Other("fp8 output dtype must be Bf16/Fp16/Fp32")),
    })
}

fn compute_type(accum: AccumMode) -> sys::CublasComputeType {
    match accum {
        AccumMode::Fp32     => sys::CublasComputeType::F32,
        AccumMode::Bf16Fast => sys::CublasComputeType::F32FastBf16,
    }
}

/// Prepared launch bundle. The descriptors are RAII-owned; `algo` caches the
/// heuristic result so repeated launches of the same shape avoid re-querying.
pub struct Fp8Gemm {
    pub desc: MatmulDesc,
    pub a: MatrixLayout,
    pub b: MatrixLayout,
    pub c: MatrixLayout,
    pub d: MatrixLayout,
    pub pref: Preference,
    pub algo: sys::CublasLtMatmulHeuristicResult,
    pub workspace_bytes: usize,
}

impl Fp8Gemm {
    /// Build all descriptors and run the heuristic. Scale / amax / bias
    /// *pointers* are baked into the descriptor — the actual device buffers
    /// must outlive the subsequent [`Self::launch`] call.
    pub fn build(
        blaslt: &BlasLt,
        recipe: &Fp8Recipe,
        shape: &Fp8Shape,
        scales: &Fp8Scales,
        epilogue: Fp8Epilogue,
        max_workspace: usize,
    ) -> Result<Self> {
        recipe.validate().map_err(Error::Other)?;

        let a_dtype = fp8_input(recipe.a);
        let b_dtype = fp8_input(recipe.b);
        let out_dtype = output_dtype(recipe.out)?;
        let compute = compute_type(recipe.accum);

        // Descriptor (compute type + FP32 scale pointers).
        let mut desc = MatmulDesc::new(compute, sys::CudaDataType::R32F)?;
        desc.set_transpose(
            if shape.trans_a { blas::Op::T } else { blas::Op::N },
            if shape.trans_b { blas::Op::T } else { blas::Op::N },
        )?;

        if let Some(p) = scales.a_scale { desc.set_scale_pointer(ScaleTensor::A, p)?; }
        if let Some(p) = scales.b_scale { desc.set_scale_pointer(ScaleTensor::B, p)?; }
        if let Some(p) = scales.d_scale { desc.set_scale_pointer(ScaleTensor::D, p)?; }
        if recipe.produce_output_amax {
            if let Some(p) = scales.amax_d { desc.set_amax_d_pointer(p)?; }
        }

        desc.set_epilogue_raw(epilogue.code())?;
        if let Some(p) = epilogue.bias_ptr() { desc.set_bias_pointer(p)?; }

        if matches!(recipe.accum, AccumMode::Bf16Fast) {
            unsafe { desc.set_attr_raw_u32(DESC_ATTR_FAST_ACCUM, 1)?; }
        }

        // Layouts — column-major is cuBLASLt's default. Leading dim defaults
        // collapse to the packed stride for the chosen transpose.
        let (rows_a, cols_a) = if shape.trans_a { (shape.k, shape.m) } else { (shape.m, shape.k) };
        let (rows_b, cols_b) = if shape.trans_b { (shape.n, shape.k) } else { (shape.k, shape.n) };
        let lda = if shape.lda != 0 { shape.lda } else { rows_a as i64 };
        let ldb = if shape.ldb != 0 { shape.ldb } else { rows_b as i64 };
        let ldc = if shape.ldc != 0 { shape.ldc } else { shape.m as i64 };
        let ldd = if shape.ldd != 0 { shape.ldd } else { shape.m as i64 };

        let a = MatrixLayout::new(a_dtype,   rows_a, cols_a, lda)?;
        let b = MatrixLayout::new(b_dtype,   rows_b, cols_b, ldb)?;
        let c = MatrixLayout::new(out_dtype, shape.m, shape.n, ldc)?;
        let d = MatrixLayout::new(out_dtype, shape.m, shape.n, ldd)?;

        let mut pref = Preference::new()?;
        pref.set_max_workspace(max_workspace)?;

        let algo = blas::heuristic(blaslt, &desc, &a, &b, &c, &d, &pref)?;
        let workspace_bytes = algo.workspace_size;

        Ok(Self { desc, a, b, c, d, pref, algo, workspace_bytes })
    }

    /// Launch with the heuristic-selected algorithm.
    ///
    /// # Safety
    /// Device pointers must match the layouts and scale-pointer attributes
    /// baked in during [`Self::build`].
    pub unsafe fn launch(
        &self,
        blaslt: &BlasLt,
        a_ptr: CUdeviceptr, b_ptr: CUdeviceptr,
        c_ptr: CUdeviceptr, d_ptr: CUdeviceptr,
        alpha: f32, beta: f32,
        workspace: Option<&mut DeviceBuf<u8>>,
        stream: &Stream,
    ) -> Result<()> {
        let alpha_le = alpha.to_ne_bytes();
        let beta_le  = beta.to_ne_bytes();
        blas::matmul(
            blaslt, &self.desc,
            &alpha_le, &beta_le,
            a_ptr, &self.a, b_ptr, &self.b,
            c_ptr, &self.c, d_ptr, &self.d,
            Some(&self.algo),
            workspace,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp8::Fp8Recipe;

    #[test]
    fn epilogue_codes_are_stable() {
        assert_eq!(Fp8Epilogue::None.code(), 1);
        assert_eq!(Fp8Epilogue::Gelu.code(), 32);
        assert!(Fp8Epilogue::Gelu.bias_ptr().is_none());
    }

    #[test]
    fn compute_type_matches_accum_mode() {
        assert!(matches!(compute_type(AccumMode::Fp32), sys::CublasComputeType::F32));
        assert!(matches!(compute_type(AccumMode::Bf16Fast), sys::CublasComputeType::F32FastBf16));
    }

    #[test]
    fn output_dtype_rejects_invalid() {
        assert!(output_dtype(DType::I32).is_err());
        assert!(output_dtype(DType::Bf16).is_ok());
    }

    #[test]
    fn input_dtype_maps_fp8_formats() {
        let r = Fp8Recipe::hopper_default();
        assert!(matches!(fp8_input(r.a), sys::CudaDataType::R8F_E4M3));
        let r2 = Fp8Recipe::hopper_backward();
        assert!(matches!(fp8_input(r2.a), sys::CudaDataType::R8F_E5M2));
    }
}
