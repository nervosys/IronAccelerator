//! MPS-backed GEMM. Apple-only.
//!
//! Wraps `MPSMatrixMultiplication` into a single-call `sgemm`/`hgemm` helper.
//! The kernel accepts row-major matrices; the wrapper handles descriptor
//! creation + encoding + commit. For multi-call workloads, prefer holding
//! the `MatrixMultiplication` object and re-encoding against pre-built
//! `Matrix` views.

#![cfg(target_vendor = "apple")]

use crate::drv::{Buffer, Device, Queue};
use ironaccelerator_core::{Error, Result};
use metal::mps::{Matrix, MatrixDescriptor, MatrixMultiplication};
use metal::MPSDataType;
use std::sync::Arc;

/// Element type for a Metal GEMM.
#[derive(Debug, Copy, Clone)]
pub enum MetalDType {
    F32,
    F16,
}

impl MetalDType {
    fn mps(self) -> MPSDataType {
        match self {
            Self::F32 => MPSDataType::Float32,
            Self::F16 => MPSDataType::Float16,
        }
    }
    fn bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

/// Row-major `C = alpha · op(A) · op(B) + beta · C`.
///
/// All buffers must be `StorageModeShared` or `StorageModePrivate` and live
/// on the same device as `queue.device()`.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    queue: &Arc<Queue>,
    m: u32,
    n: u32,
    k: u32,
    alpha: f64,
    beta: f64,
    trans_a: bool,
    trans_b: bool,
    dtype: MetalDType,
    a: &Arc<Buffer>,
    b: &Arc<Buffer>,
    c: &Arc<Buffer>,
) -> Result<()> {
    let device = queue.device().raw();
    let dt = dtype.mps();
    let es = dtype.bytes();

    // Row bytes for row-major layout.
    let (a_rows, a_cols) = if trans_a { (k, m) } else { (m, k) };
    let (b_rows, b_cols) = if trans_b { (n, k) } else { (k, n) };
    let c_rows = m;
    let c_cols = n;

    let row_bytes = |cols: u32| (cols as u64) * es;
    if (a.bytes() as u64) < (a_rows as u64) * row_bytes(a_cols)
        || (b.bytes() as u64) < (b_rows as u64) * row_bytes(b_cols)
        || (c.bytes() as u64) < (c_rows as u64) * row_bytes(c_cols)
    {
        return Err(Error::InvalidArgument(
            "gemm: buffer too small for declared shape",
        ));
    }

    let desc_a = MatrixDescriptor::init_single(a_rows as u64, a_cols as u64, row_bytes(a_cols), dt);
    let desc_b = MatrixDescriptor::init_single(b_rows as u64, b_cols as u64, row_bytes(b_cols), dt);
    let desc_c = MatrixDescriptor::init_single(c_rows as u64, c_cols as u64, row_bytes(c_cols), dt);

    let mat_a = Matrix::init_with_buffer_descriptor(a.raw(), &desc_a)
        .ok_or(Error::Other("Metal: Matrix init failed for A"))?;
    let mat_b = Matrix::init_with_buffer_descriptor(b.raw(), &desc_b)
        .ok_or(Error::Other("Metal: Matrix init failed for B"))?;
    let mat_c = Matrix::init_with_buffer_descriptor(c.raw(), &desc_c)
        .ok_or(Error::Other("Metal: Matrix init failed for C"))?;

    let mm = MatrixMultiplication::init(
        device, trans_a, trans_b, m as u64, n as u64, k as u64, alpha, beta,
    )
    .ok_or(Error::Other("Metal: MPSMatrixMultiplication init failed"))?;

    queue.scope(|cb| {
        mm.encode_to_command_buffer(cb, &mat_a, &mat_b, &mat_c);
    });
    Ok(())
}

/// Convenience: build a plan once, reuse encoding against pre-built matrices.
pub struct GemmPlan {
    mm: MatrixMultiplication,
    _device: Arc<Device>,
}

unsafe impl Send for GemmPlan {}
unsafe impl Sync for GemmPlan {}

impl GemmPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Arc<Device>,
        m: u32,
        n: u32,
        k: u32,
        alpha: f64,
        beta: f64,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<Self> {
        let mm = MatrixMultiplication::init(
            device.raw(),
            trans_a,
            trans_b,
            m as u64,
            n as u64,
            k as u64,
            alpha,
            beta,
        )
        .ok_or(Error::Other("Metal: MPSMatrixMultiplication init failed"))?;
        Ok(Self {
            mm,
            _device: device,
        })
    }

    pub fn encode(&self, cb: &metal::CommandBufferRef, a: &Matrix, b: &Matrix, c: &Matrix) {
        self.mm.encode_to_command_buffer(cb, a, b, c);
    }
}
