//! Vendor-neutral tensor descriptor.
//!
//! A descriptor is **just metadata** — it doesn't own storage. Backends
//! pair it with an [`Allocation`](crate::Allocation) to form a concrete
//! tensor. Stride is in **elements**, not bytes, so layout transforms work
//! across dtypes without re-deriving the byte stride each time.

use crate::dtype::DType;

#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Layout {
    /// Row-major (C order).
    RowMajor,
    /// Column-major (Fortran order).
    ColMajor,
    /// Channels-last (NHWC). 4-D and 5-D only.
    ChannelsLast,
    /// Blocked / tiled — backend-defined block shape.
    Blocked,
    /// Caller supplies explicit strides.
    Strided,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TensorDesc {
    pub dtype: DType,
    pub shape: Vec<u32>,
    /// Strides in **elements** (not bytes). `None` means dense + `layout`.
    pub strides: Option<Vec<i32>>,
    pub layout: Layout,
}

impl TensorDesc {
    pub fn dense(dtype: DType, shape: impl Into<Vec<u32>>) -> Self {
        Self { dtype, shape: shape.into(), strides: None, layout: Layout::RowMajor }
    }

    /// Number of elements (product of shape).
    pub fn numel(&self) -> u64 {
        self.shape.iter().map(|d| *d as u64).product()
    }

    /// Total byte footprint assuming dense packing (ignores strides).
    pub fn bytes(&self) -> u64 {
        let bits = self.dtype.bits() as u64;
        // Round up to byte boundary; quant-block (bits == 0) returns 0.
        if bits == 0 { 0 } else { (self.numel() * bits + 7) / 8 }
    }

    pub fn rank(&self) -> usize { self.shape.len() }

    /// Compute the dense row-major / col-major strides for this descriptor.
    pub fn dense_strides(&self) -> Vec<i32> {
        let r = self.rank();
        let mut s = vec![1i32; r];
        match self.layout {
            Layout::ColMajor => {
                for i in 1..r { s[i] = s[i - 1] * self.shape[i - 1] as i32; }
            }
            _ => {
                for i in (0..r.saturating_sub(1)).rev() {
                    s[i] = s[i + 1] * self.shape[i + 1] as i32;
                }
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    #[test]
    fn dense_strides_row_major() {
        let t = TensorDesc::dense(DType::F32, vec![2, 3, 4]);
        assert_eq!(t.dense_strides(), vec![12, 4, 1]);
    }

    #[test]
    fn bytes_for_bf16() {
        let t = TensorDesc::dense(DType::Bf16, vec![1024, 1024]);
        assert_eq!(t.bytes(), 2 * 1024 * 1024);
    }

    #[test]
    fn bytes_for_int4_rounds_up() {
        let t = TensorDesc::dense(DType::U4, vec![5]); // 5 nibbles = 20 bits → 3 bytes
        assert_eq!(t.bytes(), 3);
    }
}
