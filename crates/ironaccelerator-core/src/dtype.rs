//! Numeric data types. Mirrors the most common ML / HPC formats so that
//! backend kernels can be selected via dtype dispatch.

/// Broad category of a `DType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NumericClass {
    /// Floating-point format.
    Float,
    /// Signed or unsigned integer format.
    Integer,
    /// Single-bit boolean.
    Boolean,
    /// Block-quantized format (see `DType::QuantBlock`).
    Quantized,
}

/// Element data type of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DType {
    /// 64-bit IEEE floating point.
    F64,
    /// 32-bit IEEE floating point.
    F32,
    /// TensorFloat-32: 32-bit container with a 10-bit mantissa.
    Tf32,
    /// 16-bit IEEE half-precision floating point.
    F16,
    /// 16-bit bfloat: 8 exponent bits and 7 mantissa bits.
    Bf16,
    /// 8-bit float with 4 exponent and 3 mantissa bits.
    F8E4M3,
    /// 8-bit float with 5 exponent and 2 mantissa bits.
    F8E5M2,
    /// 4-bit floating point.
    F4,
    /// 64-bit signed integer.
    I64,
    /// 32-bit signed integer.
    I32,
    /// 16-bit signed integer.
    I16,
    /// 8-bit signed integer.
    I8,
    /// 8-bit unsigned integer.
    U8,
    /// 4-bit unsigned integer.
    U4,
    /// Boolean, one bit per element.
    Bool,
    /// Block-quantized weights (e.g. Q4_K, Q8_0). The discriminator is
    /// stored in the workload metadata.
    QuantBlock,
}

impl DType {
    /// Broad category (float, integer, boolean or quantized) of this type.
    pub const fn class(self) -> NumericClass {
        use DType::*;
        match self {
            F64 | F32 | Tf32 | F16 | Bf16 | F8E4M3 | F8E5M2 | F4 => NumericClass::Float,
            I64 | I32 | I16 | I8 | U8 | U4 => NumericClass::Integer,
            Bool => NumericClass::Boolean,
            QuantBlock => NumericClass::Quantized,
        }
    }

    /// Element size in bits. `0` for variable-block types.
    pub const fn bits(self) -> u32 {
        use DType::*;
        match self {
            F64 | I64 => 64,
            F32 | Tf32 | I32 => 32,
            F16 | Bf16 | I16 => 16,
            F8E4M3 | F8E5M2 | I8 | U8 => 8,
            F4 | U4 => 4,
            Bool => 1,
            QuantBlock => 0,
        }
    }
}
