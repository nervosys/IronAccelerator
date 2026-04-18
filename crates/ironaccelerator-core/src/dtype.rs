//! Numeric data types. Mirrors the most common ML / HPC formats so that
//! backend kernels can be selected via dtype dispatch.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NumericClass {
    Float,
    Integer,
    Boolean,
    Quantized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DType {
    F64,
    F32,
    Tf32,
    F16,
    Bf16,
    F8E4M3,
    F8E5M2,
    F4,
    I64,
    I32,
    I16,
    I8,
    U8,
    U4,
    Bool,
    /// Block-quantized weights (e.g. Q4_K, Q8_0). The discriminator is
    /// stored in the workload metadata.
    QuantBlock,
}

impl DType {
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
