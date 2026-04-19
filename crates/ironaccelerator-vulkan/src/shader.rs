//! WGSL → SPIR-V compilation via `naga`. Lets the Vulkan backend share
//! one kernel source with the WebGPU backend.

use naga::back::spv;
use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};

#[derive(Debug)]
pub enum ShaderError {
    Parse(String),
    Validate(String),
    Emit(String),
}

impl core::fmt::Display for ShaderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ShaderError::Parse(s) => write!(f, "wgsl parse: {s}"),
            ShaderError::Validate(s) => write!(f, "wgsl validate: {s}"),
            ShaderError::Emit(s) => write!(f, "spv-out emit: {s}"),
        }
    }
}

impl std::error::Error for ShaderError {}

/// Compile a WGSL source string into a little-endian SPIR-V `u32`
/// stream suitable for `vkCreateShaderModule`.
pub fn wgsl_to_spirv(src: &str) -> Result<Vec<u32>, ShaderError> {
    let module = wgsl::parse_str(src).map_err(|e| ShaderError::Parse(e.to_string()))?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map_err(|e| ShaderError::Validate(e.to_string()))?;
    let opts = spv::Options::default();
    let words = spv::write_vec(&module, &info, &opts, None)
        .map_err(|e| ShaderError::Emit(e.to_string()))?;
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saxpy_wgsl_compiles_to_spirv() {
        let spv = wgsl_to_spirv(crate::kernels::SAXPY_WGSL).expect("wgsl→spv");
        // SPIR-V magic word is 0x07230203 little-endian.
        assert_eq!(spv[0], 0x07230203, "missing SPIR-V magic header");
        assert!(spv.len() > 10);
    }

    #[test]
    fn gemm_wgsl_compiles_to_spirv() {
        let spv = wgsl_to_spirv(crate::kernels::GEMM_F32_WGSL).expect("gemm wgsl→spv");
        assert_eq!(spv[0], 0x07230203);
        assert!(spv.len() > 20);
    }
}
