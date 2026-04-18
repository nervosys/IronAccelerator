//! cuRAND safe wrapper.
//!
//! Thin RAII around a host (pseudorandom) generator. Exposes seed/offset/stream
//! setters and typed fill entry points for the f32/f64 variants that back
//! IronAccelerator's tensor init paths.

use crate::drv::DeviceBuf;
use crate::Session;
use iron_cuda_sys::curand as sys;
use ironaccelerator_core::{Error, Result};

pub use sys::CurandRngType as RngType;

fn fns() -> Result<&'static sys::CurandFns> {
    sys::fns().map_err(|e| Error::Other(Box::leak(
        format!("curand not available: {e}").into_boxed_str())))
}

fn check(_op: &'static str, s: sys::CurandStatus) -> Result<()> {
    if s.is_ok() { Ok(()) } else {
        Err(Error::Backend {
            backend: ironaccelerator_core::BackendKind::Cuda,
            code: (s as u32) as i64,
        })
    }
}

pub struct Rng {
    gen: sys::CurandGenerator,
}

unsafe impl Send for Rng {}
unsafe impl Sync for Rng {}

impl Rng {
    pub fn new(session: &Session, seed: u64) -> Result<Self> {
        Self::with_type(session, seed, RngType::PseudoPhilox4_32_10)
    }

    pub fn with_type(session: &Session, seed: u64, ty: RngType) -> Result<Self> {
        session.device().bind()?;
        let f = fns()?;
        let mut g = sys::CurandGenerator::default();
        unsafe {
            check("curandCreateGenerator", (f.curandCreateGenerator)(&mut g, ty))?;
            check("curandSetStream", (f.curandSetStream)(g, session.stream().raw()))?;
            check("curandSetPseudoRandomGeneratorSeed",
                  (f.curandSetPseudoRandomGeneratorSeed)(g, seed))?;
        }
        Ok(Self { gen: g })
    }

    pub fn set_seed(&mut self, seed: u64) -> Result<()> {
        unsafe {
            check("curandSetPseudoRandomGeneratorSeed",
                  (fns()?.curandSetPseudoRandomGeneratorSeed)(self.gen, seed))
        }
    }

    pub fn set_offset(&mut self, offset: u64) -> Result<()> {
        unsafe {
            check("curandSetGeneratorOffset",
                  (fns()?.curandSetGeneratorOffset)(self.gen, offset))
        }
    }

    pub fn fill_uniform_f32(&mut self, dst: &mut DeviceBuf<f32>) -> Result<()> {
        unsafe {
            check("curandGenerateUniform",
                  (fns()?.curandGenerateUniform)(self.gen, dst.device_ptr(), dst.len()))
        }
    }

    pub fn fill_uniform_f64(&mut self, dst: &mut DeviceBuf<f64>) -> Result<()> {
        unsafe {
            check("curandGenerateUniformDouble",
                  (fns()?.curandGenerateUniformDouble)(self.gen, dst.device_ptr(), dst.len()))
        }
    }

    pub fn fill_normal_f32(&mut self, dst: &mut DeviceBuf<f32>, mean: f32, std: f32) -> Result<()> {
        unsafe {
            check("curandGenerateNormal",
                  (fns()?.curandGenerateNormal)(self.gen, dst.device_ptr(), dst.len(), mean, std))
        }
    }

    pub fn fill_normal_f64(&mut self, dst: &mut DeviceBuf<f64>, mean: f64, std: f64) -> Result<()> {
        unsafe {
            check("curandGenerateNormalDouble",
                  (fns()?.curandGenerateNormalDouble)(self.gen, dst.device_ptr(), dst.len(), mean, std))
        }
    }

    /// Fill with raw 32-bit integers (useful for custom distributions).
    pub fn fill_u32(&mut self, dst: &mut DeviceBuf<u32>) -> Result<()> {
        unsafe {
            check("curandGenerate",
                  (fns()?.curandGenerate)(self.gen, dst.device_ptr(), dst.len()))
        }
    }

    /// Fill with raw 64-bit integers.
    pub fn fill_u64(&mut self, dst: &mut DeviceBuf<u64>) -> Result<()> {
        unsafe {
            check("curandGenerateLongLong",
                  (fns()?.curandGenerateLongLong)(self.gen, dst.device_ptr(), dst.len()))
        }
    }
}

impl Drop for Rng {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe { let _ = (f.curandDestroyGenerator)(self.gen); }
        }
    }
}
