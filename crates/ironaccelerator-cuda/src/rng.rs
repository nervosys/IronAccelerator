//! cuRAND safe wrapper.
//!
//! Thin RAII around a host (pseudorandom) generator. Exposes seed/offset/stream
//! setters and typed fill entry points for the f32/f64 variants that back
//! IronAccelerator's tensor init paths.

use crate::drv::{Device, DeviceBuf, Stream};
use iron_cuda_sys::curand as sys;
use ironaccelerator_core::{Error, Result};
use std::sync::Arc;

pub use sys::CurandRngType as RngType;

fn fns() -> Result<&'static sys::CurandFns> {
    sys::fns().map_err(|e| {
        Error::Other(Box::leak(
            format!("curand not available: {e}").into_boxed_str(),
        ))
    })
}

fn check(_op: &'static str, s: sys::CurandStatus) -> Result<()> {
    if s.is_ok() {
        Ok(())
    } else {
        Err(Error::Backend {
            backend: ironaccelerator_core::BackendKind::Cuda,
            code: (s as u32) as i64,
        })
    }
}

/// cuRAND generator bound to a stream. Destroyed on drop.
pub struct Rng {
    gen: sys::CurandGenerator,
}

unsafe impl Send for Rng {}
unsafe impl Sync for Rng {}

impl Rng {
    /// Create a Philox 4x32-10 generator seeded with `seed`, bound to `stream`.
    pub fn new(device: &Arc<Device>, stream: &Arc<Stream>, seed: u64) -> Result<Self> {
        Self::with_type(device, stream, seed, RngType::PseudoPhilox4_32_10)
    }

    /// Create a generator of type `ty` seeded with `seed`, bound to `stream`.
    /// Wraps `curandCreateGenerator`, `curandSetStream`, and
    /// `curandSetPseudoRandomGeneratorSeed`.
    pub fn with_type(
        device: &Arc<Device>,
        stream: &Arc<Stream>,
        seed: u64,
        ty: RngType,
    ) -> Result<Self> {
        device.bind()?;
        let f = fns()?;
        let mut g = sys::CurandGenerator::default();
        unsafe {
            check(
                "curandCreateGenerator",
                (f.curandCreateGenerator)(&mut g, ty),
            )?;
            check("curandSetStream", (f.curandSetStream)(g, stream.raw()))?;
            check(
                "curandSetPseudoRandomGeneratorSeed",
                (f.curandSetPseudoRandomGeneratorSeed)(g, seed),
            )?;
        }
        Ok(Self { gen: g })
    }

    /// Reseed the generator. Wraps `curandSetPseudoRandomGeneratorSeed`.
    pub fn set_seed(&mut self, seed: u64) -> Result<()> {
        unsafe {
            check(
                "curandSetPseudoRandomGeneratorSeed",
                (fns()?.curandSetPseudoRandomGeneratorSeed)(self.gen, seed),
            )
        }
    }

    /// Set the generator offset. Wraps `curandSetGeneratorOffset`.
    pub fn set_offset(&mut self, offset: u64) -> Result<()> {
        unsafe {
            check(
                "curandSetGeneratorOffset",
                (fns()?.curandSetGeneratorOffset)(self.gen, offset),
            )
        }
    }

    /// Fill `dst` with uniform `f32` samples. Wraps `curandGenerateUniform`.
    pub fn fill_uniform_f32(&mut self, dst: &mut DeviceBuf<f32>) -> Result<()> {
        unsafe {
            check(
                "curandGenerateUniform",
                (fns()?.curandGenerateUniform)(self.gen, dst.device_ptr(), dst.len()),
            )
        }
    }

    /// Fill `dst` with uniform `f64` samples. Wraps `curandGenerateUniformDouble`.
    pub fn fill_uniform_f64(&mut self, dst: &mut DeviceBuf<f64>) -> Result<()> {
        unsafe {
            check(
                "curandGenerateUniformDouble",
                (fns()?.curandGenerateUniformDouble)(self.gen, dst.device_ptr(), dst.len()),
            )
        }
    }

    /// Fill `dst` with normal `f32` samples of the given `mean` and `std`. Wraps `curandGenerateNormal`.
    pub fn fill_normal_f32(&mut self, dst: &mut DeviceBuf<f32>, mean: f32, std: f32) -> Result<()> {
        unsafe {
            check(
                "curandGenerateNormal",
                (fns()?.curandGenerateNormal)(self.gen, dst.device_ptr(), dst.len(), mean, std),
            )
        }
    }

    /// Fill `dst` with normal `f64` samples of the given `mean` and `std`. Wraps `curandGenerateNormalDouble`.
    pub fn fill_normal_f64(&mut self, dst: &mut DeviceBuf<f64>, mean: f64, std: f64) -> Result<()> {
        unsafe {
            check(
                "curandGenerateNormalDouble",
                (fns()?.curandGenerateNormalDouble)(
                    self.gen,
                    dst.device_ptr(),
                    dst.len(),
                    mean,
                    std,
                ),
            )
        }
    }

    /// Fill with raw 32-bit integers (useful for custom distributions).
    pub fn fill_u32(&mut self, dst: &mut DeviceBuf<u32>) -> Result<()> {
        unsafe {
            check(
                "curandGenerate",
                (fns()?.curandGenerate)(self.gen, dst.device_ptr(), dst.len()),
            )
        }
    }

    /// Fill with raw 64-bit integers.
    pub fn fill_u64(&mut self, dst: &mut DeviceBuf<u64>) -> Result<()> {
        unsafe {
            check(
                "curandGenerateLongLong",
                (fns()?.curandGenerateLongLong)(self.gen, dst.device_ptr(), dst.len()),
            )
        }
    }
}

impl Drop for Rng {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe {
                let _ = (f.curandDestroyGenerator)(self.gen);
            }
        }
    }
}
