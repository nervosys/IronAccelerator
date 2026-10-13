//! cuFFT safe wrapper.
//!
//! `FftPlan` is an RAII handle around a cuFFT plan; `FftPlanCache` is an
//! optional process-wide cache keyed on shape+type+batch+stream.

use crate::drv::{DeviceBuf, Repr, Stream};
use iron_cuda_sys::cufft as sys;
use ironaccelerator_core::{Error, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub use sys::{CufftType as FftType, CUFFT_FORWARD, CUFFT_INVERSE};

fn fns() -> Result<&'static sys::CufftFns> {
    sys::fns().map_err(|e| {
        Error::Other(Box::leak(
            format!("cufft not available: {e}").into_boxed_str(),
        ))
    })
}

fn check(_op: &'static str, s: sys::CufftResult) -> Result<()> {
    if s.is_ok() {
        Ok(())
    } else {
        Err(Error::Backend {
            backend: ironaccelerator_core::BackendKind::Cuda,
            code: s.raw() as i64,
        })
    }
}

// ─── Plan ───────────────────────────────────────────────────────────────────

/// RAII cuFFT plan (`cufftHandle`) bound to a stream; destroyed on drop.
pub struct FftPlan {
    handle: sys::CufftHandle,
    ty: FftType,
    _stream: Arc<Stream>,
}

unsafe impl Send for FftPlan {}
unsafe impl Sync for FftPlan {}

impl FftPlan {
    /// Creates a 1D plan of `nx` points with `batch` transforms (`cufftPlan1d`) and binds `stream`.
    pub fn plan_1d(stream: Arc<Stream>, nx: i32, ty: FftType, batch: i32) -> Result<Self> {
        let f = fns()?;
        let mut h = sys::CufftHandle::default();
        unsafe {
            check("cufftPlan1d", (f.cufftPlan1d)(&mut h, nx, ty, batch))?;
            check("cufftSetStream", (f.cufftSetStream)(h, stream.raw()))?;
        }
        Ok(Self {
            handle: h,
            ty,
            _stream: stream,
        })
    }

    /// Creates a plan for `rank`-dimensional transforms over `dims` with `batch` transforms (`cufftPlanMany`, contiguous layout) and binds `stream`.
    pub fn plan_many(
        stream: Arc<Stream>,
        rank: i32,
        dims: &[i32],
        ty: FftType,
        batch: i32,
    ) -> Result<Self> {
        let f = fns()?;
        let mut h = sys::CufftHandle::default();
        unsafe {
            check(
                "cufftPlanMany",
                (f.cufftPlanMany)(
                    &mut h,
                    rank,
                    dims.as_ptr(),
                    std::ptr::null(),
                    1,
                    0,
                    std::ptr::null(),
                    1,
                    0,
                    ty,
                    batch,
                ),
            )?;
            check("cufftSetStream", (f.cufftSetStream)(h, stream.raw()))?;
        }
        Ok(Self {
            handle: h,
            ty,
            _stream: stream,
        })
    }

    /// Creates a 2D plan of `nx` by `ny` points, via `plan_many`.
    pub fn plan_2d(stream: Arc<Stream>, nx: i32, ny: i32, ty: FftType) -> Result<Self> {
        Self::plan_many(stream, 2, &[nx, ny], ty, 1)
    }
    /// Creates a 3D plan of `nx` by `ny` by `nz` points, via `plan_many`.
    pub fn plan_3d(stream: Arc<Stream>, nx: i32, ny: i32, nz: i32, ty: FftType) -> Result<Self> {
        Self::plan_many(stream, 3, &[nx, ny, nz], ty, 1)
    }

    /// Returns the transform type this plan was created with.
    #[inline]
    pub fn ty(&self) -> FftType {
        self.ty
    }
    /// Returns the underlying `cufftHandle`.
    #[inline]
    pub fn raw(&self) -> sys::CufftHandle {
        self.handle
    }

    /// Execute C2C / Z2Z. `direction` is [`CUFFT_FORWARD`] or [`CUFFT_INVERSE`].
    pub fn exec_c2c<T: Repr>(
        &self,
        input: &DeviceBuf<T>,
        output: &mut DeviceBuf<T>,
        direction: i32,
    ) -> Result<()> {
        let f = fns()?;
        unsafe {
            let s = match self.ty {
                FftType::C2C => (f.cufftExecC2C)(
                    self.handle,
                    input.device_ptr(),
                    output.device_ptr(),
                    direction,
                ),
                FftType::Z2Z => (f.cufftExecZ2Z)(
                    self.handle,
                    input.device_ptr(),
                    output.device_ptr(),
                    direction,
                ),
                _ => return Err(Error::Other("exec_c2c: plan is not C2C/Z2Z")),
            };
            check("cufftExecC2C/Z2Z", s)
        }
    }

    /// Execute R2C (`cufftExecR2C`). Errors if the plan is not R2C.
    pub fn exec_r2c<T: Repr, U: Repr>(
        &self,
        input: &DeviceBuf<T>,
        output: &mut DeviceBuf<U>,
    ) -> Result<()> {
        let f = fns()?;
        unsafe {
            let s = match self.ty {
                FftType::R2C => {
                    (f.cufftExecR2C)(self.handle, input.device_ptr(), output.device_ptr())
                }
                _ => return Err(Error::Other("exec_r2c: plan is not R2C")),
            };
            check("cufftExecR2C", s)
        }
    }

    /// Execute C2R (`cufftExecC2R`). Errors if the plan is not C2R.
    pub fn exec_c2r<T: Repr, U: Repr>(
        &self,
        input: &DeviceBuf<T>,
        output: &mut DeviceBuf<U>,
    ) -> Result<()> {
        let f = fns()?;
        unsafe {
            let s = match self.ty {
                FftType::C2R => {
                    (f.cufftExecC2R)(self.handle, input.device_ptr(), output.device_ptr())
                }
                _ => return Err(Error::Other("exec_c2r: plan is not C2R")),
            };
            check("cufftExecC2R", s)
        }
    }
}

impl Drop for FftPlan {
    fn drop(&mut self) {
        if let Ok(f) = fns() {
            unsafe {
                let _ = (f.cufftDestroy)(self.handle);
            }
        }
    }
}

// ─── Plan cache ─────────────────────────────────────────────────────────────

/// Cache key identifying a cuFFT plan by shape, transform type, batch count and stream.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PlanKey {
    /// Extents per dimension; unused trailing entries are zero.
    pub dims: [i32; 3],
    /// Number of transform dimensions (1 to 3).
    pub rank: u8,
    /// Transform type, as the raw `cufftType` value.
    pub ty: u32,
    /// Number of batched transforms.
    pub batch: i32,
    /// Raw stream handle value, used as an identity key.
    pub stream: usize,
}

/// Process-wide cache of cuFFT plans keyed by [`PlanKey`].
pub struct FftPlanCache {
    plans: RwLock<HashMap<PlanKey, Arc<FftPlan>>>,
}

impl Default for FftPlanCache {
    fn default() -> Self {
        Self {
            plans: RwLock::new(HashMap::new()),
        }
    }
}

impl FftPlanCache {
    /// Creates an empty cache.
    pub fn new() -> Self {
        Self::default()
    }
    /// Returns the number of cached plans.
    pub fn len(&self) -> usize {
        self.plans.read().len()
    }
    /// Returns `true` if no plans are cached.
    pub fn is_empty(&self) -> bool {
        self.plans.read().is_empty()
    }
    /// Drops all cached plans.
    pub fn clear(&self) {
        self.plans.write().clear();
    }

    fn get_or_insert<F>(&self, key: PlanKey, build: F) -> Result<Arc<FftPlan>>
    where
        F: FnOnce() -> Result<FftPlan>,
    {
        if let Some(p) = self.plans.read().get(&key) {
            return Ok(p.clone());
        }
        let plan = Arc::new(build()?);
        self.plans.write().insert(key, plan.clone());
        Ok(plan)
    }

    /// Returns the cached 1D plan for these parameters, creating it on a miss.
    pub fn plan_1d(
        &self,
        stream: Arc<Stream>,
        nx: i32,
        ty: FftType,
        batch: i32,
    ) -> Result<Arc<FftPlan>> {
        let key = PlanKey {
            dims: [nx, 0, 0],
            rank: 1,
            ty: ty as u32,
            batch,
            stream: stream.raw().0 as usize,
        };
        self.get_or_insert(key, || FftPlan::plan_1d(stream, nx, ty, batch))
    }

    /// Returns the cached 2D plan for these parameters, creating it on a miss.
    pub fn plan_2d(
        &self,
        stream: Arc<Stream>,
        nx: i32,
        ny: i32,
        ty: FftType,
    ) -> Result<Arc<FftPlan>> {
        let key = PlanKey {
            dims: [nx, ny, 0],
            rank: 2,
            ty: ty as u32,
            batch: 1,
            stream: stream.raw().0 as usize,
        };
        self.get_or_insert(key, || FftPlan::plan_2d(stream, nx, ny, ty))
    }

    /// Returns the cached 3D plan for these parameters, creating it on a miss.
    pub fn plan_3d(
        &self,
        stream: Arc<Stream>,
        nx: i32,
        ny: i32,
        nz: i32,
        ty: FftType,
    ) -> Result<Arc<FftPlan>> {
        let key = PlanKey {
            dims: [nx, ny, nz],
            rank: 3,
            ty: ty as u32,
            batch: 1,
            stream: stream.raw().0 as usize,
        };
        self.get_or_insert(key, || FftPlan::plan_3d(stream, nx, ny, nz, ty))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plan_key_is_hashable_and_distinct() {
        use std::collections::HashSet;
        let mut s = HashSet::new();
        s.insert(PlanKey {
            dims: [256, 0, 0],
            rank: 1,
            ty: 41,
            batch: 1,
            stream: 0x1,
        });
        s.insert(PlanKey {
            dims: [256, 0, 0],
            rank: 1,
            ty: 41,
            batch: 2,
            stream: 0x1,
        });
        s.insert(PlanKey {
            dims: [256, 0, 0],
            rank: 1,
            ty: 41,
            batch: 1,
            stream: 0x1,
        });
        assert_eq!(s.len(), 2);
    }
}
