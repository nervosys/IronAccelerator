//! Opaque handle wrappers for vendor SDK objects.
//!
//! Backends embed their raw cuBLAS / rocBLAS / MPS / QNN handles in these
//! transparent newtypes so that downstream code never accidentally crosses
//! backend boundaries with a raw pointer.

use core::marker::PhantomData;
use core::ptr::NonNull;

#[repr(transparent)]
pub struct Handle<Tag> {
    raw: NonNull<()>,
    _t: PhantomData<fn() -> Tag>,
}

impl<Tag> Handle<Tag> {
    /// # Safety
    /// `ptr` must be a valid handle of the right vendor type.
    #[inline(always)]
    pub const unsafe fn from_raw(ptr: NonNull<()>) -> Self {
        Self { raw: ptr, _t: PhantomData }
    }

    #[inline(always)]
    pub const fn as_ptr(&self) -> NonNull<()> {
        self.raw
    }
}

unsafe impl<Tag: Send> Send for Handle<Tag> {}
unsafe impl<Tag: Sync> Sync for Handle<Tag> {}
