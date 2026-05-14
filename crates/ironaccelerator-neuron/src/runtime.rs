//! Thin wrappers over `nrt_load` / `nrt_execute`. The AWS Neuron
//! Runtime takes a NEFF binary (produced by the Neuron compiler) plus
//! two tensor sets (inputs + outputs) and runs it on one or more
//! NeuronCores. We don't ship the compiler — we expect the caller to
//! have a `.neff` already.

use core::ffi::{c_void, CStr};

use crate::drv::{self, Loaded, NrtModelHandle, NrtTensorSetHandle, NRT_SUCCESS};

#[derive(Debug)]
pub enum NeuronError {
    Unavailable,
    MissingSymbol(&'static str),
    Runtime(u32),
    BadName,
}

/// Compiled Neuron model, loaded from a NEFF binary onto one or more
/// NeuronCores.
pub struct Model {
    l: &'static Loaded,
    handle: NrtModelHandle,
}

impl Model {
    /// Load `neff` onto NeuronCores `[start_nc, start_nc + nc_count)`.
    /// `nc_count = 1` is the typical case for a single-core inference
    /// model; multi-core is used for tensor-parallel Trn1/Trn2 graphs.
    pub fn load(neff: &[u8], start_nc: i32, nc_count: i32) -> Result<Self, NeuronError> {
        let l = drv::loaded().ok_or(NeuronError::Unavailable)?;
        let load = l.nrt_load.ok_or(NeuronError::MissingSymbol("nrt_load"))?;
        let mut handle: NrtModelHandle = core::ptr::null_mut();
        let r = unsafe { load(neff.as_ptr(), neff.len(), start_nc, nc_count, &mut handle) };
        if r != NRT_SUCCESS {
            return Err(NeuronError::Runtime(r));
        }
        Ok(Self { l, handle })
    }

    /// Execute this model with pre-populated input / output tensor
    /// sets. Blocks until the NeuronCore signals completion.
    pub fn execute(&self, inputs: &TensorSet, outputs: &TensorSet) -> Result<(), NeuronError> {
        let exec = self
            .l
            .nrt_execute
            .ok_or(NeuronError::MissingSymbol("nrt_execute"))?;
        let r = unsafe { exec(self.handle, inputs.handle, outputs.handle) };
        if r != NRT_SUCCESS {
            return Err(NeuronError::Runtime(r));
        }
        Ok(())
    }

    pub fn raw(&self) -> NrtModelHandle {
        self.handle
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if let Some(unload) = self.l.nrt_unload {
            unsafe {
                let _ = unload(self.handle);
            }
        }
    }
}

/// A named collection of tensors — the input or output side of an
/// execute call.
pub struct TensorSet {
    l: &'static Loaded,
    handle: NrtTensorSetHandle,
}

impl TensorSet {
    pub fn new() -> Result<Self, NeuronError> {
        let l = drv::loaded().ok_or(NeuronError::Unavailable)?;
        let alloc = l
            .nrt_alloc_tensor_set
            .ok_or(NeuronError::MissingSymbol("nrt_allocate_tensor_set"))?;
        let mut handle: NrtTensorSetHandle = core::ptr::null_mut();
        let r = unsafe { alloc(&mut handle) };
        if r != NRT_SUCCESS {
            return Err(NeuronError::Runtime(r));
        }
        Ok(Self { l, handle })
    }

    /// Add a pre-allocated `nrt_tensor_t` to this set under `name`.
    /// Tensor lifetime is managed by the caller (or by the Neuron
    /// runtime via `nrt_tensor_allocate`, which we don't wrap yet).
    /// # Safety
    /// `tensor` must point to a valid `nrt_tensor_t` that outlives this set.
    pub unsafe fn add(&mut self, name: &CStr, tensor: *mut c_void) -> Result<(), NeuronError> {
        let add = self
            .l
            .nrt_add_tensor_to_set
            .ok_or(NeuronError::MissingSymbol("nrt_add_tensor_to_tensor_set"))?;
        let r = unsafe { add(self.handle, name.as_ptr(), tensor) };
        if r != NRT_SUCCESS {
            return Err(NeuronError::Runtime(r));
        }
        Ok(())
    }

    pub fn raw(&self) -> NrtTensorSetHandle {
        self.handle
    }
}

impl Drop for TensorSet {
    fn drop(&mut self) {
        if let Some(free) = self.l.nrt_free_tensor_set {
            unsafe {
                let _ = free(self.handle);
            }
        }
    }
}
