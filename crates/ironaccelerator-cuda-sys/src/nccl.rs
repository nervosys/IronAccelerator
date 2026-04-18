//! NCCL — GPU collective communication (all-reduce, all-gather, broadcast).
//!
//! Target: NCCL 2.23+ (what ships alongside CUDA 13.2).

use crate::driver::CUstream;
use crate::loader::{sym, try_load, LoadError};
use libloading::Library;
use std::ffi::{c_int, c_void};
use std::sync::{LazyLock, OnceLock};

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct NcclComm(pub *mut c_void);
unsafe impl Send for NcclComm {}
unsafe impl Sync for NcclComm {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct NcclUniqueId { pub internal: [u8; 128] }

impl Default for NcclUniqueId {
    fn default() -> Self { Self { internal: [0; 128] } }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclResult {
    Success = 0, UnhandledCudaError = 1, SystemError = 2, InternalError = 3,
    InvalidArgument = 4, InvalidUsage = 5, RemoteError = 6, InProgress = 7,
    Other = 0xFFFF_FFFF,
}

impl NcclResult {
    pub fn from_raw(r: u32) -> Self {
        if r <= 7 { unsafe { std::mem::transmute(r) } } else { Self::Other }
    }
    pub fn ok(self) -> Result<(), Self> {
        if self == Self::Success { Ok(()) } else { Err(self) }
    }
    pub fn is_ok(self) -> bool { self == Self::Success }
}

#[repr(u32)] #[derive(Copy, Clone, Debug)] pub enum NcclDataType {
    Int8 = 0, Uint8 = 1, Int32 = 2, Uint32 = 3, Int64 = 4, Uint64 = 5,
    Float16 = 6, Float32 = 7, Float64 = 8, Bfloat16 = 9,
    Fp8E4M3 = 10, Fp8E5M2 = 11,
}

#[repr(u32)] #[derive(Copy, Clone, Debug)] pub enum NcclRedOp {
    Sum = 0, Prod = 1, Max = 2, Min = 3, Avg = 4,
}

pub struct NcclFns {
    pub ncclGetVersion: unsafe extern "C" fn(*mut c_int) -> NcclResult,
    pub ncclGetUniqueId: unsafe extern "C" fn(*mut NcclUniqueId) -> NcclResult,
    pub ncclCommInitRank: unsafe extern "C" fn(
        *mut NcclComm, c_int, NcclUniqueId, c_int,
    ) -> NcclResult,
    pub ncclCommDestroy: unsafe extern "C" fn(NcclComm) -> NcclResult,
    pub ncclCommAbort: unsafe extern "C" fn(NcclComm) -> NcclResult,
    pub ncclCommCount: unsafe extern "C" fn(NcclComm, *mut c_int) -> NcclResult,
    pub ncclCommUserRank: unsafe extern "C" fn(NcclComm, *mut c_int) -> NcclResult,
    pub ncclGetErrorString: unsafe extern "C" fn(NcclResult) -> *const i8,

    pub ncclAllReduce: unsafe extern "C" fn(
        *const c_void, *mut c_void, usize, NcclDataType, NcclRedOp, NcclComm, CUstream,
    ) -> NcclResult,
    pub ncclAllGather: unsafe extern "C" fn(
        *const c_void, *mut c_void, usize, NcclDataType, NcclComm, CUstream,
    ) -> NcclResult,
    pub ncclBroadcast: unsafe extern "C" fn(
        *const c_void, *mut c_void, usize, NcclDataType, c_int, NcclComm, CUstream,
    ) -> NcclResult,
    pub ncclReduce: unsafe extern "C" fn(
        *const c_void, *mut c_void, usize, NcclDataType, NcclRedOp, c_int, NcclComm, CUstream,
    ) -> NcclResult,
    pub ncclReduceScatter: unsafe extern "C" fn(
        *const c_void, *mut c_void, usize, NcclDataType, NcclRedOp, NcclComm, CUstream,
    ) -> NcclResult,

    pub ncclGroupStart: unsafe extern "C" fn() -> NcclResult,
    pub ncclGroupEnd: unsafe extern "C" fn() -> NcclResult,
}

fn candidates() -> &'static [&'static str] {
    &["libnccl.so.2", "libnccl.so", "nccl.dll"]
}

static LIB: OnceLock<Library> = OnceLock::new();
static FNS: LazyLock<Result<NcclFns, LoadError>> = LazyLock::new(|| {
    let lib = try_load(candidates())?;
    let lib = LIB.get_or_init(|| lib);
    unsafe {
        Ok(NcclFns {
            ncclGetVersion: sym(lib, "nccl", "ncclGetVersion")?,
            ncclGetUniqueId: sym(lib, "nccl", "ncclGetUniqueId")?,
            ncclCommInitRank: sym(lib, "nccl", "ncclCommInitRank")?,
            ncclCommDestroy: sym(lib, "nccl", "ncclCommDestroy")?,
            ncclCommAbort: sym(lib, "nccl", "ncclCommAbort")?,
            ncclCommCount: sym(lib, "nccl", "ncclCommCount")?,
            ncclCommUserRank: sym(lib, "nccl", "ncclCommUserRank")?,
            ncclGetErrorString: sym(lib, "nccl", "ncclGetErrorString")?,
            ncclAllReduce: sym(lib, "nccl", "ncclAllReduce")?,
            ncclAllGather: sym(lib, "nccl", "ncclAllGather")?,
            ncclBroadcast: sym(lib, "nccl", "ncclBroadcast")?,
            ncclReduce: sym(lib, "nccl", "ncclReduce")?,
            ncclReduceScatter: sym(lib, "nccl", "ncclReduceScatter")?,
            ncclGroupStart: sym(lib, "nccl", "ncclGroupStart")?,
            ncclGroupEnd: sym(lib, "nccl", "ncclGroupEnd")?,
        })
    }
});

pub fn fns() -> Result<&'static NcclFns, &'static LoadError> { FNS.as_ref() }
pub fn is_available() -> bool { FNS.is_ok() }
