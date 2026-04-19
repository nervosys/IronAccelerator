//! RCCL — AMD collective comms. API parity with NCCL.

use crate::hip::HipStream;
use crate::loader::{sym, try_load, LoadError, LoaderResult};
use libloading::Library;
use std::ffi::{c_int, c_void};
use std::sync::{LazyLock, OnceLock};

#[repr(transparent)] #[derive(Copy, Clone, Debug, Default)] pub struct RcclComm(pub *mut c_void);
unsafe impl Send for RcclComm {} unsafe impl Sync for RcclComm {}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RcclResult {
    Success = 0, UnhandledHipError = 1, SystemError = 2, InternalError = 3,
    InvalidArgument = 4, InvalidUsage = 5, RemoteError = 6, Other = 0xFFFF_FFFF,
}
impl RcclResult {
    #[inline] pub fn ok(self) -> Result<(), Self> { if self == Self::Success { Ok(()) } else { Err(self) } }
    #[inline] pub fn is_ok(self) -> bool { self == Self::Success }
}

#[repr(u32)] #[derive(Copy, Clone, Debug)]
pub enum RcclDataType {
    I8 = 0, U8 = 1, I32 = 2, U32 = 3, I64 = 4, U64 = 5,
    F16 = 6, F32 = 7, F64 = 8, BF16 = 9,
}

#[repr(u32)] #[derive(Copy, Clone, Debug)]
pub enum RcclRedOp { Sum = 0, Prod = 1, Max = 2, Min = 3, Avg = 4 }

#[repr(C)] #[derive(Copy, Clone, Debug)]
pub struct RcclUniqueId { pub bytes: [u8; 128] }
impl Default for RcclUniqueId {
    fn default() -> Self { Self { bytes: [0; 128] } }
}

#[allow(clippy::type_complexity)]
pub struct RcclFns {
    pub ncclGetUniqueId: unsafe extern "C" fn(*mut RcclUniqueId) -> RcclResult,
    pub ncclCommInitRank: unsafe extern "C" fn(*mut RcclComm, c_int, RcclUniqueId, c_int) -> RcclResult,
    pub ncclCommDestroy: unsafe extern "C" fn(RcclComm) -> RcclResult,
    pub ncclCommCount: unsafe extern "C" fn(RcclComm, *mut c_int) -> RcclResult,
    pub ncclCommUserRank: unsafe extern "C" fn(RcclComm, *mut c_int) -> RcclResult,
    pub ncclGroupStart: unsafe extern "C" fn() -> RcclResult,
    pub ncclGroupEnd: unsafe extern "C" fn() -> RcclResult,

    pub ncclAllReduce: unsafe extern "C" fn(
        *const c_void, *mut c_void, usize, RcclDataType, RcclRedOp, RcclComm, HipStream,
    ) -> RcclResult,
    pub ncclAllGather: unsafe extern "C" fn(
        *const c_void, *mut c_void, usize, RcclDataType, RcclComm, HipStream,
    ) -> RcclResult,
    pub ncclBroadcast: unsafe extern "C" fn(
        *const c_void, *mut c_void, usize, RcclDataType, c_int, RcclComm, HipStream,
    ) -> RcclResult,
    pub ncclReduceScatter: unsafe extern "C" fn(
        *const c_void, *mut c_void, usize, RcclDataType, RcclRedOp, RcclComm, HipStream,
    ) -> RcclResult,
}

static LIB: LazyLock<LoaderResult<Library>> = LazyLock::new(|| {
    // RCCL exports the `ncclXxx` names just like NCCL, for API portability.
    try_load(&["librccl.so", "librccl.so.1", "rccl.dll"])
});
static FNS: OnceLock<LoaderResult<RcclFns>> = OnceLock::new();

fn load_fns(lib: &Library) -> LoaderResult<RcclFns> {
    macro_rules! g { ($s:ident) => { sym(lib, "rccl", stringify!($s))? } }
    unsafe {
        Ok(RcclFns {
            ncclGetUniqueId: g!(ncclGetUniqueId),
            ncclCommInitRank: g!(ncclCommInitRank),
            ncclCommDestroy: g!(ncclCommDestroy),
            ncclCommCount: g!(ncclCommCount),
            ncclCommUserRank: g!(ncclCommUserRank),
            ncclGroupStart: g!(ncclGroupStart),
            ncclGroupEnd: g!(ncclGroupEnd),
            ncclAllReduce: g!(ncclAllReduce),
            ncclAllGather: g!(ncclAllGather),
            ncclBroadcast: g!(ncclBroadcast),
            ncclReduceScatter: g!(ncclReduceScatter),
        })
    }
}

pub fn fns() -> Result<&'static RcclFns, &'static LoadError> {
    FNS.get_or_init(|| {
        let lib = LIB.as_ref().map_err(clone_err)?;
        load_fns(lib)
    }).as_ref()
}

pub fn is_available() -> bool { fns().is_ok() }

fn clone_err(e: &LoadError) -> LoadError {
    match e {
        LoadError::LibraryNotFound { tried, last } =>
            LoadError::LibraryNotFound { tried: tried.clone(), last: last.clone() },
        LoadError::SymbolMissing { lib, symbol, err } =>
            LoadError::SymbolMissing { lib, symbol, err: err.clone() },
    }
}
