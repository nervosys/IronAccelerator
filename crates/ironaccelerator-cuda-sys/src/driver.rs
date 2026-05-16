//! CUDA Driver API (`libcuda.so` / `nvcuda.dll`). Targets 13.2.
//!
//! Only the functions IronAccelerator's safe layer actually uses are bound.
//! If you need a new one, add it to [`DriverFns`] and `DriverFns::load`.

use crate::loader::{sym, try_load, LoadError, LoaderResult};
use libloading::Library;
use std::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{LazyLock, OnceLock};

// ── opaque handles ──────────────────────────────────────────────────────────

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CUcontext(pub *mut c_void);
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CUstream(pub *mut c_void);
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CUevent(pub *mut c_void);
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CUmodule(pub *mut c_void);
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CUfunction(pub *mut c_void);

// ── cudarc-compatibility opaque-pointer aliases ────────────────────────────
//
// cudarc declares its handles as opaque pointer typedefs:
//   pub struct CUstream_st { _unused: [u8; 0] }
//   pub type CUstream = *mut CUstream_st;
// Some downstream code casts via these opaque pointer types
// (`*mut CUstream_st`). Provide them as zero-sized opaque structs so code
// like `ptr as *mut iron_cuda_sys::driver::CUstream_st` resolves.
// These do NOT replace [`CUstream`] — they're solely for opaque-pointer-cast
// compatibility with cudarc-shaped call sites during migration.

/// cudarc-style opaque-pointer counterpart of [`CUstream`].
/// Use `*mut CUstream_st` where cudarc-shaped pointer-typedef APIs expect it.
#[repr(C)]
pub struct CUstream_st {
    _unused: [u8; 0],
}

/// cudarc-style opaque-pointer counterpart of [`CUevent`].
#[repr(C)]
pub struct CUevent_st {
    _unused: [u8; 0],
}

/// cudarc-style opaque-pointer counterpart of [`CUmodule`].
#[repr(C)]
pub struct CUmod_st {
    _unused: [u8; 0],
}

/// cudarc-style opaque-pointer counterpart of [`CUfunction`].
#[repr(C)]
pub struct CUfunc_st {
    _unused: [u8; 0],
}
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CUgraph(pub *mut c_void);
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CUgraphExec(pub *mut c_void);
/// `CUgraphExecUpdateResult` — why an in-place graph update succeeded or
/// failed. Per `cuda.h` `CUgraphExecUpdateResult_enum`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum CUgraphExecUpdateResult_enum {
    CU_GRAPH_EXEC_UPDATE_SUCCESS = 0x0,
    CU_GRAPH_EXEC_UPDATE_ERROR = 0x1,
    CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = 0x2,
    CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = 0x3,
    CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = 0x4,
    CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = 0x5,
    CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = 0x6,
    CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE = 0x7,
    CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED = 0x8,
}

/// `CUgraphExecUpdateResultInfo_st` — result-info struct out-parameter of
/// `cuGraphExecUpdate_v2`. Field order matches `cuda.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct CUgraphExecUpdateResultInfo_st {
    pub result: CUgraphExecUpdateResult_enum,
    pub errorNode: CUgraphNode,
    pub errorFromNode: CUgraphNode,
}
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CUmemPool(pub *mut c_void);

/// `CUmemPool_attribute_enum`. Subset matching CUDA 13. The release
/// threshold is the key one for retain-on-free behaviour.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum CUmemPool_attribute {
    /// `cuuint64_t` — pool releases memory back to OS only when usage
    /// exceeds this many bytes. Set to `u64::MAX` to retain forever
    /// (CUDA caching-allocator pattern). Default is 0 (release every free).
    ReleaseThreshold = 1,
    ReservedMemCurrent = 2,
    ReservedMemHigh = 3,
    UsedMemCurrent = 4,
    UsedMemHigh = 5,
    ReuseFollowEventDependencies = 6,
    ReuseAllowOpportunistic = 7,
    ReuseAllowInternalDependencies = 8,
}

impl CUmemPool_attribute {
    /// cudarc-compatibility alias.
    #[allow(non_upper_case_globals)]
    pub const CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: Self = Self::ReleaseThreshold;
}

// ── CUDA 12.3+/12.4+/13.x additions (optional symbols) ──────────────────────

/// Virtual-memory allocation handle (`CUmemGenericAllocationHandle`).
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CUmemGenericAllocationHandle(pub u64);

/// Multicast team handle. Added in CUDA 12.3.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CUmemcastObjectHandle(pub u64);

/// Green context. Added in CUDA 12.4.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CUgreenCtx(pub *mut c_void);

/// Opaque resource descriptor for green-context partitioning.
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct CUdevResource(pub [u8; 48]); // 48-byte opaque struct per the header
impl Default for CUdevResource {
    fn default() -> Self {
        Self([0u8; 48])
    }
}

/// Graph conditional handle (if/while predicate). Added in CUDA 12.4.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CUgraphConditionalHandle(pub u64);

/// Graph node handle.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default)]
pub struct CUgraphNode(pub *mut c_void);

unsafe impl Send for CUgreenCtx {}
unsafe impl Sync for CUgreenCtx {}
unsafe impl Send for CUgraphNode {}
unsafe impl Sync for CUgraphNode {}

/// `CUmemAllocationType`.
#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum CUmemAllocationType {
    Invalid = 0,
    Pinned = 1,
}

/// `CUmemLocationType`.
#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum CUmemLocationType {
    Invalid = 0,
    Device = 1,
    Host = 2,
    HostNuma = 3,
    HostNumaCurrent = 4,
}

/// `CUmemAllocationHandleType`.
#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum CUmemAllocationHandleType {
    None = 0,
    PosixFileDescriptor = 1,
    Win32 = 2,
    Win32Kmt = 4,
    Fabric = 8,
}

/// Flags bit-field for `CUmemAllocationProp`. CUDA 13 adds the bit for
/// confidential-computing / encrypted memory regions.
pub const CU_MEM_CREATE_USAGE_ENCRYPT: u32 = 1 << 1;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUmemLocation {
    pub kind: CUmemLocationType,
    pub id: c_int,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUmemAllocationProp {
    pub kind: CUmemAllocationType,
    pub requested_handle_types: CUmemAllocationHandleType,
    pub location: CUmemLocation,
    pub win32_handle_metadata: *mut c_void,
    pub alloc_flags: CUmemAllocationPropFlags,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct CUmemAllocationPropFlags {
    pub compression_type: u8,
    pub gpu_direct_rdma_capable: u8,
    pub usage: u16, // bitfield — OR of CU_MEM_CREATE_USAGE_* (encrypted, etc.)
    pub reserved: [u8; 4],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUmemAccessDesc {
    pub location: CUmemLocation,
    pub flags: u32, // CUmemAccess_flags: None=0, Read=1, ReadWrite=3
}

/// Graph-node type.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CUgraphNodeType {
    Kernel = 0,
    Memcpy = 1,
    Memset = 2,
    Host = 3,
    Graph = 4,
    Empty = 5,
    WaitEvent = 6,
    EventRecord = 7,
    ExtSemasSignal = 8,
    ExtSemasWait = 9,
    MemAlloc = 10,
    MemFree = 11,
    BatchMemOp = 12,
    Conditional = 13,
}

impl CUgraphNodeType {
    /// cudarc-compatibility aliases (CU_GRAPH_NODE_TYPE_*).
    #[allow(non_upper_case_globals)] pub const CU_GRAPH_NODE_TYPE_KERNEL: Self = Self::Kernel;
    #[allow(non_upper_case_globals)] pub const CU_GRAPH_NODE_TYPE_MEMCPY: Self = Self::Memcpy;
    #[allow(non_upper_case_globals)] pub const CU_GRAPH_NODE_TYPE_MEMSET: Self = Self::Memset;
    #[allow(non_upper_case_globals)] pub const CU_GRAPH_NODE_TYPE_HOST: Self = Self::Host;
    #[allow(non_upper_case_globals)] pub const CU_GRAPH_NODE_TYPE_GRAPH: Self = Self::Graph;
    #[allow(non_upper_case_globals)] pub const CU_GRAPH_NODE_TYPE_EMPTY: Self = Self::Empty;
    #[allow(non_upper_case_globals)] pub const CU_GRAPH_NODE_TYPE_WAIT_EVENT: Self = Self::WaitEvent;
    #[allow(non_upper_case_globals)] pub const CU_GRAPH_NODE_TYPE_EVENT_RECORD: Self = Self::EventRecord;
    #[allow(non_upper_case_globals)] pub const CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL: Self = Self::ExtSemasSignal;
    #[allow(non_upper_case_globals)] pub const CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT: Self = Self::ExtSemasWait;
    #[allow(non_upper_case_globals)] pub const CU_GRAPH_NODE_TYPE_MEM_ALLOC: Self = Self::MemAlloc;
    #[allow(non_upper_case_globals)] pub const CU_GRAPH_NODE_TYPE_MEM_FREE: Self = Self::MemFree;
    #[allow(non_upper_case_globals)] pub const CU_GRAPH_NODE_TYPE_BATCH_MEM_OP: Self = Self::BatchMemOp;
}

/// Conditional-node type.
#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum CUgraphConditionalNodeType {
    If = 0,
    While = 1,
    Switch = 2,
}

/// Device ordinal (i32). Not a pointer — the driver actually passes this by value.
pub type CUdevice = c_int;

/// Device-side pointer. 64-bit on every supported platform.
pub type CUdeviceptr = u64;

unsafe impl Send for CUcontext {}
unsafe impl Sync for CUcontext {}
unsafe impl Send for CUstream {}
unsafe impl Sync for CUstream {}
unsafe impl Send for CUevent {}
unsafe impl Sync for CUevent {}
unsafe impl Send for CUmodule {}
unsafe impl Sync for CUmodule {}
unsafe impl Send for CUfunction {}
unsafe impl Sync for CUfunction {}
unsafe impl Send for CUgraph {}
unsafe impl Sync for CUgraph {}
unsafe impl Send for CUgraphExec {}
unsafe impl Sync for CUgraphExec {}
unsafe impl Send for CUmemPool {}
unsafe impl Sync for CUmemPool {}

// ── result codes (subset, with `Ok` = 0) ───────────────────────────────────

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CUresult {
    Success = 0,
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    Deinitialized = 4,
    NoDevice = 100,
    InvalidDevice = 101,
    InvalidContext = 201,
    MapFailed = 205,
    UnmapFailed = 206,
    ArrayIsMapped = 207,
    AlreadyMapped = 208,
    NoBinaryForGpu = 209,
    AlreadyAcquired = 210,
    NotMapped = 211,
    InvalidSource = 300,
    FileNotFound = 301,
    SharedObjectSymbolNotFound = 302,
    SharedObjectInitFailed = 303,
    OperatingSystem = 304,
    InvalidHandle = 400,
    NotFound = 500,
    NotReady = 600,
    LaunchFailed = 700,
    LaunchOutOfResources = 701,
    LaunchTimeout = 702,
    PeerAccessAlreadyEnabled = 704,
    ContextIsDestroyed = 709,
    StreamCaptureUnsupported = 900,
    Unknown = 999,
    /// Any code we don't model — hold on to the numeric value.
    Other = 0xFFFF_FFFF,
}

impl CUresult {
    /// cudarc-compatibility alias: equivalent to [`CUresult::Success`].
    /// Lets `cudarc::driver::sys::CUresult::CUDA_SUCCESS`-style call sites
    /// migrate to `iron_cuda_sys::driver::CUresult::CUDA_SUCCESS` with no
    /// other change. Same memory representation as `Success` (variant 0).
    #[allow(non_upper_case_globals)]
    pub const CUDA_SUCCESS: Self = Self::Success;

    #[inline]
    pub fn from_raw(r: u32) -> Self {
        // Map the ones we know; everything else → Other.
        match r {
            0 => Self::Success,
            1 => Self::InvalidValue,
            2 => Self::OutOfMemory,
            3 => Self::NotInitialized,
            4 => Self::Deinitialized,
            100 => Self::NoDevice,
            101 => Self::InvalidDevice,
            201 => Self::InvalidContext,
            205 => Self::MapFailed,
            206 => Self::UnmapFailed,
            207 => Self::ArrayIsMapped,
            208 => Self::AlreadyMapped,
            209 => Self::NoBinaryForGpu,
            210 => Self::AlreadyAcquired,
            211 => Self::NotMapped,
            300 => Self::InvalidSource,
            301 => Self::FileNotFound,
            302 => Self::SharedObjectSymbolNotFound,
            303 => Self::SharedObjectInitFailed,
            304 => Self::OperatingSystem,
            400 => Self::InvalidHandle,
            500 => Self::NotFound,
            600 => Self::NotReady,
            700 => Self::LaunchFailed,
            701 => Self::LaunchOutOfResources,
            702 => Self::LaunchTimeout,
            704 => Self::PeerAccessAlreadyEnabled,
            709 => Self::ContextIsDestroyed,
            900 => Self::StreamCaptureUnsupported,
            999 => Self::Unknown,
            _ => Self::Other,
        }
    }

    #[inline]
    pub fn ok(self) -> Result<(), Self> {
        if self == Self::Success {
            Ok(())
        } else {
            Err(self)
        }
    }
    #[inline]
    pub fn is_ok(self) -> bool {
        self == Self::Success
    }
}

// ── attribute / flag enums we need ─────────────────────────────────────────

#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CUdevice_attribute {
    MaxThreadsPerBlock = 1,
    MaxSharedMemoryPerBlock = 8,
    TotalConstantMemory = 9,
    WarpSize = 10,
    MaxRegistersPerBlock = 12,
    ClockRate = 13,
    MultiprocessorCount = 16,
    IntegrationType = 18,
    ComputeCapabilityMajor = 75,
    ComputeCapabilityMinor = 76,
    PciBusId = 33,
    PciDeviceId = 34,
    PciDomainId = 50,
    MemoryClockRate = 36,
    GlobalMemoryBusWidth = 37,
    L2CacheSize = 38,
    MaxThreadsPerMultiProcessor = 39,
    AsyncEngineCount = 40,
    UnifiedAddressing = 41,
    StreamPrioritiesSupported = 78,
    CooperativeLaunch = 95,
    ConcurrentManagedAccess = 89,
    ComputePreemptionSupported = 90,
    ComputeMode = 20,
    ManagedMemory = 83,
    MultiGpuBoard = 84,
    MemoryPoolsSupported = 115,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum CUevent_flags {
    Default = 0x0,
    BlockingSync = 0x1,
    DisableTiming = 0x2,
    Interprocess = 0x4,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum CUstreamCaptureMode {
    Global = 0,
    ThreadLocal = 1,
    Relaxed = 2,
}

impl CUstreamCaptureMode {
    /// cudarc-compatibility alias.
    #[allow(non_upper_case_globals)]
    pub const CU_STREAM_CAPTURE_MODE_GLOBAL: Self = Self::Global;
    /// cudarc-compatibility alias.
    #[allow(non_upper_case_globals)]
    pub const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: Self = Self::ThreadLocal;
    /// cudarc-compatibility alias.
    #[allow(non_upper_case_globals)]
    pub const CU_STREAM_CAPTURE_MODE_RELAXED: Self = Self::Relaxed;
}

/// cudarc-style type alias for [`CUstreamCaptureMode`] (matches the
/// `_enum` suffix bindgen produces).
#[allow(non_camel_case_types)]
pub type CUstreamCaptureMode_enum = CUstreamCaptureMode;

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum CUhostAllocFlags {
    Default = 0x0,
    Portable = 0x1,
    Mapped = 0x2,
    WriteCombined = 0x4,
}

/// Per-function tunables. The IDs match the CUDA driver `CUfunction_attribute`
/// enum so they can be passed directly to `cuFuncSetAttribute` /
/// `cuFuncGetAttribute`.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CUfunction_attribute {
    MaxThreadsPerBlock = 0,
    SharedSizeBytes = 1,
    ConstSizeBytes = 2,
    LocalSizeBytes = 3,
    NumRegs = 4,
    PtxVersion = 5,
    BinaryVersion = 6,
    CacheModeCa = 7,
    /// **Hopper+ / required for >48 KB dynamic shared memory.**
    MaxDynamicSharedSizeBytes = 8,
    PreferredSharedMemoryCarveout = 9,
    ClusterSizeMustBeSet = 10,
    RequiredClusterWidth = 11,
    RequiredClusterHeight = 12,
    RequiredClusterDepth = 13,
}

impl CUfunction_attribute {
    #[allow(non_upper_case_globals)] pub const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: Self = Self::MaxThreadsPerBlock;
    #[allow(non_upper_case_globals)] pub const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: Self = Self::SharedSizeBytes;
    #[allow(non_upper_case_globals)] pub const CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: Self = Self::ConstSizeBytes;
    #[allow(non_upper_case_globals)] pub const CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: Self = Self::LocalSizeBytes;
    #[allow(non_upper_case_globals)] pub const CU_FUNC_ATTRIBUTE_NUM_REGS: Self = Self::NumRegs;
    #[allow(non_upper_case_globals)] pub const CU_FUNC_ATTRIBUTE_PTX_VERSION: Self = Self::PtxVersion;
    #[allow(non_upper_case_globals)] pub const CU_FUNC_ATTRIBUTE_BINARY_VERSION: Self = Self::BinaryVersion;
    #[allow(non_upper_case_globals)] pub const CU_FUNC_ATTRIBUTE_CACHE_MODE_CA: Self = Self::CacheModeCa;
    #[allow(non_upper_case_globals)] pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: Self = Self::MaxDynamicSharedSizeBytes;
    #[allow(non_upper_case_globals)] pub const CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: Self = Self::PreferredSharedMemoryCarveout;
}

impl CUdevice_attribute {
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: Self = Self::MaxThreadsPerBlock;
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: Self = Self::MaxSharedMemoryPerBlock;
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: Self = Self::TotalConstantMemory;
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_WARP_SIZE: Self = Self::WarpSize;
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: Self = Self::MaxRegistersPerBlock;
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_CLOCK_RATE: Self = Self::ClockRate;
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: Self = Self::MultiprocessorCount;
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: Self = Self::ComputeCapabilityMajor;
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: Self = Self::ComputeCapabilityMinor;
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: Self = Self::PciBusId;
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: Self = Self::PciDeviceId;
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: Self = Self::PciDomainId;
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: Self = Self::MemoryClockRate;
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: Self = Self::GlobalMemoryBusWidth;
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: Self = Self::L2CacheSize;
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: Self = Self::MaxThreadsPerMultiProcessor;
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT: Self = Self::AsyncEngineCount;
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH: Self = Self::CooperativeLaunch;
    #[allow(non_upper_case_globals)] pub const CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED: Self = Self::MemoryPoolsSupported;
}

/// 16-byte device UUID, matches the CUDA driver `CUuuid` struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CUuuid {
    pub bytes: [u8; 16],
}

pub const CU_STREAM_DEFAULT: u32 = 0x0;
pub const CU_STREAM_NON_BLOCKING: u32 = 0x1;

// ── function pointer table ─────────────────────────────────────────────────

#[allow(clippy::type_complexity)]
pub struct DriverFns {
    pub cuInit: unsafe extern "C" fn(c_uint) -> CUresult,
    pub cuDriverGetVersion: unsafe extern "C" fn(*mut c_int) -> CUresult,

    pub cuDeviceGetCount: unsafe extern "C" fn(*mut c_int) -> CUresult,
    pub cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
    pub cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult,
    pub cuDeviceGetAttribute:
        unsafe extern "C" fn(*mut c_int, CUdevice_attribute, CUdevice) -> CUresult,
    pub cuDeviceTotalMem_v2: unsafe extern "C" fn(*mut usize, CUdevice) -> CUresult,
    pub cuDeviceCanAccessPeer: unsafe extern "C" fn(*mut c_int, CUdevice, CUdevice) -> CUresult,
    pub cuDeviceGetUuid_v2: unsafe extern "C" fn(*mut CUuuid, CUdevice) -> CUresult,

    pub cuDevicePrimaryCtxRetain: unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult,
    pub cuDevicePrimaryCtxRelease_v2: unsafe extern "C" fn(CUdevice) -> CUresult,

    pub cuCtxSetCurrent: unsafe extern "C" fn(CUcontext) -> CUresult,
    pub cuCtxGetCurrent: unsafe extern "C" fn(*mut CUcontext) -> CUresult,
    pub cuCtxPushCurrent_v2: unsafe extern "C" fn(CUcontext) -> CUresult,
    pub cuCtxPopCurrent_v2: unsafe extern "C" fn(*mut CUcontext) -> CUresult,
    pub cuCtxSynchronize: unsafe extern "C" fn() -> CUresult,
    pub cuCtxEnablePeerAccess: unsafe extern "C" fn(CUcontext, c_uint) -> CUresult,

    pub cuStreamCreate: unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult,
    pub cuStreamCreateWithPriority: unsafe extern "C" fn(*mut CUstream, c_uint, c_int) -> CUresult,
    pub cuStreamDestroy_v2: unsafe extern "C" fn(CUstream) -> CUresult,
    pub cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUresult,
    pub cuStreamWaitEvent: unsafe extern "C" fn(CUstream, CUevent, c_uint) -> CUresult,
    pub cuStreamGetPriorityRange: unsafe extern "C" fn(*mut c_int, *mut c_int) -> CUresult,

    pub cuEventCreate: unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult,
    pub cuEventDestroy_v2: unsafe extern "C" fn(CUevent) -> CUresult,
    pub cuEventRecord: unsafe extern "C" fn(CUevent, CUstream) -> CUresult,
    pub cuEventSynchronize: unsafe extern "C" fn(CUevent) -> CUresult,
    pub cuEventElapsedTime: unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult,

    pub cuMemAlloc_v2: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    pub cuMemAllocAsync: unsafe extern "C" fn(*mut CUdeviceptr, usize, CUstream) -> CUresult,
    pub cuMemGetInfo_v2: unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult,
    pub cuMemFree_v2: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    pub cuMemFreeAsync: unsafe extern "C" fn(CUdeviceptr, CUstream) -> CUresult,
    pub cuMemsetD8Async: unsafe extern "C" fn(CUdeviceptr, u8, usize, CUstream) -> CUresult,
    pub cuMemcpyHtoDAsync_v2:
        unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult,
    pub cuMemcpyDtoHAsync_v2:
        unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult,
    /// Synchronous variants. Block the calling thread until the copy is
    /// complete. Use these when you would otherwise issue `Async_v2` +
    /// `cuStreamSynchronize` — the synchronous form skips the per-call
    /// stream-state machinery and is ~10-20 % faster on small-to-medium
    /// transfers (matches cudarc's `clone_dtoh` behaviour).
    pub cuMemcpyHtoD_v2: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    pub cuMemcpyDtoH_v2: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    pub cuMemcpyDtoDAsync_v2:
        unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult,
    /// Cross-device async memcpy. Either or both contexts must allow peer
    /// access; the call enqueues on `stream` (which belongs to the *source*
    /// context, per the CUDA driver contract).
    pub cuMemcpyPeerAsync: unsafe extern "C" fn(
        CUdeviceptr, // dst
        CUcontext,   // dst ctx
        CUdeviceptr, // src
        CUcontext,   // src ctx
        usize,
        CUstream,
    ) -> CUresult,
    pub cuMemHostAlloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CUresult,
    pub cuMemFreeHost: unsafe extern "C" fn(*mut c_void) -> CUresult,
    /// Page-lock an existing host allocation so the CUDA driver can DMA
    /// directly to/from it without staging through an internal pinned
    /// bounce buffer. The flag word controls portability across contexts
    /// and write-combined semantics; `0` is the default suitable for
    /// general use. Used by IA's host-registration cache to make repeated
    /// transfers of the same buffer bandwidth-optimal.
    pub cuMemHostRegister_v2: unsafe extern "C" fn(*mut c_void, usize, c_uint) -> CUresult,
    pub cuMemHostUnregister: unsafe extern "C" fn(*mut c_void) -> CUresult,
    /// Profiler control. Brackets between `cuProfilerStart`/`cuProfilerStop`
    /// define a "capture range" that `nsys` / `ncu` with `--capture-range=
    /// cudaProfilerApi` will exclusively trace. Required to measure
    /// graph-replayed kernels (which `nsys` otherwise sees as one opaque
    /// `cuGraphLaunch`).
    pub cuProfilerStart: unsafe extern "C" fn() -> CUresult,
    pub cuProfilerStop: unsafe extern "C" fn() -> CUresult,

    pub cuModuleLoadData: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    pub cuModuleUnload: unsafe extern "C" fn(CUmodule) -> CUresult,
    pub cuModuleGetFunction:
        unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    /// Look up a device-side `__constant__` / `__device__` symbol by name.
    /// Returns its device pointer and byte size.
    pub cuModuleGetGlobal_v2:
        unsafe extern "C" fn(*mut CUdeviceptr, *mut usize, CUmodule, *const c_char) -> CUresult,

    pub cuLaunchKernel: unsafe extern "C" fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint, // grid
        c_uint,
        c_uint,
        c_uint, // block
        c_uint, // shared mem bytes
        CUstream,
        *mut *mut c_void, // kernel params
        *mut *mut c_void, // extra
    ) -> CUresult,
    /// Cooperative-groups launch. Same arg order as `cuLaunchKernel`,
    /// minus the `extra` slot. The kernel must be compiled with
    /// `--cooperative-groups` and all blocks must fit on the device
    /// simultaneously.
    pub cuLaunchCooperativeKernel: unsafe extern "C" fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
    ) -> CUresult,

    /// Read a per-function attribute (e.g. `MaxDynamicSharedSizeBytes`).
    pub cuFuncGetAttribute:
        unsafe extern "C" fn(*mut c_int, CUfunction_attribute, CUfunction) -> CUresult,
    /// Set a per-function attribute. Required to opt in to >48 KiB of
    /// dynamic shared memory and to set cluster dims on Hopper+.
    pub cuFuncSetAttribute:
        unsafe extern "C" fn(CUfunction, CUfunction_attribute, c_int) -> CUresult,

    /// Query a device's **default** stream-ordered memory pool. The pool
    /// returned is the one `cuMemAllocAsync` draws from when called without
    /// an explicit pool — i.e. the one IronAccelerator's allocator uses.
    pub cuDeviceGetDefaultMemPool:
        unsafe extern "C" fn(*mut CUmemPool, CUdevice) -> CUresult,

    /// Set a pool attribute. **The single most important call here is
    /// `ReleaseThreshold = u64::MAX`** — without it, every free returns
    /// memory to the OS at next sync, costing remap latency on the next
    /// allocation. With it set, the pool retains memory across free/alloc
    /// cycles (the CUDA caching-allocator pattern PyTorch/cudarc use).
    pub cuMemPoolSetAttribute: unsafe extern "C" fn(
        CUmemPool,
        CUmemPool_attribute,
        *mut c_void,
    ) -> CUresult,

    /// Query a pool attribute.
    pub cuMemPoolGetAttribute: unsafe extern "C" fn(
        CUmemPool,
        CUmemPool_attribute,
        *mut c_void,
    ) -> CUresult,

    /// Returns the maximum number of active thread blocks per SM for the
    /// given function with the supplied block-size and dynamic shmem usage.
    /// Use with `MultiprocessorCount` device attribute for occupancy-based
    /// grid sizing.
    pub cuOccupancyMaxActiveBlocksPerMultiprocessor: unsafe extern "C" fn(
        *mut c_int,
        CUfunction,
        c_int, // block size
        usize, // dynamic shmem bytes
    ) -> CUresult,

    pub cuStreamBeginCapture_v2: unsafe extern "C" fn(CUstream, CUstreamCaptureMode) -> CUresult,
    pub cuStreamEndCapture: unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUresult,
    pub cuGraphDestroy: unsafe extern "C" fn(CUgraph) -> CUresult,
    pub cuGraphInstantiateWithFlags:
        unsafe extern "C" fn(*mut CUgraphExec, CUgraph, u64) -> CUresult,
    pub cuGraphExecDestroy: unsafe extern "C" fn(CUgraphExec) -> CUresult,
    pub cuGraphLaunch: unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult,

    /// In-place patch of an instantiated graph (CUDA 12.0+). Hot-path fast
    /// path: a re-captured decode graph with same topology can be applied to
    /// an existing exec in ~µs vs ~10× slower full re-instantiation.
    pub cuGraphExecUpdate_v2: unsafe extern "C" fn(
        CUgraphExec,
        CUgraph,
        *mut CUgraphExecUpdateResultInfo_st,
    ) -> CUresult,

    /// Enumerate nodes in a graph. Used for graph inspection / validation.
    pub cuGraphGetNodes: unsafe extern "C" fn(
        CUgraph,
        *mut CUgraphNode,
        *mut usize,
    ) -> CUresult,

    /// Query a node's type. Used to validate graph topology after capture.
    pub cuGraphNodeGetType:
        unsafe extern "C" fn(CUgraphNode, *mut CUgraphNodeType) -> CUresult,

    // ── Optional / CUDA 12.3+ additions. All loaded via sym_opt so the crate
    // ── still links on older drivers; callers probe with `is_some()`.
    /// Virtual-memory allocation — physical backing. Needed for encrypted
    /// and multicast-bindable memory.
    pub cuMemCreate: Option<
        unsafe extern "C" fn(
            *mut CUmemGenericAllocationHandle,
            usize,
            *const CUmemAllocationProp,
            u64,
        ) -> CUresult,
    >,
    pub cuMemRelease: Option<unsafe extern "C" fn(CUmemGenericAllocationHandle) -> CUresult>,
    pub cuMemAddressReserve:
        Option<unsafe extern "C" fn(*mut CUdeviceptr, usize, usize, CUdeviceptr, u64) -> CUresult>,
    pub cuMemAddressFree: Option<unsafe extern "C" fn(CUdeviceptr, usize) -> CUresult>,
    pub cuMemMap: Option<
        unsafe extern "C" fn(
            CUdeviceptr,
            usize,
            usize,
            CUmemGenericAllocationHandle,
            u64,
        ) -> CUresult,
    >,
    pub cuMemUnmap: Option<unsafe extern "C" fn(CUdeviceptr, usize) -> CUresult>,
    pub cuMemSetAccess:
        Option<unsafe extern "C" fn(CUdeviceptr, usize, *const CUmemAccessDesc, usize) -> CUresult>,
    pub cuMemGetAllocationGranularity:
        Option<unsafe extern "C" fn(*mut usize, *const CUmemAllocationProp, u32) -> CUresult>,

    /// Multicast (driver-initiated collectives). CUDA 12.3+.
    pub cuMulticastCreate: Option<
        unsafe extern "C" fn(*mut CUmemcastObjectHandle, *const CUmemcastObjectProp) -> CUresult,
    >,
    pub cuMulticastAddDevice:
        Option<unsafe extern "C" fn(CUmemcastObjectHandle, CUdevice) -> CUresult>,
    pub cuMulticastBindMem: Option<
        unsafe extern "C" fn(
            CUmemcastObjectHandle,
            usize,
            CUmemGenericAllocationHandle,
            usize,
            usize,
            u64,
        ) -> CUresult,
    >,
    pub cuMulticastBindAddr: Option<
        unsafe extern "C" fn(CUmemcastObjectHandle, usize, CUdeviceptr, usize, u64) -> CUresult,
    >,
    pub cuMulticastUnbind:
        Option<unsafe extern "C" fn(CUmemcastObjectHandle, CUdevice, usize, usize) -> CUresult>,
    pub cuMulticastGetGranularity:
        Option<unsafe extern "C" fn(*mut usize, *const CUmemcastObjectProp, u32) -> CUresult>,

    /// Green contexts. CUDA 12.4+.
    pub cuGreenCtxCreate:
        Option<unsafe extern "C" fn(*mut CUgreenCtx, CUdevResource, CUdevice, u32) -> CUresult>,
    pub cuGreenCtxDestroy: Option<unsafe extern "C" fn(CUgreenCtx) -> CUresult>,
    pub cuGreenCtxRecordEvent: Option<unsafe extern "C" fn(CUgreenCtx, CUevent) -> CUresult>,
    pub cuCtxFromGreenCtx: Option<unsafe extern "C" fn(*mut CUcontext, CUgreenCtx) -> CUresult>,
    pub cuDeviceGetDevResource:
        Option<unsafe extern "C" fn(CUdevice, *mut CUdevResource, u32) -> CUresult>,
    pub cuDevResourceGenerateDesc:
        Option<unsafe extern "C" fn(*mut CUdevResource, *mut CUdevResource, u32) -> CUresult>,
    pub cuStreamCreateFromGreenCtx:
        Option<unsafe extern "C" fn(*mut CUstream, CUgreenCtx, u32) -> CUresult>,

    /// Graph conditional nodes. CUDA 12.4+.
    pub cuGraphConditionalHandleCreate: Option<
        unsafe extern "C" fn(
            *mut CUgraphConditionalHandle,
            CUgraph,
            CUcontext,
            u32,
            u32,
        ) -> CUresult,
    >,
    pub cuGraphAddNode: Option<
        unsafe extern "C" fn(
            *mut CUgraphNode,
            CUgraph,
            *const CUgraphNode,
            usize,
            *mut c_void,
        ) -> CUresult,
    >,
}

/// `CUmulticastObjectProp` — parameters for a multicast team. Placed here
/// because the pointer-type appears in [`DriverFns`] fields.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUmemcastObjectProp {
    pub num_devices: u32,
    pub size: usize,
    pub handle_types: u64, // bitfield of CUmemAllocationHandleType
    pub flags: u64,
}

fn load_library() -> LoaderResult<Library> {
    try_load(&[
        "libcuda.so.1",
        "libcuda.so",
        "nvcuda.dll",
        "/usr/lib/wsl/lib/libcuda.so.1",
    ])
}

fn load_fns(lib: &Library) -> LoaderResult<DriverFns> {
    macro_rules! g {
        ($sym:ident) => {
            sym(lib, "libcuda", stringify!($sym))?
        };
    }
    // Some driver versions (notably the Windows nvcuda.dll) export the
    // context-scoped alias `cuCtxGetStreamPriorityRange` instead of, or in
    // addition to, `cuStreamGetPriorityRange`. Signatures are identical. Fall
    // back to the alias so we work on both.
    let cu_stream_get_priority_range: unsafe extern "C" fn(*mut c_int, *mut c_int) -> CUresult =
        match unsafe { crate::loader::sym_opt(lib, "cuStreamGetPriorityRange") } {
            Some(f) => f,
            None => unsafe { sym(lib, "libcuda", "cuCtxGetStreamPriorityRange")? },
        };
    unsafe {
        Ok(DriverFns {
            cuInit: g!(cuInit),
            cuDriverGetVersion: g!(cuDriverGetVersion),
            cuDeviceGetCount: g!(cuDeviceGetCount),
            cuDeviceGet: g!(cuDeviceGet),
            cuDeviceGetName: g!(cuDeviceGetName),
            cuDeviceGetAttribute: g!(cuDeviceGetAttribute),
            cuDeviceTotalMem_v2: g!(cuDeviceTotalMem_v2),
            cuDeviceCanAccessPeer: g!(cuDeviceCanAccessPeer),
            cuDeviceGetUuid_v2: g!(cuDeviceGetUuid_v2),
            cuDevicePrimaryCtxRetain: g!(cuDevicePrimaryCtxRetain),
            cuDevicePrimaryCtxRelease_v2: g!(cuDevicePrimaryCtxRelease_v2),
            cuCtxSetCurrent: g!(cuCtxSetCurrent),
            cuCtxGetCurrent: g!(cuCtxGetCurrent),
            cuCtxPushCurrent_v2: g!(cuCtxPushCurrent_v2),
            cuCtxPopCurrent_v2: g!(cuCtxPopCurrent_v2),
            cuCtxSynchronize: g!(cuCtxSynchronize),
            cuCtxEnablePeerAccess: g!(cuCtxEnablePeerAccess),
            cuStreamCreate: g!(cuStreamCreate),
            cuStreamCreateWithPriority: g!(cuStreamCreateWithPriority),
            cuStreamDestroy_v2: g!(cuStreamDestroy_v2),
            cuStreamSynchronize: g!(cuStreamSynchronize),
            cuStreamWaitEvent: g!(cuStreamWaitEvent),
            cuStreamGetPriorityRange: cu_stream_get_priority_range,
            cuEventCreate: g!(cuEventCreate),
            cuEventDestroy_v2: g!(cuEventDestroy_v2),
            cuEventRecord: g!(cuEventRecord),
            cuEventSynchronize: g!(cuEventSynchronize),
            cuEventElapsedTime: g!(cuEventElapsedTime),
            cuMemAlloc_v2: g!(cuMemAlloc_v2),
            cuMemAllocAsync: g!(cuMemAllocAsync),
            cuMemFree_v2: g!(cuMemFree_v2),
            cuMemFreeAsync: g!(cuMemFreeAsync),
            cuMemGetInfo_v2: g!(cuMemGetInfo_v2),
            cuMemsetD8Async: g!(cuMemsetD8Async),
            cuMemcpyHtoDAsync_v2: g!(cuMemcpyHtoDAsync_v2),
            cuMemcpyDtoHAsync_v2: g!(cuMemcpyDtoHAsync_v2),
            cuMemcpyHtoD_v2: g!(cuMemcpyHtoD_v2),
            cuMemcpyDtoH_v2: g!(cuMemcpyDtoH_v2),
            cuMemcpyDtoDAsync_v2: g!(cuMemcpyDtoDAsync_v2),
            cuMemcpyPeerAsync: g!(cuMemcpyPeerAsync),
            cuMemHostAlloc: g!(cuMemHostAlloc),
            cuMemFreeHost: g!(cuMemFreeHost),
            cuMemHostRegister_v2: g!(cuMemHostRegister_v2),
            cuMemHostUnregister: g!(cuMemHostUnregister),
            cuProfilerStart: g!(cuProfilerStart),
            cuProfilerStop: g!(cuProfilerStop),
            cuModuleLoadData: g!(cuModuleLoadData),
            cuModuleUnload: g!(cuModuleUnload),
            cuModuleGetFunction: g!(cuModuleGetFunction),
            cuModuleGetGlobal_v2: g!(cuModuleGetGlobal_v2),
            cuLaunchKernel: g!(cuLaunchKernel),
            cuLaunchCooperativeKernel: g!(cuLaunchCooperativeKernel),
            cuFuncGetAttribute: g!(cuFuncGetAttribute),
            cuFuncSetAttribute: g!(cuFuncSetAttribute),
            cuDeviceGetDefaultMemPool: g!(cuDeviceGetDefaultMemPool),
            cuMemPoolSetAttribute: g!(cuMemPoolSetAttribute),
            cuMemPoolGetAttribute: g!(cuMemPoolGetAttribute),
            cuOccupancyMaxActiveBlocksPerMultiprocessor: g!(
                cuOccupancyMaxActiveBlocksPerMultiprocessor
            ),
            cuStreamBeginCapture_v2: g!(cuStreamBeginCapture_v2),
            cuStreamEndCapture: g!(cuStreamEndCapture),
            cuGraphDestroy: g!(cuGraphDestroy),
            cuGraphInstantiateWithFlags: g!(cuGraphInstantiateWithFlags),
            cuGraphExecDestroy: g!(cuGraphExecDestroy),
            cuGraphLaunch: g!(cuGraphLaunch),
            cuGraphExecUpdate_v2: g!(cuGraphExecUpdate_v2),
            cuGraphGetNodes: g!(cuGraphGetNodes),
            cuGraphNodeGetType: g!(cuGraphNodeGetType),

            cuMemCreate: crate::loader::sym_opt(lib, "cuMemCreate"),
            cuMemRelease: crate::loader::sym_opt(lib, "cuMemRelease"),
            cuMemAddressReserve: crate::loader::sym_opt(lib, "cuMemAddressReserve"),
            cuMemAddressFree: crate::loader::sym_opt(lib, "cuMemAddressFree"),
            cuMemMap: crate::loader::sym_opt(lib, "cuMemMap"),
            cuMemUnmap: crate::loader::sym_opt(lib, "cuMemUnmap"),
            cuMemSetAccess: crate::loader::sym_opt(lib, "cuMemSetAccess"),
            cuMemGetAllocationGranularity: crate::loader::sym_opt(
                lib,
                "cuMemGetAllocationGranularity",
            ),

            cuMulticastCreate: crate::loader::sym_opt(lib, "cuMulticastCreate"),
            cuMulticastAddDevice: crate::loader::sym_opt(lib, "cuMulticastAddDevice"),
            cuMulticastBindMem: crate::loader::sym_opt(lib, "cuMulticastBindMem"),
            cuMulticastBindAddr: crate::loader::sym_opt(lib, "cuMulticastBindAddr"),
            cuMulticastUnbind: crate::loader::sym_opt(lib, "cuMulticastUnbind"),
            cuMulticastGetGranularity: crate::loader::sym_opt(lib, "cuMulticastGetGranularity"),

            cuGreenCtxCreate: crate::loader::sym_opt(lib, "cuGreenCtxCreate"),
            cuGreenCtxDestroy: crate::loader::sym_opt(lib, "cuGreenCtxDestroy"),
            cuGreenCtxRecordEvent: crate::loader::sym_opt(lib, "cuGreenCtxRecordEvent"),
            cuCtxFromGreenCtx: crate::loader::sym_opt(lib, "cuCtxFromGreenCtx"),
            cuDeviceGetDevResource: crate::loader::sym_opt(lib, "cuDeviceGetDevResource"),
            cuDevResourceGenerateDesc: crate::loader::sym_opt(lib, "cuDevResourceGenerateDesc"),
            cuStreamCreateFromGreenCtx: crate::loader::sym_opt(lib, "cuStreamCreateFromGreenCtx"),

            cuGraphConditionalHandleCreate: crate::loader::sym_opt(
                lib,
                "cuGraphConditionalHandleCreate",
            ),
            cuGraphAddNode: crate::loader::sym_opt(lib, "cuGraphAddNode"),
        })
    }
}

static LIB: OnceLock<Library> = OnceLock::new();

static FNS: LazyLock<Result<DriverFns, LoadError>> = LazyLock::new(|| {
    let lib = load_library()?;
    let stored = LIB.get_or_init(|| lib);
    load_fns(stored)
});

static INIT_DONE: OnceLock<CUresult> = OnceLock::new();

/// Hot-path cache. After the first successful `fns()` call this holds a
/// non-null pointer to the function table; subsequent calls become a single
/// relaxed atomic load + null check, avoiding two `OnceLock` acquires per
/// driver call (~30–50 ns saved on every wrapped op).
static FNS_HOT: AtomicPtr<DriverFns> = AtomicPtr::new(std::ptr::null_mut());

/// Resolve the driver function table. First call performs a `cuInit(0)`.
#[inline]
pub fn fns() -> Result<&'static DriverFns, &'static LoadError> {
    // Fast path: pointer cached, library loaded, cuInit done. Acquire pairs
    // with the Release store in `fns_slow` so we observe a fully-initialised
    // `DriverFns` on weak-memory targets (ARM / Apple Silicon).
    let cached = FNS_HOT.load(Ordering::Acquire);
    if !cached.is_null() {
        // SAFETY: only ever set to a `&'static DriverFns` reference below,
        // and never cleared, so the pointer is valid for 'static.
        return Ok(unsafe { &*cached });
    }
    fns_slow()
}

#[cold]
#[inline(never)]
fn fns_slow() -> Result<&'static DriverFns, &'static LoadError> {
    let f = FNS.as_ref()?;
    INIT_DONE.get_or_init(|| unsafe { (f.cuInit)(0) });
    // Publish the static pointer for the hot path. `Release` so that the
    // OnceLock writes happen-before the pointer becomes observable.
    FNS_HOT.store(f as *const _ as *mut _, Ordering::Release);
    Ok(f)
}

/// Library is present and `cuInit(0)` returned `Success`. Useful as a
/// `Backend::is_available` probe.
pub fn is_available() -> bool {
    match fns() {
        Ok(_) => INIT_DONE.get().is_some_and(|r| r.is_ok()),
        Err(_) => false,
    }
}

// ── convenience wrappers so call sites read cleanly ─────────────────────────

/// Assert `CUresult::Success`, returning the `CUresult` on failure.
#[inline]
pub fn check(r: CUresult) -> Result<(), CUresult> {
    r.ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_enum_roundtrips_known_codes() {
        assert_eq!(CUresult::from_raw(0), CUresult::Success);
        assert_eq!(CUresult::from_raw(700), CUresult::LaunchFailed);
        assert_eq!(CUresult::from_raw(12345), CUresult::Other);
    }

    #[test]
    fn handles_are_zero_sized_newtypes() {
        assert_eq!(
            std::mem::size_of::<CUcontext>(),
            std::mem::size_of::<*mut ()>()
        );
        assert_eq!(
            std::mem::size_of::<CUstream>(),
            std::mem::size_of::<*mut ()>()
        );
    }

    #[test]
    fn result_ok_is_false_for_errors() {
        assert!(CUresult::Success.is_ok());
        assert!(!CUresult::OutOfMemory.is_ok());
    }
}
