//! D3D12 compute submission: buffers, a command queue/allocator/list, fence
//! sync, and dispatch of a DXIL compute shader.
//!
//! This is the driver line and stops there. It gives you the primitives to get
//! bytes onto a device, run a shader you compiled, and get bytes back. It does
//! not compile shaders (bring DXIL, as the CUDA backend takes PTX) and it does
//! not decide what to run.
//!
//! # Binding model
//!
//! Buffers bind through **root UAV descriptors**, not descriptor tables. That
//! is a deliberate simplification: a root descriptor takes a raw GPU virtual
//! address, so there is no descriptor heap to allocate and — more to the point
//! — no call to `GetCPUDescriptorHandleForHeapStart`, which returns a small
//! struct by value and is a well-known ABI trap for hand-written D3D12
//! bindings. Root UAVs cover the flat `RWStructuredBuffer` / `RWByteAddressBuffer`
//! case that compute kernels overwhelmingly use. Texture and sampler binding
//! would need the heap path, and are not supported here.
//!
//! Vtable slot order is load-bearing throughout, exactly as in [`crate::drv`].

use core::ffi::c_void;

use crate::drv::{com_release, Device, Guid, HeapProperties, Hresult, ResourceDesc, S_OK};

// ── IIDs ───────────────────────────────────────────────────────────────────

const IID_ID3D12_COMMAND_QUEUE: Guid = Guid {
    data1: 0x0ec8_70a6,
    data2: 0x5d7e,
    data3: 0x4c22,
    data4: [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
};
const IID_ID3D12_COMMAND_ALLOCATOR: Guid = Guid {
    data1: 0x6102_dee4,
    data2: 0xaf59,
    data3: 0x4b09,
    data4: [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
};
const IID_ID3D12_GRAPHICS_COMMAND_LIST: Guid = Guid {
    data1: 0x5b16_0d0f,
    data2: 0xac1b,
    data3: 0x4185,
    data4: [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
};
const IID_ID3D12_FENCE: Guid = Guid {
    data1: 0x0a75_3dcf,
    data2: 0xc4d8,
    data3: 0x4b91,
    data4: [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
};
const IID_ID3D12_RESOURCE: Guid = Guid {
    data1: 0x6964_42be,
    data2: 0xa72e,
    data3: 0x4059,
    data4: [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
};
const IID_ID3D12_ROOT_SIGNATURE: Guid = Guid {
    data1: 0xc54a_6b66,
    data2: 0x72df,
    data3: 0x4ee8,
    data4: [0x8b, 0xe5, 0xa9, 0x46, 0xa1, 0x42, 0x92, 0x14],
};
const IID_ID3D12_PIPELINE_STATE: Guid = Guid {
    data1: 0x765a_30f3,
    data2: 0xf624,
    data3: 0x4c6f,
    data4: [0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45],
};

// ── Enums ──────────────────────────────────────────────────────────────────

const COMMAND_LIST_TYPE_COMPUTE: u32 = 2;

const HEAP_TYPE_DEFAULT: u32 = 1;
const HEAP_TYPE_UPLOAD: u32 = 2;
const HEAP_TYPE_READBACK: u32 = 3;

const RESOURCE_DIMENSION_BUFFER: u32 = 1;
const TEXTURE_LAYOUT_ROW_MAJOR: u32 = 1;
const RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS: u32 = 0x4;

const RESOURCE_STATE_COMMON: u32 = 0;
const RESOURCE_STATE_UNORDERED_ACCESS: u32 = 0x8;
const RESOURCE_STATE_COPY_DEST: u32 = 0x400;
const RESOURCE_STATE_COPY_SOURCE: u32 = 0x800;
const RESOURCE_STATE_GENERIC_READ: u32 = 0xAC3;

/// `D3D12_ROOT_PARAMETER_TYPE`: DESCRIPTOR_TABLE 0, 32BIT_CONSTANTS 1, CBV 2,
/// SRV 3, UAV 4. Getting this off by one silently builds a root signature that
/// binds `t0` instead of `u0`, which surfaces only as `E_INVALIDARG` from
/// `CreateComputePipelineState` when it fails to match the shader.
const ROOT_PARAMETER_TYPE_UAV: u32 = 4;
const SHADER_VISIBILITY_ALL: u32 = 0;
const ROOT_SIGNATURE_VERSION_1: u32 = 1;

// ── Interface vtables ──────────────────────────────────────────────────────

/// `IUnknown`(3) → `ID3D12Object`(4) → `ID3D12DeviceChild`(1) → `ID3D12Pageable`(0)
/// → `ID3D12CommandQueue`. `ExecuteCommandLists` is index 10, `Signal` is 14.
#[repr(C)]
struct ICommandQueueVtbl {
    _iunknown: [*const c_void; 3],
    _object: [*const c_void; 4],
    _get_device: *const c_void,
    _update_tile_mappings: *const c_void,
    _copy_tile_mappings: *const c_void,
    execute_command_lists: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void),
    _set_marker: *const c_void,
    _begin_event: *const c_void,
    _end_event: *const c_void,
    signal: unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> Hresult,
    _wait: *const c_void,
}

/// `…ID3D12CommandAllocator`; `Reset` is index 8.
#[repr(C)]
struct ICommandAllocatorVtbl {
    _iunknown: [*const c_void; 3],
    _object: [*const c_void; 4],
    _get_device: *const c_void,
    reset: unsafe extern "system" fn(*mut c_void) -> Hresult,
}

/// `…ID3D12CommandList`(1: GetType) → `ID3D12GraphicsCommandList`.
/// `Close` 9, `Reset` 10, `Dispatch` 14, `CopyBufferRegion` 15,
/// `SetPipelineState` 25, `ResourceBarrier` 26, `SetComputeRootSignature` 29,
/// `SetComputeRootUnorderedAccessView` 41.
#[repr(C)]
struct IGraphicsCommandListVtbl {
    _iunknown: [*const c_void; 3],
    _object: [*const c_void; 4],
    _get_device: *const c_void,
    _get_type: *const c_void,
    close: unsafe extern "system" fn(*mut c_void) -> Hresult,
    reset: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) -> Hresult,
    _clear_state: *const c_void,
    _draw_instanced: *const c_void,
    _draw_indexed_instanced: *const c_void,
    dispatch: unsafe extern "system" fn(*mut c_void, u32, u32, u32),
    copy_buffer_region:
        unsafe extern "system" fn(*mut c_void, *mut c_void, u64, *mut c_void, u64, u64),
    _copy_texture_region: *const c_void,
    _copy_resource: *const c_void,
    _copy_tiles: *const c_void,
    _resolve_subresource: *const c_void,
    _ia_set_primitive_topology: *const c_void,
    _rs_set_viewports: *const c_void,
    _rs_set_scissor_rects: *const c_void,
    _om_set_blend_factor: *const c_void,
    _om_set_stencil_ref: *const c_void,
    set_pipeline_state: unsafe extern "system" fn(*mut c_void, *mut c_void),
    resource_barrier: unsafe extern "system" fn(*mut c_void, u32, *const ResourceBarrier),
    _execute_bundle: *const c_void,
    _set_descriptor_heaps: *const c_void,
    set_compute_root_signature: unsafe extern "system" fn(*mut c_void, *mut c_void),
    _set_graphics_root_signature: *const c_void,
    _set_compute_root_descriptor_table: *const c_void,
    _set_graphics_root_descriptor_table: *const c_void,
    _set_compute_root_32bit_constant: *const c_void,
    _set_graphics_root_32bit_constant: *const c_void,
    _set_compute_root_32bit_constants: *const c_void,
    _set_graphics_root_32bit_constants: *const c_void,
    _set_compute_root_constant_buffer_view: *const c_void,
    _set_graphics_root_constant_buffer_view: *const c_void,
    _set_compute_root_shader_resource_view: *const c_void,
    _set_graphics_root_shader_resource_view: *const c_void,
    set_compute_root_unordered_access_view: unsafe extern "system" fn(*mut c_void, u32, u64),
}

/// `…ID3D12Fence`; `GetCompletedValue` 8, `SetEventOnCompletion` 9, `Signal` 10.
#[repr(C)]
struct IFenceVtbl {
    _iunknown: [*const c_void; 3],
    _object: [*const c_void; 4],
    _get_device: *const c_void,
    get_completed_value: unsafe extern "system" fn(*mut c_void) -> u64,
}

/// `…ID3D12Resource`; `Map` 8, `Unmap` 9, `GetDesc` 10 (by value — not called),
/// `GetGPUVirtualAddress` 11.
#[repr(C)]
struct IResourceVtbl {
    _iunknown: [*const c_void; 3],
    _object: [*const c_void; 4],
    _get_device: *const c_void,
    map: unsafe extern "system" fn(*mut c_void, u32, *const Range, *mut *mut c_void) -> Hresult,
    unmap: unsafe extern "system" fn(*mut c_void, u32, *const Range),
    _get_desc: *const c_void,
    get_gpu_virtual_address: unsafe extern "system" fn(*mut c_void) -> u64,
}

/// `ID3DBlob`: `IUnknown`(3) → `GetBufferPointer` 3, `GetBufferSize` 4.
#[repr(C)]
struct IBlobVtbl {
    _iunknown: [*const c_void; 3],
    get_buffer_pointer: unsafe extern "system" fn(*mut c_void) -> *mut c_void,
    get_buffer_size: unsafe extern "system" fn(*mut c_void) -> usize,
}

// ── Plain structs ──────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
struct Range {
    begin: usize,
    end: usize,
}

/// `D3D12_RESOURCE_BARRIER` specialised to the transition variant. The C union
/// is 24 bytes and 8-aligned (it holds a pointer), so the transition fields sit
/// at offset 8 and the struct is 32 bytes.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct ResourceBarrier {
    type_: u32,
    flags: u32,
    resource: *mut c_void,
    subresource: u32,
    state_before: u32,
    state_after: u32,
    _pad: u32,
}

/// `D3D12_ROOT_PARAMETER` specialised to the root-descriptor variant. The union
/// is 16 bytes (its widest member, `D3D12_ROOT_DESCRIPTOR_TABLE`, is a `UINT`
/// plus a pointer) and 8-aligned, so the descriptor's two `UINT`s occupy the
/// first 8 bytes of it and the visibility field lands at offset 24.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct RootParameterUav {
    parameter_type: u32,
    _pad0: u32,
    shader_register: u32,
    register_space: u32,
    _union_tail: [u32; 2],
    shader_visibility: u32,
    _pad1: u32,
}

#[repr(C)]
struct RootSignatureDesc {
    num_parameters: u32,
    _pad: u32,
    p_parameters: *const RootParameterUav,
    num_static_samplers: u32,
    _pad2: u32,
    p_static_samplers: *const c_void,
    flags: u32,
    _pad3: u32,
}

#[repr(C)]
struct ComputePipelineStateDesc {
    root_signature: *mut c_void,
    cs_bytecode: *const c_void,
    cs_length: usize,
    node_mask: u32,
    _pad: u32,
    cached_blob: *const c_void,
    cached_size: usize,
    flags: u32,
    _pad2: u32,
}

#[repr(C)]
struct CommandQueueDesc {
    type_: u32,
    priority: i32,
    flags: u32,
    node_mask: u32,
}

// ── Error ──────────────────────────────────────────────────────────────────

/// A failed D3D12 call, carrying the operation name and raw `HRESULT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error {
    pub op: &'static str,
    pub hr: Hresult,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "d3d12 {} failed (hr {:#010x})", self.op, self.hr)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;

fn check(op: &'static str, hr: Hresult) -> Result<()> {
    if hr == S_OK {
        Ok(())
    } else {
        Err(Error { op, hr })
    }
}

// ── Owned COM wrappers ─────────────────────────────────────────────────────

macro_rules! com_owned {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        pub struct $name {
            raw: *mut c_void,
        }
        // SAFETY: D3D12 objects are free-threaded; the raw pointer is owned
        // solely by this wrapper and released exactly once on drop.
        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}
        impl $name {
            /// Borrowed raw pointer — do not release.
            pub fn as_raw(&self) -> *mut c_void {
                self.raw
            }
        }
        impl Drop for $name {
            fn drop(&mut self) {
                unsafe { com_release(self.raw) }
            }
        }
    };
}

com_owned!(CommandQueue, "An `ID3D12CommandQueue` of type COMPUTE.");
com_owned!(CommandAllocator, "An `ID3D12CommandAllocator`.");
com_owned!(RootSignature, "An `ID3D12RootSignature`.");
com_owned!(
    PipelineState,
    "An `ID3D12PipelineState` for a compute shader."
);
com_owned!(Fence, "An `ID3D12Fence` used for host/device sync.");

/// A committed buffer resource in one of the three standard heaps.
pub struct Buffer {
    raw: *mut c_void,
    len_bytes: u64,
    heap: u32,
}

// SAFETY: see `com_owned!`.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { com_release(self.raw) }
    }
}

impl Buffer {
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }

    pub fn len_bytes(&self) -> u64 {
        self.len_bytes
    }

    /// GPU virtual address, for binding as a root UAV.
    pub fn gpu_address(&self) -> u64 {
        unsafe {
            let v = *(self.raw as *mut *mut IResourceVtbl);
            ((*v).get_gpu_virtual_address)(self.raw)
        }
    }

    /// Copy `data` into an UPLOAD-heap buffer.
    ///
    /// Only valid on UPLOAD buffers — DEFAULT-heap memory is not CPU-visible,
    /// and mapping it is a driver error rather than a slow path.
    pub fn write(&self, data: &[u8]) -> Result<()> {
        if self.heap != HEAP_TYPE_UPLOAD {
            return Err(Error {
                op: "write (buffer is not in the UPLOAD heap)",
                hr: 0,
            });
        }
        let n = data.len().min(self.len_bytes as usize);
        unsafe {
            let v = *(self.raw as *mut *mut IResourceVtbl);
            let mut p: *mut c_void = core::ptr::null_mut();
            // Read range begin == end tells the driver we will not read.
            let no_read = Range { begin: 0, end: 0 };
            check(
                "ID3D12Resource::Map",
                ((*v).map)(self.raw, 0, &no_read, &mut p),
            )?;
            core::ptr::copy_nonoverlapping(data.as_ptr(), p as *mut u8, n);
            let written = Range { begin: 0, end: n };
            ((*v).unmap)(self.raw, 0, &written);
        }
        Ok(())
    }

    /// Copy out of a READBACK-heap buffer into `out`.
    pub fn read(&self, out: &mut [u8]) -> Result<()> {
        if self.heap != HEAP_TYPE_READBACK {
            return Err(Error {
                op: "read (buffer is not in the READBACK heap)",
                hr: 0,
            });
        }
        let n = out.len().min(self.len_bytes as usize);
        unsafe {
            let v = *(self.raw as *mut *mut IResourceVtbl);
            let mut p: *mut c_void = core::ptr::null_mut();
            let all = Range { begin: 0, end: n };
            check("ID3D12Resource::Map", ((*v).map)(self.raw, 0, &all, &mut p))?;
            core::ptr::copy_nonoverlapping(p as *const u8, out.as_mut_ptr(), n);
            let wrote_nothing = Range { begin: 0, end: 0 };
            ((*v).unmap)(self.raw, 0, &wrote_nothing);
        }
        Ok(())
    }
}

// ── Context ────────────────────────────────────────────────────────────────

/// A compute context: one device, one COMPUTE queue, one allocator, one list,
/// and a fence. Enough to submit work and wait for it.
pub struct Context {
    device: Device,
    queue: CommandQueue,
    allocator: CommandAllocator,
    list: *mut c_void,
    fence: Fence,
    next_fence_value: core::cell::Cell<u64>,
}

// SAFETY: every owned handle is free-threaded; `Cell<u64>` is not `Sync`, so
// `Context` is deliberately `Send` only.
unsafe impl Send for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { com_release(self.list) }
    }
}

impl Context {
    /// Bring up a compute context on the `ordinal`-th enumerated adapter.
    pub fn new(ordinal: u32) -> Result<Self> {
        let device = crate::drv::open(ordinal).ok_or(Error {
            op: "open device",
            hr: 0,
        })?;
        Self::from_device(device)
    }

    /// Bring up a compute context on an already-open device.
    pub fn from_device(device: Device) -> Result<Self> {
        unsafe {
            let v = device.vtbl();
            let raw = device.as_raw();

            let qdesc = CommandQueueDesc {
                type_: COMMAND_LIST_TYPE_COMPUTE,
                priority: 0,
                flags: 0,
                node_mask: 0,
            };
            let mut queue: *mut c_void = core::ptr::null_mut();
            check(
                "ID3D12Device::CreateCommandQueue",
                ((*v).create_command_queue)(
                    raw,
                    &qdesc as *const _ as *const c_void,
                    &IID_ID3D12_COMMAND_QUEUE,
                    &mut queue,
                ),
            )?;
            let queue = CommandQueue { raw: queue };

            let mut alloc: *mut c_void = core::ptr::null_mut();
            check(
                "ID3D12Device::CreateCommandAllocator",
                ((*v).create_command_allocator)(
                    raw,
                    COMMAND_LIST_TYPE_COMPUTE,
                    &IID_ID3D12_COMMAND_ALLOCATOR,
                    &mut alloc,
                ),
            )?;
            let allocator = CommandAllocator { raw: alloc };

            let mut list: *mut c_void = core::ptr::null_mut();
            check(
                "ID3D12Device::CreateCommandList",
                ((*v).create_command_list)(
                    raw,
                    0,
                    COMMAND_LIST_TYPE_COMPUTE,
                    allocator.as_raw(),
                    core::ptr::null_mut(),
                    &IID_ID3D12_GRAPHICS_COMMAND_LIST,
                    &mut list,
                ),
            )?;
            // Lists are created open; close it so `begin` can Reset uniformly.
            let lv = *(list as *mut *mut IGraphicsCommandListVtbl);
            check("ID3D12GraphicsCommandList::Close", ((*lv).close)(list))?;

            let mut fence: *mut c_void = core::ptr::null_mut();
            check(
                "ID3D12Device::CreateFence",
                ((*v).create_fence)(raw, 0, 0, &IID_ID3D12_FENCE, &mut fence),
            )?;

            Ok(Context {
                device,
                queue,
                allocator,
                list,
                fence: Fence { raw: fence },
                next_fence_value: core::cell::Cell::new(1),
            })
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    fn create_buffer(&self, len_bytes: u64, heap: u32) -> Result<Buffer> {
        let (initial_state, flags) = match heap {
            HEAP_TYPE_UPLOAD => (RESOURCE_STATE_GENERIC_READ, 0),
            HEAP_TYPE_READBACK => (RESOURCE_STATE_COPY_DEST, 0),
            _ => (RESOURCE_STATE_COMMON, RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        };
        let props = HeapProperties {
            type_: heap,
            ..Default::default()
        };
        let desc = ResourceDesc {
            dimension: RESOURCE_DIMENSION_BUFFER,
            width: len_bytes.max(1),
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 0,
            sample_count: 1,
            sample_quality: 0,
            layout: TEXTURE_LAYOUT_ROW_MAJOR,
            flags,
            ..Default::default()
        };
        unsafe {
            let v = self.device.vtbl();
            let mut res: *mut c_void = core::ptr::null_mut();
            check(
                "ID3D12Device::CreateCommittedResource",
                ((*v).create_committed_resource)(
                    self.device.as_raw(),
                    &props,
                    0,
                    &desc,
                    initial_state,
                    core::ptr::null(),
                    &IID_ID3D12_RESOURCE,
                    &mut res,
                ),
            )?;
            Ok(Buffer {
                raw: res,
                len_bytes,
                heap,
            })
        }
    }

    /// Device-local buffer, UAV-capable. Not CPU-visible.
    pub fn device_buffer(&self, len_bytes: u64) -> Result<Buffer> {
        self.create_buffer(len_bytes, HEAP_TYPE_DEFAULT)
    }

    /// CPU-writable staging buffer.
    pub fn upload_buffer(&self, len_bytes: u64) -> Result<Buffer> {
        self.create_buffer(len_bytes, HEAP_TYPE_UPLOAD)
    }

    /// CPU-readable staging buffer.
    pub fn readback_buffer(&self, len_bytes: u64) -> Result<Buffer> {
        self.create_buffer(len_bytes, HEAP_TYPE_READBACK)
    }

    /// A root signature with `n` root UAV descriptors bound to `u0..un`.
    pub fn root_signature_with_uavs(&self, n: u32) -> Result<RootSignature> {
        let l = crate::drv::loaded().ok_or(Error {
            op: "d3d12.dll not loaded",
            hr: 0,
        })?;
        let serialize = l.serialize_root_signature.ok_or(Error {
            op: "D3D12SerializeRootSignature not exported",
            hr: 0,
        })?;

        let params: Vec<RootParameterUav> = (0..n)
            .map(|i| RootParameterUav {
                parameter_type: ROOT_PARAMETER_TYPE_UAV,
                _pad0: 0,
                shader_register: i,
                register_space: 0,
                _union_tail: [0; 2],
                shader_visibility: SHADER_VISIBILITY_ALL,
                _pad1: 0,
            })
            .collect();
        let desc = RootSignatureDesc {
            num_parameters: n,
            _pad: 0,
            p_parameters: params.as_ptr(),
            num_static_samplers: 0,
            _pad2: 0,
            p_static_samplers: core::ptr::null(),
            flags: 0,
            _pad3: 0,
        };

        unsafe {
            let mut blob: *mut c_void = core::ptr::null_mut();
            let mut err: *mut c_void = core::ptr::null_mut();
            let hr = serialize(
                &desc as *const _ as *const c_void,
                ROOT_SIGNATURE_VERSION_1,
                &mut blob,
                &mut err,
            );
            com_release(err);
            check("D3D12SerializeRootSignature", hr)?;

            let bv = *(blob as *mut *mut IBlobVtbl);
            let ptr = ((*bv).get_buffer_pointer)(blob);
            let size = ((*bv).get_buffer_size)(blob);

            let v = self.device.vtbl();
            let mut sig: *mut c_void = core::ptr::null_mut();
            let hr = ((*v).create_root_signature)(
                self.device.as_raw(),
                0,
                ptr,
                size,
                &IID_ID3D12_ROOT_SIGNATURE,
                &mut sig,
            );
            com_release(blob);
            check("ID3D12Device::CreateRootSignature", hr)?;
            Ok(RootSignature { raw: sig })
        }
    }

    /// Extract the root signature embedded in a compiled shader blob.
    ///
    /// `CreateRootSignature` accepts either a serialised root signature or a
    /// whole shader container carrying one from `[RootSignature(...)]`. This is
    /// the path to prefer when the shader already declares its bindings: the
    /// signature cannot then disagree with the shader.
    pub fn root_signature_from_blob(&self, blob: &[u8]) -> Result<RootSignature> {
        unsafe {
            let v = self.device.vtbl();
            let mut sig: *mut c_void = core::ptr::null_mut();
            check(
                "ID3D12Device::CreateRootSignature (from shader blob)",
                ((*v).create_root_signature)(
                    self.device.as_raw(),
                    0,
                    blob.as_ptr() as *const c_void,
                    blob.len(),
                    &IID_ID3D12_ROOT_SIGNATURE,
                    &mut sig,
                ),
            )?;
            Ok(RootSignature { raw: sig })
        }
    }

    /// Build a compute pipeline from a **DXIL** blob. Compile it yourself —
    /// `dxc -T cs_6_0 -E main shader.hlsl -Fo shader.dxil`.
    ///
    /// Pass `None` for `root` when the shader carries its own root signature
    /// via `[RootSignature(...)]`; D3D12 then takes the embedded one.
    pub fn compute_pipeline(
        &self,
        root: Option<&RootSignature>,
        dxil: &[u8],
    ) -> Result<PipelineState> {
        let desc = ComputePipelineStateDesc {
            root_signature: root.map(|r| r.as_raw()).unwrap_or(core::ptr::null_mut()),
            cs_bytecode: dxil.as_ptr() as *const c_void,
            cs_length: dxil.len(),
            node_mask: 0,
            _pad: 0,
            cached_blob: core::ptr::null(),
            cached_size: 0,
            flags: 0,
            _pad2: 0,
        };
        unsafe {
            let v = self.device.vtbl();
            let mut pso: *mut c_void = core::ptr::null_mut();
            check(
                "ID3D12Device::CreateComputePipelineState",
                ((*v).create_compute_pipeline_state)(
                    self.device.as_raw(),
                    &desc as *const _ as *const c_void,
                    &IID_ID3D12_PIPELINE_STATE,
                    &mut pso,
                ),
            )?;
            Ok(PipelineState { raw: pso })
        }
    }

    /// Reset the allocator and open the command list for recording.
    fn begin(&self, pso: Option<&PipelineState>) -> Result<()> {
        unsafe {
            let av = *(self.allocator.as_raw() as *mut *mut ICommandAllocatorVtbl);
            check(
                "ID3D12CommandAllocator::Reset",
                ((*av).reset)(self.allocator.as_raw()),
            )?;
            let lv = *(self.list as *mut *mut IGraphicsCommandListVtbl);
            check(
                "ID3D12GraphicsCommandList::Reset",
                ((*lv).reset)(
                    self.list,
                    self.allocator.as_raw(),
                    pso.map(|p| p.as_raw()).unwrap_or(core::ptr::null_mut()),
                ),
            )
        }
    }

    /// Close, submit, and block until the GPU has finished.
    fn end_and_wait(&self) -> Result<()> {
        unsafe {
            let lv = *(self.list as *mut *mut IGraphicsCommandListVtbl);
            check("ID3D12GraphicsCommandList::Close", ((*lv).close)(self.list))?;

            let qv = *(self.queue.as_raw() as *mut *mut ICommandQueueVtbl);
            let lists = [self.list];
            ((*qv).execute_command_lists)(self.queue.as_raw(), 1, lists.as_ptr());

            let target = self.next_fence_value.get();
            self.next_fence_value.set(target + 1);
            check(
                "ID3D12CommandQueue::Signal",
                ((*qv).signal)(self.queue.as_raw(), self.fence.as_raw(), target),
            )?;

            // Spin on the fence rather than taking a Win32 event: it keeps the
            // crate free of a kernel32 dependency, and every submission here is
            // short enough that parking would cost more than it saves.
            let fv = *(self.fence.as_raw() as *mut *mut IFenceVtbl);
            while ((*fv).get_completed_value)(self.fence.as_raw()) < target {
                core::hint::spin_loop();
            }
            Ok(())
        }
    }

    unsafe fn barrier(&self, buf: &Buffer, before: u32, after: u32) {
        if before == after {
            return;
        }
        let b = ResourceBarrier {
            type_: 0,
            flags: 0,
            resource: buf.as_raw(),
            subresource: 0xffff_ffff,
            state_before: before,
            state_after: after,
            _pad: 0,
        };
        let lv = *(self.list as *mut *mut IGraphicsCommandListVtbl);
        ((*lv).resource_barrier)(self.list, 1, &b);
    }

    /// Upload `data` into a fresh device-local buffer and wait for the copy.
    pub fn upload(&self, data: &[u8]) -> Result<Buffer> {
        let staging = self.upload_buffer(data.len() as u64)?;
        staging.write(data)?;
        let dst = self.device_buffer(data.len() as u64)?;

        self.begin(None)?;
        unsafe {
            self.barrier(&dst, RESOURCE_STATE_COMMON, RESOURCE_STATE_COPY_DEST);
            let lv = *(self.list as *mut *mut IGraphicsCommandListVtbl);
            ((*lv).copy_buffer_region)(
                self.list,
                dst.as_raw(),
                0,
                staging.as_raw(),
                0,
                data.len() as u64,
            );
            self.barrier(
                &dst,
                RESOURCE_STATE_COPY_DEST,
                RESOURCE_STATE_UNORDERED_ACCESS,
            );
        }
        self.end_and_wait()?;
        Ok(dst)
    }

    /// Read a device-local buffer back to host memory.
    pub fn download(&self, src: &Buffer, out: &mut [u8]) -> Result<()> {
        let staging = self.readback_buffer(out.len() as u64)?;
        self.begin(None)?;
        unsafe {
            self.barrier(
                src,
                RESOURCE_STATE_UNORDERED_ACCESS,
                RESOURCE_STATE_COPY_SOURCE,
            );
            let lv = *(self.list as *mut *mut IGraphicsCommandListVtbl);
            ((*lv).copy_buffer_region)(
                self.list,
                staging.as_raw(),
                0,
                src.as_raw(),
                0,
                out.len() as u64,
            );
            self.barrier(
                src,
                RESOURCE_STATE_COPY_SOURCE,
                RESOURCE_STATE_UNORDERED_ACCESS,
            );
        }
        self.end_and_wait()?;
        staging.read(out)
    }

    /// Bind `buffers` to root UAV slots `0..n` and dispatch, then wait.
    pub fn dispatch(
        &self,
        root: &RootSignature,
        pso: &PipelineState,
        buffers: &[&Buffer],
        groups: [u32; 3],
    ) -> Result<()> {
        self.begin(Some(pso))?;
        unsafe {
            let lv = *(self.list as *mut *mut IGraphicsCommandListVtbl);
            ((*lv).set_compute_root_signature)(self.list, root.as_raw());
            ((*lv).set_pipeline_state)(self.list, pso.as_raw());
            for (i, b) in buffers.iter().enumerate() {
                ((*lv).set_compute_root_unordered_access_view)(
                    self.list,
                    i as u32,
                    b.gpu_address(),
                );
            }
            ((*lv).dispatch)(self.list, groups[0], groups[1], groups[2]);
        }
        self.end_and_wait()
    }
}

/// A compute pipeline paired with the root signature it was built against —
/// D3D12 needs both at dispatch, so the unified
/// [`ComputeDevice`](ironaccelerator_core::ComputeDevice) trait bundles them.
pub struct BoundPipeline {
    root: RootSignature,
    pso: PipelineState,
}

/// Unified cross-backend compute surface. `code` is a signed DXIL container;
/// the root signature is generated with `bindings` root UAVs at `u0..un`.
impl ironaccelerator_core::ComputeDevice for Context {
    type Buffer = Buffer;
    type Pipeline = BoundPipeline;
    type Error = Error;

    fn device_buffer(&self, bytes: u64) -> Result<Buffer> {
        Context::device_buffer(self, bytes)
    }

    fn upload(&self, data: &[u8]) -> Result<Buffer> {
        Context::upload(self, data)
    }

    fn download(&self, buffer: &Buffer, out: &mut [u8]) -> Result<()> {
        Context::download(self, buffer, out)
    }

    fn pipeline(&self, code: &[u8], bindings: u32) -> Result<BoundPipeline> {
        let root = self.root_signature_with_uavs(bindings)?;
        let pso = self.compute_pipeline(Some(&root), code)?;
        Ok(BoundPipeline { root, pso })
    }

    fn dispatch(
        &self,
        pipeline: &BoundPipeline,
        buffers: &[&Buffer],
        groups: [u32; 3],
    ) -> Result<()> {
        Context::dispatch(self, &pipeline.root, &pipeline.pso, buffers, groups)
    }

    fn buffer_len(&self, buffer: &Buffer) -> u64 {
        buffer.len_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> Option<Context> {
        if crate::drv::enumerate().is_empty() {
            return None;
        }
        Some(Context::new(0).expect("adapter enumerated but context failed"))
    }

    #[test]
    fn context_builds_on_every_enumerated_adapter() {
        for a in crate::drv::enumerate() {
            let c = Context::new(a.ordinal).expect("context");
            assert!(!c.queue().as_raw().is_null());
        }
    }

    #[test]
    fn buffer_round_trip_preserves_bytes() {
        let Some(c) = ctx() else { return };
        let src: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        let dev = c.upload(&src).expect("upload");
        assert_eq!(dev.len_bytes(), src.len() as u64);
        assert_ne!(dev.gpu_address(), 0, "device buffer must have a GPU VA");

        let mut out = vec![0u8; src.len()];
        c.download(&dev, &mut out).expect("download");
        assert_eq!(out, src, "device round-trip corrupted data");
    }

    #[test]
    fn root_signature_with_uavs_builds() {
        let Some(c) = ctx() else { return };
        let sig = c.root_signature_with_uavs(1).expect("root signature");
        assert!(!sig.as_raw().is_null());
    }

    #[test]
    fn upload_heap_rejects_read_and_readback_rejects_write() {
        let Some(c) = ctx() else { return };
        let up = c.upload_buffer(64).unwrap();
        assert!(up.read(&mut [0u8; 64]).is_err());
        let rb = c.readback_buffer(64).unwrap();
        assert!(rb.write(&[0u8; 64]).is_err());
    }
}
