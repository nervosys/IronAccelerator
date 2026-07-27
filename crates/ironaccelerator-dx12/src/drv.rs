//! Minimal Direct3D 12 driver access: DXGI adapter enumeration plus the
//! D3D12 feature-support probes needed to fill in capability bits.
//!
//! `d3d12.dll` and `dxgi.dll` are loaded with `libloading` rather than linked,
//! so the crate builds and runs on hosts without them — the backend just
//! reports unavailable, the same contract every other IronAccelerator backend
//! honours. On non-Windows targets the candidate list is empty and every entry
//! point below is simply never resolved.
//!
//! COM is hand-rolled here: we declare the vtable layout for the three
//! interfaces we touch and call through it directly. Slots we never invoke are
//! typed as opaque pointers, which keeps the layout correct without pulling in
//! a full COM binding crate. Slot order is load-bearing — it mirrors the
//! inheritance chain (`IUnknown` → `IDXGIObject` → `IDXGIAdapter` → …) and
//! must not be reordered.
//!
//! `drop(sym)` on a `libloading::Symbol` ends the borrow on the `Library` so
//! the next `lib.get(...)` can proceed; the lint that fires on it is spurious
//! here for the same reason as in the Level Zero backend.

#![allow(clippy::drop_non_drop)] // see module docs above

use core::ffi::c_void;
use libloading::{Library, Symbol};
use once_cell::sync::OnceCell;

// ── COM primitives ─────────────────────────────────────────────────────────

pub type Hresult = i32;
const S_OK: Hresult = 0;
/// `DXGI_ERROR_NOT_FOUND` — returned by `EnumAdapters1` past the last adapter.
const DXGI_ERROR_NOT_FOUND: Hresult = 0x887A_0002u32 as i32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Guid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

/// `IID_IDXGIFactory1` — {770aae78-f26f-4dba-a829-253c83d1b387}
const IID_IDXGI_FACTORY1: Guid = Guid {
    data1: 0x770a_ae78,
    data2: 0xf26f,
    data3: 0x4dba,
    data4: [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87],
};

/// `IID_ID3D12Device` — {189819f1-1db6-4b57-be54-1821339b85f7}
const IID_ID3D12_DEVICE: Guid = Guid {
    data1: 0x1898_19f1,
    data2: 0x1db6,
    data3: 0x4b57,
    data4: [0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7],
};

#[repr(C)]
struct IUnknownVtbl {
    query_interface:
        unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> Hresult,
    add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
}

/// Release any COM pointer. Safe to call with null (no-op).
unsafe fn com_release(obj: *mut c_void) {
    if obj.is_null() {
        return;
    }
    let vtbl = *(obj as *mut *mut IUnknownVtbl);
    ((*vtbl).release)(obj);
}

// ── IDXGIFactory1 ──────────────────────────────────────────────────────────

/// `IUnknown` (3) → `IDXGIObject` (4) → `IDXGIFactory` (5) → `IDXGIFactory1` (2).
#[repr(C)]
struct IDxgiFactory1Vtbl {
    _query_interface: *const c_void,
    _add_ref: *const c_void,
    _release: *const c_void,
    _set_private_data: *const c_void,
    _set_private_data_interface: *const c_void,
    _get_private_data: *const c_void,
    _get_parent: *const c_void,
    _enum_adapters: *const c_void,
    _make_window_association: *const c_void,
    _get_window_association: *const c_void,
    _create_swap_chain: *const c_void,
    _create_software_adapter: *const c_void,
    enum_adapters1: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> Hresult,
    _is_current: *const c_void,
}

// ── IDXGIAdapter1 ──────────────────────────────────────────────────────────

/// `IUnknown` (3) → `IDXGIObject` (4) → `IDXGIAdapter` (3) → `IDXGIAdapter1` (1).
#[repr(C)]
struct IDxgiAdapter1Vtbl {
    _query_interface: *const c_void,
    _add_ref: *const c_void,
    _release: *const c_void,
    _set_private_data: *const c_void,
    _set_private_data_interface: *const c_void,
    _get_private_data: *const c_void,
    _get_parent: *const c_void,
    _enum_outputs: *const c_void,
    _get_desc: *const c_void,
    _check_interface_support: *const c_void,
    get_desc1: unsafe extern "system" fn(*mut c_void, *mut DxgiAdapterDesc1) -> Hresult,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Luid {
    pub low_part: u32,
    pub high_part: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DxgiAdapterDesc1 {
    pub description: [u16; 128],
    pub vendor_id: u32,
    pub device_id: u32,
    pub sub_sys_id: u32,
    pub revision: u32,
    pub dedicated_video_memory: usize,
    pub dedicated_system_memory: usize,
    pub shared_system_memory: usize,
    pub adapter_luid: Luid,
    pub flags: u32,
}

/// `DXGI_ADAPTER_FLAG_SOFTWARE` — WARP and other software rasterisers.
const DXGI_ADAPTER_FLAG_SOFTWARE: u32 = 2;

// ── ID3D12Device ───────────────────────────────────────────────────────────

/// `IUnknown` (3) → `ID3D12Object` (4) → `ID3D12Device`; we only need
/// `CheckFeatureSupport`, which is the 7th `ID3D12Device` slot (index 13).
#[repr(C)]
struct ID3d12DeviceVtbl {
    _query_interface: *const c_void,
    _add_ref: *const c_void,
    _release: *const c_void,
    _get_private_data: *const c_void,
    _set_private_data: *const c_void,
    _set_private_data_interface: *const c_void,
    _set_name: *const c_void,
    _get_node_count: *const c_void,
    _create_command_queue: *const c_void,
    _create_command_allocator: *const c_void,
    _create_graphics_pipeline_state: *const c_void,
    _create_compute_pipeline_state: *const c_void,
    _create_command_list: *const c_void,
    check_feature_support: unsafe extern "system" fn(*mut c_void, u32, *mut c_void, u32) -> Hresult,
}

// D3D12_FEATURE selectors.
const D3D12_FEATURE_D3D12_OPTIONS: u32 = 0;
const D3D12_FEATURE_FEATURE_LEVELS: u32 = 2;
const D3D12_FEATURE_D3D12_OPTIONS1: u32 = 8;
const D3D12_FEATURE_ARCHITECTURE1: u32 = 16;
const D3D12_FEATURE_D3D12_OPTIONS4: u32 = 23;

// D3D_FEATURE_LEVEL values, ascending.
pub const D3D_FEATURE_LEVEL_11_0: u32 = 0xb000;
pub const D3D_FEATURE_LEVEL_11_1: u32 = 0xb100;
pub const D3D_FEATURE_LEVEL_12_0: u32 = 0xc000;
pub const D3D_FEATURE_LEVEL_12_1: u32 = 0xc100;
pub const D3D_FEATURE_LEVEL_12_2: u32 = 0xc200;

const REQUESTED_LEVELS: [u32; 5] = [
    D3D_FEATURE_LEVEL_11_0,
    D3D_FEATURE_LEVEL_11_1,
    D3D_FEATURE_LEVEL_12_0,
    D3D_FEATURE_LEVEL_12_1,
    D3D_FEATURE_LEVEL_12_2,
];

#[repr(C)]
struct FeatureDataFeatureLevels {
    num_feature_levels: u32,
    p_feature_levels_requested: *const u32,
    max_supported_feature_level: u32,
}

#[repr(C)]
#[derive(Default)]
struct FeatureDataOptions {
    double_precision_float_shader_ops: i32,
    output_merger_logic_op: i32,
    /// `D3D12_SHADER_MIN_PRECISION_SUPPORT`: bit 0 = 10-bit, bit 1 = 16-bit.
    min_precision_support: i32,
    tiled_resources_tier: i32,
    resource_binding_tier: i32,
    ps_specified_stencil_ref_supported: i32,
    typed_uav_load_additional_formats: i32,
    rovs_supported: i32,
    conservative_rasterization_tier: i32,
    max_gpu_virtual_address_bits_per_resource: u32,
    standard_swizzle_64kb_supported: i32,
    cross_node_sharing_tier: i32,
    cross_adapter_row_major_texture_supported: i32,
    vp_and_rt_array_index_from_any_shader_feeding_rasterizer_supported_without_gs_emulation: i32,
    resource_heap_tier: i32,
}

const MIN_PRECISION_16_BIT: i32 = 2;

#[repr(C)]
#[derive(Default)]
struct FeatureDataOptions1 {
    wave_ops: i32,
    wave_lane_count_min: u32,
    wave_lane_count_max: u32,
    total_lane_count: u32,
    expanded_compute_resource_states: i32,
    int64_shader_ops: i32,
}

#[repr(C)]
#[derive(Default)]
struct FeatureDataOptions4 {
    msaa_64kb_aligned_texture_supported: i32,
    shared_resource_compatibility_tier: i32,
    native_16bit_shader_ops_supported: i32,
}

#[repr(C)]
#[derive(Default)]
struct FeatureDataArchitecture1 {
    node_index: u32,
    tile_based_renderer: i32,
    uma: i32,
    cache_coherent_uma: i32,
    isolated_mmu: i32,
}

// ── Loader ─────────────────────────────────────────────────────────────────

type CreateDxgiFactoryFn =
    unsafe extern "system" fn(riid: *const Guid, pp_factory: *mut *mut c_void) -> Hresult;
type CreateDxgiFactory2Fn = unsafe extern "system" fn(
    flags: u32,
    riid: *const Guid,
    pp_factory: *mut *mut c_void,
) -> Hresult;
type D3d12CreateDeviceFn = unsafe extern "system" fn(
    p_adapter: *mut c_void,
    minimum_feature_level: u32,
    riid: *const Guid,
    pp_device: *mut *mut c_void,
) -> Hresult;

pub struct Loaded {
    _dxgi: Library,
    _d3d12: Library,
    create_factory1: Option<CreateDxgiFactoryFn>,
    create_factory2: Option<CreateDxgiFactory2Fn>,
    create_device: D3d12CreateDeviceFn,
}

// SAFETY: the fields are plain function pointers into libraries that are kept
// loaded for the life of the process; none carry thread affinity.
unsafe impl Send for Loaded {}
unsafe impl Sync for Loaded {}

static LIB: OnceCell<Option<Loaded>> = OnceCell::new();

#[cfg(target_os = "windows")]
const DXGI_CANDIDATES: &[&str] = &["dxgi.dll"];
#[cfg(not(target_os = "windows"))]
const DXGI_CANDIDATES: &[&str] = &[];

#[cfg(target_os = "windows")]
const D3D12_CANDIDATES: &[&str] = &["d3d12.dll"];
#[cfg(not(target_os = "windows"))]
const D3D12_CANDIDATES: &[&str] = &[];

pub(crate) fn loaded() -> Option<&'static Loaded> {
    LIB.get_or_init(load).as_ref()
}

fn load() -> Option<Loaded> {
    let dxgi = DXGI_CANDIDATES
        .iter()
        .find_map(|n| unsafe { Library::new(*n) }.ok())?;
    let d3d12 = D3D12_CANDIDATES
        .iter()
        .find_map(|n| unsafe { Library::new(*n) }.ok())?;

    unsafe {
        // `CreateDXGIFactory2` is Windows 8.1+; keep `CreateDXGIFactory1` as a
        // fallback so a stripped-down host still enumerates.
        let create_factory2 = dxgi
            .get::<CreateDxgiFactory2Fn>(b"CreateDXGIFactory2\0")
            .ok()
            .map(|s| {
                let f = *s;
                drop(s);
                f
            });
        let create_factory1 = dxgi
            .get::<CreateDxgiFactoryFn>(b"CreateDXGIFactory1\0")
            .ok()
            .map(|s| {
                let f = *s;
                drop(s);
                f
            });
        if create_factory1.is_none() && create_factory2.is_none() {
            return None;
        }

        let dev: Symbol<D3d12CreateDeviceFn> = d3d12.get(b"D3D12CreateDevice\0").ok()?;
        let create_device = *dev;
        drop(dev);

        Some(Loaded {
            _dxgi: dxgi,
            _d3d12: d3d12,
            create_factory1,
            create_factory2,
            create_device,
        })
    }
}

impl Loaded {
    /// Create an `IDXGIFactory1`. Caller owns the returned pointer.
    unsafe fn create_factory(&self) -> Option<*mut c_void> {
        let mut factory: *mut c_void = core::ptr::null_mut();
        if let Some(f2) = self.create_factory2 {
            if f2(0, &IID_IDXGI_FACTORY1, &mut factory) == S_OK && !factory.is_null() {
                return Some(factory);
            }
            factory = core::ptr::null_mut();
        }
        let f1 = self.create_factory1?;
        if f1(&IID_IDXGI_FACTORY1, &mut factory) == S_OK && !factory.is_null() {
            return Some(factory);
        }
        None
    }
}

// ── Enumeration ────────────────────────────────────────────────────────────

/// One D3D12-capable adapter, described at enumeration time.
#[derive(Debug, Clone)]
pub struct EnumeratedAdapter {
    pub ordinal: u32,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub dedicated_video_memory: u64,
    pub shared_system_memory: u64,
    /// Highest `D3D_FEATURE_LEVEL` the device reports.
    pub feature_level: u32,
    /// Unified memory architecture — integrated GPUs report true.
    pub uma: bool,
    /// SM 6.0 wave intrinsics.
    pub wave_ops: bool,
    pub wave_lane_count_min: u32,
    pub wave_lane_count_max: u32,
    pub total_lane_count: u32,
    /// Native 16-bit shader ops (true FP16, not min-precision emulation).
    pub native_16bit_ops: bool,
    /// 16-bit min-precision support — FP16 storage without native math.
    pub min_precision_16bit: bool,
    pub int64_shader_ops: bool,
    /// Double-precision shader ops.
    pub fp64: bool,
}

pub fn is_available() -> bool {
    loaded().is_some() && !enumerate().is_empty()
}

/// Enumerate every hardware adapter that successfully creates an
/// `ID3D12Device`. Software adapters (WARP) are skipped — they are a
/// correctness reference, not an accelerator.
pub fn enumerate() -> Vec<EnumeratedAdapter> {
    let mut out = Vec::new();
    let Some(l) = loaded() else { return out };

    unsafe {
        let Some(factory) = l.create_factory() else {
            return out;
        };
        let fvtbl = *(factory as *mut *mut IDxgiFactory1Vtbl);

        let mut index = 0u32;
        let mut ordinal = 0u32;
        loop {
            let mut adapter: *mut c_void = core::ptr::null_mut();
            let hr = ((*fvtbl).enum_adapters1)(factory, index, &mut adapter);
            if hr == DXGI_ERROR_NOT_FOUND || hr != S_OK || adapter.is_null() {
                break;
            }
            index += 1;

            if let Some(desc) = describe_adapter(l, adapter, ordinal) {
                out.push(desc);
                ordinal += 1;
            }
            com_release(adapter);
        }
        com_release(factory);
    }
    out
}

/// Read an adapter's DXGI description and probe its D3D12 feature support.
/// Returns `None` for software adapters and for adapters that cannot create a
/// device at feature level 11_0.
unsafe fn describe_adapter(
    l: &Loaded,
    adapter: *mut c_void,
    ordinal: u32,
) -> Option<EnumeratedAdapter> {
    let avtbl = *(adapter as *mut *mut IDxgiAdapter1Vtbl);
    let mut desc: DxgiAdapterDesc1 = core::mem::zeroed();
    if ((*avtbl).get_desc1)(adapter, &mut desc) != S_OK {
        return None;
    }
    if desc.flags & DXGI_ADAPTER_FLAG_SOFTWARE != 0 {
        return None;
    }

    let mut device: *mut c_void = core::ptr::null_mut();
    if (l.create_device)(
        adapter,
        D3D_FEATURE_LEVEL_11_0,
        &IID_ID3D12_DEVICE,
        &mut device,
    ) != S_OK
        || device.is_null()
    {
        return None;
    }

    let dvtbl = *(device as *mut *mut ID3d12DeviceVtbl);
    let check = (*dvtbl).check_feature_support;

    let mut levels = FeatureDataFeatureLevels {
        num_feature_levels: REQUESTED_LEVELS.len() as u32,
        p_feature_levels_requested: REQUESTED_LEVELS.as_ptr(),
        max_supported_feature_level: D3D_FEATURE_LEVEL_11_0,
    };
    let feature_level = if check(
        device,
        D3D12_FEATURE_FEATURE_LEVELS,
        &mut levels as *mut _ as *mut c_void,
        core::mem::size_of::<FeatureDataFeatureLevels>() as u32,
    ) == S_OK
    {
        levels.max_supported_feature_level
    } else {
        D3D_FEATURE_LEVEL_11_0
    };

    let mut options = FeatureDataOptions::default();
    let has_options = check(
        device,
        D3D12_FEATURE_D3D12_OPTIONS,
        &mut options as *mut _ as *mut c_void,
        core::mem::size_of::<FeatureDataOptions>() as u32,
    ) == S_OK;
    let min_precision_16bit =
        has_options && options.min_precision_support & MIN_PRECISION_16_BIT != 0;
    let fp64 = has_options && options.double_precision_float_shader_ops != 0;

    let mut o1 = FeatureDataOptions1::default();
    let has_o1 = check(
        device,
        D3D12_FEATURE_D3D12_OPTIONS1,
        &mut o1 as *mut _ as *mut c_void,
        core::mem::size_of::<FeatureDataOptions1>() as u32,
    ) == S_OK;

    let mut o4 = FeatureDataOptions4::default();
    let native_16bit_ops = check(
        device,
        D3D12_FEATURE_D3D12_OPTIONS4,
        &mut o4 as *mut _ as *mut c_void,
        core::mem::size_of::<FeatureDataOptions4>() as u32,
    ) == S_OK
        && o4.native_16bit_shader_ops_supported != 0;

    let mut arch = FeatureDataArchitecture1::default();
    let uma = check(
        device,
        D3D12_FEATURE_ARCHITECTURE1,
        &mut arch as *mut _ as *mut c_void,
        core::mem::size_of::<FeatureDataArchitecture1>() as u32,
    ) == S_OK
        && arch.uma != 0;

    com_release(device);

    Some(EnumeratedAdapter {
        ordinal,
        name: utf16_to_string(&desc.description),
        vendor_id: desc.vendor_id,
        device_id: desc.device_id,
        dedicated_video_memory: desc.dedicated_video_memory as u64,
        shared_system_memory: desc.shared_system_memory as u64,
        feature_level,
        uma,
        wave_ops: has_o1 && o1.wave_ops != 0,
        wave_lane_count_min: if has_o1 { o1.wave_lane_count_min } else { 0 },
        wave_lane_count_max: if has_o1 { o1.wave_lane_count_max } else { 0 },
        total_lane_count: if has_o1 { o1.total_lane_count } else { 0 },
        native_16bit_ops,
        min_precision_16bit,
        int64_shader_ops: has_o1 && o1.int64_shader_ops != 0,
        fp64,
    })
}

fn utf16_to_string(raw: &[u16]) -> String {
    let len = raw.iter().position(|&c| c == 0).unwrap_or(raw.len());
    String::from_utf16_lossy(&raw[..len])
}

// ── Device handle ──────────────────────────────────────────────────────────

/// An owned `ID3D12Device`. Released on drop.
///
/// This is the handle a consumer builds command queues, root signatures, and
/// compute pipelines from. Building them is the consumer's job — this crate
/// stops at handing over a live device, the same way the CUDA backend stops at
/// handing over a context.
pub struct Device {
    raw: *mut c_void,
    pub ordinal: u32,
    pub feature_level: u32,
}

// SAFETY: `ID3D12Device` is free-threaded per the D3D12 documentation; all its
// methods are safe to call concurrently from multiple threads.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    /// Raw `ID3D12Device` pointer. Borrowed — do not release it.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { com_release(self.raw) }
    }
}

/// Create an `ID3D12Device` for the `ordinal`-th enumerated adapter.
pub fn open(ordinal: u32) -> Option<Device> {
    let l = loaded()?;
    unsafe {
        let factory = l.create_factory()?;
        let fvtbl = *(factory as *mut *mut IDxgiFactory1Vtbl);

        let mut index = 0u32;
        let mut seen = 0u32;
        let mut result = None;
        loop {
            let mut adapter: *mut c_void = core::ptr::null_mut();
            let hr = ((*fvtbl).enum_adapters1)(factory, index, &mut adapter);
            if hr == DXGI_ERROR_NOT_FOUND || hr != S_OK || adapter.is_null() {
                break;
            }
            index += 1;

            // Re-walk with the same filter enumerate() uses so ordinals agree.
            if let Some(desc) = describe_adapter(l, adapter, seen) {
                if seen == ordinal {
                    let mut device: *mut c_void = core::ptr::null_mut();
                    if (l.create_device)(
                        adapter,
                        D3D_FEATURE_LEVEL_11_0,
                        &IID_ID3D12_DEVICE,
                        &mut device,
                    ) == S_OK
                        && !device.is_null()
                    {
                        result = Some(Device {
                            raw: device,
                            ordinal,
                            feature_level: desc.feature_level,
                        });
                    }
                    com_release(adapter);
                    break;
                }
                seen += 1;
            }
            com_release(adapter);
        }
        com_release(factory);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_does_not_panic() {
        let _ = is_available();
        let _ = enumerate();
    }

    #[test]
    fn enumerated_ordinals_are_dense_and_ordered() {
        for (i, a) in enumerate().iter().enumerate() {
            assert_eq!(a.ordinal, i as u32);
        }
    }

    #[test]
    fn open_matches_enumerate() {
        let adapters = enumerate();
        // No D3D12 on this host is a valid outcome; the loop simply won't run.
        for a in &adapters {
            let dev = open(a.ordinal).expect("adapter enumerated but would not open");
            assert_eq!(dev.ordinal, a.ordinal);
            assert_eq!(dev.feature_level, a.feature_level);
            assert!(!dev.as_raw().is_null());
        }
    }
}
