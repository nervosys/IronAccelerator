//! Minimal `ze_loader` driver: `zeInit`, driver walk, device walk,
//! `zeDeviceGetProperties`. Enough for the planner; kernel launch lives
//! in higher layers.
//!
//! `drop(sym)` on a `libloading::Symbol` is intentional — it releases the
//! borrow on the Library so the next `lib.get(...)` can proceed. Symbol is
//! `Copy` and doesn't impl `Drop`, but the borrow lives in its lifetime
//! parameter, which `drop` does end. We silence the spurious lint module-wide.
//!
//! Everything is loaded via `libloading` so the crate compiles on hosts
//! without Level Zero installed; the backend simply reports unavailable.

#![allow(clippy::drop_non_drop)] // see module docs above

use core::ffi::c_void;
use libloading::{Library, Symbol};
use once_cell::sync::OnceCell;

// ── Raw FFI types ──────────────────────────────────────────────────────────

pub type ZeResult = u32;
pub const ZE_RESULT_SUCCESS: ZeResult = 0;

pub type ZeDriverHandle = *mut c_void;
pub type ZeDeviceHandle = *mut c_void;

pub const ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES: u32 = 0x1;

pub const ZE_DEVICE_TYPE_GPU: u32 = 1;
pub const ZE_DEVICE_TYPE_CPU: u32 = 2;
pub const ZE_DEVICE_TYPE_FPGA: u32 = 3;
pub const ZE_DEVICE_TYPE_MCA: u32 = 4;
pub const ZE_DEVICE_TYPE_VPU: u32 = 5;

pub const ZE_MAX_DEVICE_NAME: usize = 256;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ZeDeviceUuid {
    pub id: [u8; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ZeDeviceProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub type_: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub subdevice_id: u32,
    pub core_clock_rate: u32,
    pub max_mem_alloc_size: u64,
    pub max_hardware_contexts: u32,
    pub max_command_queue_priority: u32,
    pub num_threads_per_eu: u32,
    pub physical_eu_simd_width: u32,
    pub num_eus_per_subslice: u32,
    pub num_subslices_per_slice: u32,
    pub num_slices: u32,
    pub timer_resolution: u64,
    pub timestamp_valid_bits: u32,
    pub kernel_timestamp_valid_bits: u32,
    pub uuid: ZeDeviceUuid,
    pub name: [core::ffi::c_char; ZE_MAX_DEVICE_NAME],
}

pub type ZeContextHandle = *mut c_void;
pub type ZeCommandQueueHandle = *mut c_void;
pub type ZeCommandListHandle = *mut c_void;

pub const ZE_STRUCTURE_TYPE_CONTEXT_DESC: u32 = 0x2;
pub const ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC: u32 = 0x3;
pub const ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC: u32 = 0x4;
pub const ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC: u32 = 0xc;
pub const ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC: u32 = 0xd;
pub const ZE_STRUCTURE_TYPE_MODULE_DESC: u32 = 0xf;
pub const ZE_STRUCTURE_TYPE_KERNEL_DESC: u32 = 0x10;

pub const ZE_MODULE_FORMAT_IL_SPIRV: u32 = 0;
pub const ZE_MODULE_FORMAT_NATIVE: u32 = 1;

pub type ZeModuleHandle = *mut c_void;
pub type ZeKernelHandle = *mut c_void;

#[repr(C)]
pub struct ZeDeviceMemAllocDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub ordinal: u32,
}

#[repr(C)]
pub struct ZeHostMemAllocDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
}

#[repr(C)]
pub struct ZeModuleDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub format: u32,
    pub input_size: usize,
    pub p_input_module: *const u8,
    pub p_build_flags: *const core::ffi::c_char,
    pub p_constants: *const c_void,
}

#[repr(C)]
pub struct ZeKernelDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub p_kernel_name: *const core::ffi::c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ZeGroupCount {
    pub group_count_x: u32,
    pub group_count_y: u32,
    pub group_count_z: u32,
}

pub const ZE_COMMAND_QUEUE_MODE_DEFAULT: u32 = 0;
pub const ZE_COMMAND_QUEUE_PRIORITY_NORMAL: u32 = 0;

#[repr(C)]
pub struct ZeContextDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
}

#[repr(C)]
pub struct ZeCommandQueueDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub ordinal: u32,
    pub index: u32,
    pub flags: u32,
    pub mode: u32,
    pub priority: u32,
}

#[repr(C)]
pub struct ZeCommandListDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub command_queue_group_ordinal: u32,
    pub flags: u32,
}

type ZeInitFn = unsafe extern "C" fn(flags: u32) -> ZeResult;
pub type ZeContextCreateFn = unsafe extern "C" fn(
    h_driver: ZeDriverHandle,
    desc: *const ZeContextDesc,
    ph_context: *mut ZeContextHandle,
) -> ZeResult;
pub type ZeContextDestroyFn = unsafe extern "C" fn(h_context: ZeContextHandle) -> ZeResult;
pub type ZeCommandQueueCreateFn = unsafe extern "C" fn(
    h_context: ZeContextHandle,
    h_device: ZeDeviceHandle,
    desc: *const ZeCommandQueueDesc,
    ph_queue: *mut ZeCommandQueueHandle,
) -> ZeResult;
pub type ZeCommandQueueDestroyFn = unsafe extern "C" fn(h_queue: ZeCommandQueueHandle) -> ZeResult;
pub type ZeCommandListCreateFn = unsafe extern "C" fn(
    h_context: ZeContextHandle,
    h_device: ZeDeviceHandle,
    desc: *const ZeCommandListDesc,
    ph_list: *mut ZeCommandListHandle,
) -> ZeResult;
pub type ZeCommandListDestroyFn = unsafe extern "C" fn(h_list: ZeCommandListHandle) -> ZeResult;
type ZeDriverGetFn =
    unsafe extern "C" fn(p_count: *mut u32, ph_drivers: *mut ZeDriverHandle) -> ZeResult;
type ZeDeviceGetFn = unsafe extern "C" fn(
    h_driver: ZeDriverHandle,
    p_count: *mut u32,
    ph_devices: *mut ZeDeviceHandle,
) -> ZeResult;
type ZeDeviceGetPropertiesFn = unsafe extern "C" fn(
    h_device: ZeDeviceHandle,
    p_properties: *mut ZeDeviceProperties,
) -> ZeResult;

pub type ZeMemAllocDeviceFn = unsafe extern "C" fn(
    h_context: ZeContextHandle,
    device_desc: *const ZeDeviceMemAllocDesc,
    size: usize,
    alignment: usize,
    h_device: ZeDeviceHandle,
    pptr: *mut *mut c_void,
) -> ZeResult;
pub type ZeMemAllocSharedFn = unsafe extern "C" fn(
    h_context: ZeContextHandle,
    device_desc: *const ZeDeviceMemAllocDesc,
    host_desc: *const ZeHostMemAllocDesc,
    size: usize,
    alignment: usize,
    h_device: ZeDeviceHandle,
    pptr: *mut *mut c_void,
) -> ZeResult;
pub type ZeMemFreeFn =
    unsafe extern "C" fn(h_context: ZeContextHandle, ptr: *mut c_void) -> ZeResult;
pub type ZeModuleCreateFn = unsafe extern "C" fn(
    h_context: ZeContextHandle,
    h_device: ZeDeviceHandle,
    desc: *const ZeModuleDesc,
    ph_module: *mut ZeModuleHandle,
    ph_build_log: *mut *mut c_void,
) -> ZeResult;
pub type ZeModuleDestroyFn = unsafe extern "C" fn(h_module: ZeModuleHandle) -> ZeResult;
pub type ZeKernelCreateFn = unsafe extern "C" fn(
    h_module: ZeModuleHandle,
    desc: *const ZeKernelDesc,
    ph_kernel: *mut ZeKernelHandle,
) -> ZeResult;
pub type ZeKernelDestroyFn = unsafe extern "C" fn(h_kernel: ZeKernelHandle) -> ZeResult;
pub type ZeKernelSetGroupSizeFn =
    unsafe extern "C" fn(h_kernel: ZeKernelHandle, gx: u32, gy: u32, gz: u32) -> ZeResult;
pub type ZeKernelSetArgumentValueFn = unsafe extern "C" fn(
    h_kernel: ZeKernelHandle,
    arg_index: u32,
    arg_size: usize,
    p_arg_value: *const c_void,
) -> ZeResult;
pub type ZeCommandListAppendLaunchKernelFn = unsafe extern "C" fn(
    h_list: ZeCommandListHandle,
    h_kernel: ZeKernelHandle,
    p_launch_args: *const ZeGroupCount,
    h_signal_event: *mut c_void,
    num_wait_events: u32,
    ph_wait_events: *mut *mut c_void,
) -> ZeResult;
pub type ZeCommandListAppendMemoryCopyFn = unsafe extern "C" fn(
    h_list: ZeCommandListHandle,
    dstptr: *mut c_void,
    srcptr: *const c_void,
    size: usize,
    h_signal_event: *mut c_void,
    num_wait_events: u32,
    ph_wait_events: *mut *mut c_void,
) -> ZeResult;
pub type ZeCommandListCloseFn = unsafe extern "C" fn(h_list: ZeCommandListHandle) -> ZeResult;
pub type ZeCommandListResetFn = unsafe extern "C" fn(h_list: ZeCommandListHandle) -> ZeResult;
pub type ZeCommandQueueExecuteCommandListsFn = unsafe extern "C" fn(
    h_queue: ZeCommandQueueHandle,
    num_lists: u32,
    ph_lists: *const ZeCommandListHandle,
    h_fence: *mut c_void,
) -> ZeResult;
pub type ZeCommandQueueSynchronizeFn =
    unsafe extern "C" fn(h_queue: ZeCommandQueueHandle, timeout: u64) -> ZeResult;

// ── Loader ─────────────────────────────────────────────────────────────────

static LIB: OnceCell<Option<Loaded>> = OnceCell::new();

pub struct Loaded {
    _lib: Library,
    pub ze_driver_get: ZeDriverGetFn,
    pub ze_device_get: ZeDeviceGetFn,
    pub ze_device_get_properties: ZeDeviceGetPropertiesFn,
    pub ze_context_create: ZeContextCreateFn,
    pub ze_context_destroy: ZeContextDestroyFn,
    pub ze_command_queue_create: ZeCommandQueueCreateFn,
    pub ze_command_queue_destroy: ZeCommandQueueDestroyFn,
    pub ze_command_list_create: ZeCommandListCreateFn,
    pub ze_command_list_destroy: ZeCommandListDestroyFn,
    pub ze_mem_alloc_device: ZeMemAllocDeviceFn,
    pub ze_mem_alloc_shared: ZeMemAllocSharedFn,
    pub ze_mem_free: ZeMemFreeFn,
    pub ze_module_create: ZeModuleCreateFn,
    pub ze_module_destroy: ZeModuleDestroyFn,
    pub ze_kernel_create: ZeKernelCreateFn,
    pub ze_kernel_destroy: ZeKernelDestroyFn,
    pub ze_kernel_set_group_size: ZeKernelSetGroupSizeFn,
    pub ze_kernel_set_argument_value: ZeKernelSetArgumentValueFn,
    pub ze_command_list_append_launch_kernel: ZeCommandListAppendLaunchKernelFn,
    pub ze_command_list_append_memory_copy: ZeCommandListAppendMemoryCopyFn,
    pub ze_command_list_close: ZeCommandListCloseFn,
    pub ze_command_list_reset: ZeCommandListResetFn,
    pub ze_command_queue_execute_command_lists: ZeCommandQueueExecuteCommandListsFn,
    pub ze_command_queue_synchronize: ZeCommandQueueSynchronizeFn,
}

unsafe impl Send for Loaded {}
unsafe impl Sync for Loaded {}

#[cfg(target_os = "windows")]
const LIB_CANDIDATES: &[&str] = &["ze_loader.dll"];
#[cfg(any(target_os = "linux", target_os = "android"))]
const LIB_CANDIDATES: &[&str] = &["libze_loader.so.1", "libze_loader.so"];
#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "android")))]
const LIB_CANDIDATES: &[&str] = &[];

pub(crate) fn loaded() -> Option<&'static Loaded> {
    LIB.get_or_init(load).as_ref()
}

fn load() -> Option<Loaded> {
    for name in LIB_CANDIDATES {
        let lib = match unsafe { Library::new(*name) } {
            Ok(l) => l,
            Err(_) => continue,
        };
        unsafe {
            let init: Symbol<ZeInitFn> = match lib.get(b"zeInit\0") {
                Ok(s) => s,
                Err(_) => continue,
            };
            if init(0) != ZE_RESULT_SUCCESS {
                continue;
            }
            let dg: Symbol<ZeDriverGetFn> = match lib.get(b"zeDriverGet\0") {
                Ok(s) => s,
                Err(_) => continue,
            };
            let devg: Symbol<ZeDeviceGetFn> = match lib.get(b"zeDeviceGet\0") {
                Ok(s) => s,
                Err(_) => continue,
            };
            let devp: Symbol<ZeDeviceGetPropertiesFn> = match lib.get(b"zeDeviceGetProperties\0") {
                Ok(s) => s,
                Err(_) => continue,
            };
            let ze_driver_get = *dg;
            let ze_device_get = *devg;
            let ze_device_get_properties = *devp;
            drop(dg);
            drop(devg);
            drop(devp);
            drop(init);

            let ctx_create: Symbol<ZeContextCreateFn> = match lib.get(b"zeContextCreate\0") {
                Ok(s) => s,
                Err(_) => continue,
            };
            let ctx_destroy: Symbol<ZeContextDestroyFn> = match lib.get(b"zeContextDestroy\0") {
                Ok(s) => s,
                Err(_) => continue,
            };
            let q_create: Symbol<ZeCommandQueueCreateFn> = match lib.get(b"zeCommandQueueCreate\0")
            {
                Ok(s) => s,
                Err(_) => continue,
            };
            let q_destroy: Symbol<ZeCommandQueueDestroyFn> =
                match lib.get(b"zeCommandQueueDestroy\0") {
                    Ok(s) => s,
                    Err(_) => continue,
                };
            let l_create: Symbol<ZeCommandListCreateFn> = match lib.get(b"zeCommandListCreate\0") {
                Ok(s) => s,
                Err(_) => continue,
            };
            let l_destroy: Symbol<ZeCommandListDestroyFn> = match lib.get(b"zeCommandListDestroy\0")
            {
                Ok(s) => s,
                Err(_) => continue,
            };

            let ze_context_create = *ctx_create;
            let ze_context_destroy = *ctx_destroy;
            let ze_command_queue_create = *q_create;
            let ze_command_queue_destroy = *q_destroy;
            let ze_command_list_create = *l_create;
            let ze_command_list_destroy = *l_destroy;
            drop(ctx_create);
            drop(ctx_destroy);
            drop(q_create);
            drop(q_destroy);
            drop(l_create);
            drop(l_destroy);

            macro_rules! sym {
                ($ty:ty, $name:literal) => {{
                    let s: Symbol<$ty> = match lib.get($name) {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                    let f = *s;
                    drop(s);
                    f
                }};
            }
            let ze_mem_alloc_device = sym!(ZeMemAllocDeviceFn, b"zeMemAllocDevice\0");
            let ze_mem_alloc_shared = sym!(ZeMemAllocSharedFn, b"zeMemAllocShared\0");
            let ze_mem_free = sym!(ZeMemFreeFn, b"zeMemFree\0");
            let ze_module_create = sym!(ZeModuleCreateFn, b"zeModuleCreate\0");
            let ze_module_destroy = sym!(ZeModuleDestroyFn, b"zeModuleDestroy\0");
            let ze_kernel_create = sym!(ZeKernelCreateFn, b"zeKernelCreate\0");
            let ze_kernel_destroy = sym!(ZeKernelDestroyFn, b"zeKernelDestroy\0");
            let ze_kernel_set_group_size = sym!(ZeKernelSetGroupSizeFn, b"zeKernelSetGroupSize\0");
            let ze_kernel_set_argument_value =
                sym!(ZeKernelSetArgumentValueFn, b"zeKernelSetArgumentValue\0");
            let ze_command_list_append_launch_kernel = sym!(
                ZeCommandListAppendLaunchKernelFn,
                b"zeCommandListAppendLaunchKernel\0"
            );
            let ze_command_list_append_memory_copy = sym!(
                ZeCommandListAppendMemoryCopyFn,
                b"zeCommandListAppendMemoryCopy\0"
            );
            let ze_command_list_close = sym!(ZeCommandListCloseFn, b"zeCommandListClose\0");
            let ze_command_list_reset = sym!(ZeCommandListResetFn, b"zeCommandListReset\0");
            let ze_command_queue_execute_command_lists = sym!(
                ZeCommandQueueExecuteCommandListsFn,
                b"zeCommandQueueExecuteCommandLists\0"
            );
            let ze_command_queue_synchronize =
                sym!(ZeCommandQueueSynchronizeFn, b"zeCommandQueueSynchronize\0");

            return Some(Loaded {
                _lib: lib,
                ze_driver_get,
                ze_device_get,
                ze_device_get_properties,
                ze_context_create,
                ze_context_destroy,
                ze_command_queue_create,
                ze_command_queue_destroy,
                ze_command_list_create,
                ze_command_list_destroy,
                ze_mem_alloc_device,
                ze_mem_alloc_shared,
                ze_mem_free,
                ze_module_create,
                ze_module_destroy,
                ze_kernel_create,
                ze_kernel_destroy,
                ze_kernel_set_group_size,
                ze_kernel_set_argument_value,
                ze_command_list_append_launch_kernel,
                ze_command_list_append_memory_copy,
                ze_command_list_close,
                ze_command_list_reset,
                ze_command_queue_execute_command_lists,
                ze_command_queue_synchronize,
            });
        }
    }
    None
}

// ── Enumeration ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct EnumeratedDevice {
    pub ordinal: u32,
    pub type_: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub name: String,
    pub core_clock_khz: u32,
    pub max_mem_alloc_size: u64,
    pub num_slices: u32,
    pub num_subslices_per_slice: u32,
    pub num_eus_per_subslice: u32,
}

pub fn is_available() -> bool {
    loaded().is_some()
}

pub fn enumerate() -> Vec<EnumeratedDevice> {
    let Some(l) = loaded() else { return Vec::new() };
    let mut out = Vec::new();
    let mut ordinal = 0u32;
    unsafe {
        let mut driver_count: u32 = 0;
        if (l.ze_driver_get)(&mut driver_count, core::ptr::null_mut()) != ZE_RESULT_SUCCESS
            || driver_count == 0
        {
            return out;
        }
        let mut drivers = vec![core::ptr::null_mut::<c_void>(); driver_count as usize];
        if (l.ze_driver_get)(&mut driver_count, drivers.as_mut_ptr()) != ZE_RESULT_SUCCESS {
            return out;
        }
        for driver in drivers.into_iter().take(driver_count as usize) {
            let mut dev_count: u32 = 0;
            if (l.ze_device_get)(driver, &mut dev_count, core::ptr::null_mut()) != ZE_RESULT_SUCCESS
                || dev_count == 0
            {
                continue;
            }
            let mut devs = vec![core::ptr::null_mut::<c_void>(); dev_count as usize];
            if (l.ze_device_get)(driver, &mut dev_count, devs.as_mut_ptr()) != ZE_RESULT_SUCCESS {
                continue;
            }
            for dev in devs.into_iter().take(dev_count as usize) {
                let mut props: ZeDeviceProperties = core::mem::zeroed();
                props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
                if (l.ze_device_get_properties)(dev, &mut props) != ZE_RESULT_SUCCESS {
                    continue;
                }
                out.push(EnumeratedDevice {
                    ordinal,
                    type_: props.type_,
                    vendor_id: props.vendor_id,
                    device_id: props.device_id,
                    name: c_name_to_string(&props.name),
                    core_clock_khz: props.core_clock_rate.saturating_mul(1000),
                    max_mem_alloc_size: props.max_mem_alloc_size,
                    num_slices: props.num_slices,
                    num_subslices_per_slice: props.num_subslices_per_slice,
                    num_eus_per_subslice: props.num_eus_per_subslice,
                });
                ordinal += 1;
            }
        }
    }
    out
}

fn c_name_to_string(raw: &[core::ffi::c_char]) -> String {
    let bytes: Vec<u8> = raw
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_does_not_panic() {
        let _ = is_available();
        let _ = enumerate();
    }
}
