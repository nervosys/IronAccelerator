//! Level Zero context + command queue / list scaffold.
//!
//! Resolves one driver + device by ordinal, creates a `ze_context` on
//! it, a compute `ze_command_queue`, and a default `ze_command_list`.
//! Higher layers will allocate memory and kernels on top; this module
//! is intentionally the minimum to reach a dispatchable state.

use core::ffi::c_void;

use crate::drv::{
    self, Loaded, ZeCommandListDesc, ZeCommandListHandle, ZeCommandQueueDesc,
    ZeCommandQueueHandle, ZeContextDesc, ZeContextHandle, ZeDeviceHandle,
    ZeDeviceMemAllocDesc, ZeDriverHandle, ZeGroupCount, ZeHostMemAllocDesc, ZeKernelDesc,
    ZeKernelHandle, ZeModuleDesc, ZeModuleHandle,
    ZE_COMMAND_QUEUE_MODE_DEFAULT, ZE_COMMAND_QUEUE_PRIORITY_NORMAL, ZE_MODULE_FORMAT_IL_SPIRV,
    ZE_RESULT_SUCCESS, ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
    ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, ZE_STRUCTURE_TYPE_CONTEXT_DESC,
    ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
    ZE_STRUCTURE_TYPE_KERNEL_DESC, ZE_STRUCTURE_TYPE_MODULE_DESC,
};

pub struct Context {
    l: &'static Loaded,
    pub driver: ZeDriverHandle,
    pub device: ZeDeviceHandle,
    pub context: ZeContextHandle,
    pub queue: ZeCommandQueueHandle,
    pub list: ZeCommandListHandle,
    /// Command-queue-group ordinal chosen for compute. Kept for later
    /// `zeCommandListAppendLaunchKernel` dispatches.
    pub queue_ordinal: u32,
}

impl Context {
    /// Walk drivers + devices, pick the `global_ordinal`-th device in
    /// the same order [`drv::enumerate`] produces, and bring up a
    /// context + compute queue + list on it.
    pub fn new(global_ordinal: u32) -> Option<Self> {
        let l = drv::loaded()?;
        unsafe {
            let (driver, device) = locate_device(l, global_ordinal)?;

            let mut context: ZeContextHandle = core::ptr::null_mut();
            let ctx_desc = ZeContextDesc {
                stype: ZE_STRUCTURE_TYPE_CONTEXT_DESC,
                p_next: core::ptr::null(),
                flags: 0,
            };
            if (l.ze_context_create)(driver, &ctx_desc, &mut context) != ZE_RESULT_SUCCESS {
                return None;
            }

            // Ordinal 0 is the default compute group across every Level
            // Zero driver shipped to date. A fuller implementation would
            // query `zeDeviceGetCommandQueueGroupProperties` and pick
            // the first group whose flags advertise COMPUTE.
            let queue_ordinal = 0u32;

            let mut queue: ZeCommandQueueHandle = core::ptr::null_mut();
            let q_desc = ZeCommandQueueDesc {
                stype: ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                p_next: core::ptr::null(),
                ordinal: queue_ordinal,
                index: 0,
                flags: 0,
                mode: ZE_COMMAND_QUEUE_MODE_DEFAULT,
                priority: ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
            };
            if (l.ze_command_queue_create)(context, device, &q_desc, &mut queue)
                != ZE_RESULT_SUCCESS
            {
                (l.ze_context_destroy)(context);
                return None;
            }

            let mut list: ZeCommandListHandle = core::ptr::null_mut();
            let l_desc = ZeCommandListDesc {
                stype: ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                p_next: core::ptr::null(),
                command_queue_group_ordinal: queue_ordinal,
                flags: 0,
            };
            if (l.ze_command_list_create)(context, device, &l_desc, &mut list)
                != ZE_RESULT_SUCCESS
            {
                (l.ze_command_queue_destroy)(queue);
                (l.ze_context_destroy)(context);
                return None;
            }

            Some(Context {
                l,
                driver,
                device,
                context,
                queue,
                list,
                queue_ordinal,
            })
        }
    }
}

impl Context {
    /// Allocate `size` bytes of device-local memory on this context's
    /// device. Returned pointer is an unmapped USM address valid for
    /// `zeCommandListAppendMemoryCopy` and kernel arg binding.
    pub fn alloc_device(&self, size: usize, alignment: usize) -> Option<DeviceBuffer> {
        let desc = ZeDeviceMemAllocDesc {
            stype: ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            p_next: core::ptr::null(),
            flags: 0,
            ordinal: self.queue_ordinal,
        };
        let mut ptr: *mut c_void = core::ptr::null_mut();
        unsafe {
            if (self.l.ze_mem_alloc_device)(
                self.context,
                &desc,
                size,
                alignment,
                self.device,
                &mut ptr,
            ) != ZE_RESULT_SUCCESS
            {
                return None;
            }
        }
        Some(DeviceBuffer {
            l: self.l,
            context: self.context,
            ptr,
            size,
        })
    }

    /// Allocate `size` bytes of shared USM (host + device accessible).
    pub fn alloc_shared(&self, size: usize, alignment: usize) -> Option<DeviceBuffer> {
        let ddesc = ZeDeviceMemAllocDesc {
            stype: ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            p_next: core::ptr::null(),
            flags: 0,
            ordinal: self.queue_ordinal,
        };
        let hdesc = ZeHostMemAllocDesc {
            stype: ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
            p_next: core::ptr::null(),
            flags: 0,
        };
        let mut ptr: *mut c_void = core::ptr::null_mut();
        unsafe {
            if (self.l.ze_mem_alloc_shared)(
                self.context,
                &ddesc,
                &hdesc,
                size,
                alignment,
                self.device,
                &mut ptr,
            ) != ZE_RESULT_SUCCESS
            {
                return None;
            }
        }
        Some(DeviceBuffer {
            l: self.l,
            context: self.context,
            ptr,
            size,
        })
    }

    /// Load a SPIR-V module onto the device.
    pub fn load_spirv(&self, spirv: &[u8]) -> Option<Module> {
        let desc = ZeModuleDesc {
            stype: ZE_STRUCTURE_TYPE_MODULE_DESC,
            p_next: core::ptr::null(),
            format: ZE_MODULE_FORMAT_IL_SPIRV,
            input_size: spirv.len(),
            p_input_module: spirv.as_ptr(),
            p_build_flags: core::ptr::null(),
            p_constants: core::ptr::null(),
        };
        let mut module: ZeModuleHandle = core::ptr::null_mut();
        unsafe {
            if (self.l.ze_module_create)(
                self.context,
                self.device,
                &desc,
                &mut module,
                core::ptr::null_mut(),
            ) != ZE_RESULT_SUCCESS
            {
                return None;
            }
        }
        Some(Module { l: self.l, module })
    }

    /// Append a kernel launch to `self.list`, close + execute the list,
    /// and wait for the queue to drain. One-shot pattern — higher layers
    /// will want to reuse command lists.
    pub fn launch(
        &self,
        kernel: &Kernel,
        group_count: [u32; 3],
    ) -> Result<(), u32> {
        let gc = ZeGroupCount {
            group_count_x: group_count[0],
            group_count_y: group_count[1],
            group_count_z: group_count[2],
        };
        unsafe {
            let r = (self.l.ze_command_list_append_launch_kernel)(
                self.list,
                kernel.kernel,
                &gc,
                core::ptr::null_mut(),
                0,
                core::ptr::null_mut(),
            );
            if r != ZE_RESULT_SUCCESS { return Err(r); }
            let r = (self.l.ze_command_list_close)(self.list);
            if r != ZE_RESULT_SUCCESS { return Err(r); }
            let lists = [self.list];
            let r = (self.l.ze_command_queue_execute_command_lists)(
                self.queue,
                1,
                lists.as_ptr(),
                core::ptr::null_mut(),
            );
            if r != ZE_RESULT_SUCCESS { return Err(r); }
            let r = (self.l.ze_command_queue_synchronize)(self.queue, u64::MAX);
            if r != ZE_RESULT_SUCCESS { return Err(r); }
            let _ = (self.l.ze_command_list_reset)(self.list);
        }
        Ok(())
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            (self.l.ze_command_list_destroy)(self.list);
            (self.l.ze_command_queue_destroy)(self.queue);
            (self.l.ze_context_destroy)(self.context);
        }
    }
}

pub struct DeviceBuffer {
    l: &'static Loaded,
    context: ZeContextHandle,
    pub ptr: *mut c_void,
    pub size: usize,
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe { (self.l.ze_mem_free)(self.context, self.ptr); }
    }
}

pub struct Module {
    l: &'static Loaded,
    pub module: ZeModuleHandle,
}

impl Module {
    /// Create a kernel object bound to `name` inside this module.
    pub fn kernel(&self, name: &str) -> Option<Kernel> {
        let cname = std::ffi::CString::new(name).ok()?;
        let desc = ZeKernelDesc {
            stype: ZE_STRUCTURE_TYPE_KERNEL_DESC,
            p_next: core::ptr::null(),
            flags: 0,
            p_kernel_name: cname.as_ptr(),
        };
        let mut k: ZeKernelHandle = core::ptr::null_mut();
        unsafe {
            if (self.l.ze_kernel_create)(self.module, &desc, &mut k) != ZE_RESULT_SUCCESS {
                return None;
            }
        }
        Some(Kernel { l: self.l, kernel: k })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe { (self.l.ze_module_destroy)(self.module); }
    }
}

pub struct Kernel {
    l: &'static Loaded,
    pub kernel: ZeKernelHandle,
}

impl Kernel {
    pub fn set_group_size(&self, gx: u32, gy: u32, gz: u32) -> Result<(), u32> {
        unsafe {
            match (self.l.ze_kernel_set_group_size)(self.kernel, gx, gy, gz) {
                ZE_RESULT_SUCCESS => Ok(()),
                e => Err(e),
            }
        }
    }

    /// Bind argument `index` to `value` (bytewise — for pointer args,
    /// pass `&buf.ptr`; for scalar args, pass `&scalar`).
    ///
    /// # Safety
    /// `value` must remain valid for the call's duration and match the
    /// kernel's declared argument layout.
    pub unsafe fn set_arg<T>(&self, index: u32, value: &T) -> Result<(), u32> {
        match (self.l.ze_kernel_set_argument_value)(
            self.kernel,
            index,
            core::mem::size_of::<T>(),
            value as *const T as *const c_void,
        ) {
            ZE_RESULT_SUCCESS => Ok(()),
            e => Err(e),
        }
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { (self.l.ze_kernel_destroy)(self.kernel); }
    }
}

unsafe fn locate_device(l: &Loaded, target: u32) -> Option<(ZeDriverHandle, ZeDeviceHandle)> {
    let mut driver_count: u32 = 0;
    if (l.ze_driver_get)(&mut driver_count, core::ptr::null_mut()) != ZE_RESULT_SUCCESS
        || driver_count == 0
    {
        return None;
    }
    let mut drivers = vec![core::ptr::null_mut::<c_void>(); driver_count as usize];
    if (l.ze_driver_get)(&mut driver_count, drivers.as_mut_ptr()) != ZE_RESULT_SUCCESS {
        return None;
    }
    let mut seen = 0u32;
    for driver in drivers.into_iter().take(driver_count as usize) {
        let mut dev_count: u32 = 0;
        if (l.ze_device_get)(driver, &mut dev_count, core::ptr::null_mut())
            != ZE_RESULT_SUCCESS
            || dev_count == 0
        {
            continue;
        }
        let mut devs = vec![core::ptr::null_mut::<c_void>(); dev_count as usize];
        if (l.ze_device_get)(driver, &mut dev_count, devs.as_mut_ptr()) != ZE_RESULT_SUCCESS {
            continue;
        }
        for dev in devs.into_iter().take(dev_count as usize) {
            if seen == target {
                return Some((driver, dev));
            }
            seen += 1;
        }
    }
    None
}
