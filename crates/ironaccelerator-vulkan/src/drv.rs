//! Vulkan driver wrappers — instance + physical-device enumeration.
//!
//! We hold the `ash::Entry` in a process-wide `OnceCell` so repeated
//! enumeration doesn't re-dlopen `libvulkan`. An `Instance` is created once
//! with `VK_API_VERSION_1_3`, no layers and no extensions beyond what's
//! needed for enumeration.

use ash::{vk, Entry, Instance};
use once_cell::sync::OnceCell;

static ENTRY: OnceCell<Option<Entry>> = OnceCell::new();
static INSTANCE: OnceCell<Option<Instance>> = OnceCell::new();

/// Process-wide `ash::Entry` + `ash::Instance`. Both are cached; both live
/// for the rest of the process.
pub fn own_instance() -> Option<(&'static Entry, &'static Instance)> {
    Some((entry()?, instance()?))
}

fn entry() -> Option<&'static Entry> {
    ENTRY.get_or_init(|| unsafe { Entry::load().ok() }).as_ref()
}

fn instance() -> Option<&'static Instance> {
    INSTANCE
        .get_or_init(|| {
            let entry = entry()?;
            let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));
            let ci = vk::InstanceCreateInfo::default().application_info(&app);
            unsafe { entry.create_instance(&ci, None).ok() }
        })
        .as_ref()
}

/// Minimal info about one physical device, pulled once at enumeration time.
#[derive(Debug, Clone)]
pub struct PhysicalDevice {
    pub ordinal: u32,
    pub name: String,
    pub vendor_id: u32,
    pub device_type: vk::PhysicalDeviceType,
    pub api_version: u32,
    pub driver_version: u32,
    pub heap_size_bytes: u64,
    pub compute_queue_family: Option<u32>,
    pub subgroup_size: u32,
    pub shader_int8: bool,
    pub shader_int16: bool,
    pub shader_float16: bool,
    pub cooperative_matrix: bool,
}

pub fn enumerate() -> Vec<PhysicalDevice> {
    let Some(inst) = instance() else {
        return Vec::new();
    };
    let raw = match unsafe { inst.enumerate_physical_devices() } {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    raw.into_iter()
        .enumerate()
        .map(|(i, pd)| describe(inst, pd, i as u32))
        .collect()
}

fn describe(inst: &Instance, pd: vk::PhysicalDevice, ordinal: u32) -> PhysicalDevice {
    let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup_props);
    unsafe { inst.get_physical_device_properties2(pd, &mut props2) };

    let mut f16i8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let (shader_int16, shader_int8, shader_float16) = {
        let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f16i8);
        unsafe { inst.get_physical_device_features2(pd, &mut features2) };
        (
            features2.features.shader_int16 != 0,
            f16i8.shader_int8 != 0,
            f16i8.shader_float16 != 0,
        )
    };

    let props = props2.properties;
    let name = raw_name_to_string(&props.device_name);

    let mem = unsafe { inst.get_physical_device_memory_properties(pd) };
    let heap_size_bytes = mem.memory_heaps[..mem.memory_heap_count as usize]
        .iter()
        .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|h| h.size)
        .max()
        .unwrap_or(0);

    let families = unsafe { inst.get_physical_device_queue_family_properties(pd) };
    let compute_queue_family = families
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .map(|i| i as u32);

    // Cooperative matrix is a KHR extension; probe by name.
    let exts = unsafe {
        inst.enumerate_device_extension_properties(pd)
            .unwrap_or_default()
    };
    let cooperative_matrix = exts
        .iter()
        .any(|e| raw_name_to_string(&e.extension_name) == "VK_KHR_cooperative_matrix");

    PhysicalDevice {
        ordinal,
        name,
        vendor_id: props.vendor_id,
        device_type: props.device_type,
        api_version: props.api_version,
        driver_version: props.driver_version,
        heap_size_bytes,
        compute_queue_family,
        subgroup_size: subgroup_props.subgroup_size,
        shader_int8,
        shader_int16,
        shader_float16,
        cooperative_matrix,
    }
}

fn raw_name_to_string(raw: &[core::ffi::c_char]) -> String {
    let bytes: Vec<u8> = raw
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}
