//! WebGPU driver wrappers — instance + adapter enumeration via `wgpu`.
//!
//! On native we iterate every available adapter across all compiled-in
//! backends. On WASM the browser hands out exactly one adapter; hosts that
//! want to preselect can stash a ready `wgpu::Device` via [`bind_device`].

use once_cell::sync::OnceCell;
use std::sync::Mutex;

static INSTANCE: OnceCell<wgpu::Instance> = OnceCell::new();
static BOUND: OnceCell<Mutex<Option<BoundDevice>>> = OnceCell::new();

pub struct BoundDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: AdapterInfo,
}

#[derive(Debug, Clone)]
pub struct AdapterInfo {
    pub name: String,
    pub vendor: u32,
    pub device_id: u32,
    pub backend: wgpu::Backend,
    pub device_type: wgpu::DeviceType,
    pub driver: String,
    pub driver_info: String,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u32,
    pub max_compute_workgroup_size_x: u32,
    pub subgroup_support: bool,
}

fn instance() -> &'static wgpu::Instance {
    INSTANCE.get_or_init(|| {
        wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        })
    })
}

pub fn enumerate() -> Vec<AdapterInfo> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        instance()
            .enumerate_adapters(wgpu::Backends::all())
            .into_iter()
            .map(|a| describe(&a))
            .collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        BOUND
            .get()
            .and_then(|m| {
                m.lock()
                    .ok()
                    .and_then(|g| g.as_ref().map(|b| b.info.clone()))
            })
            .map(|i| vec![i])
            .unwrap_or_default()
    }
}

pub(crate) fn describe(a: &wgpu::Adapter) -> AdapterInfo {
    let info = a.get_info();
    let limits = a.limits();
    let features = a.features();
    AdapterInfo {
        name: info.name,
        vendor: info.vendor,
        device_id: info.device,
        backend: info.backend,
        device_type: info.device_type,
        driver: info.driver,
        driver_info: info.driver_info,
        max_buffer_size: limits.max_buffer_size,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
        subgroup_support: features.contains(wgpu::Features::SUBGROUP),
    }
}

/// WASM / preselected path: stash an already-requested device + queue so the
/// backend can enumerate without redoing adapter negotiation.
pub fn bind_device(device: wgpu::Device, queue: wgpu::Queue, info: AdapterInfo) {
    let slot = BOUND.get_or_init(|| Mutex::new(None));
    *slot.lock().unwrap() = Some(BoundDevice {
        device,
        queue,
        info,
    });
}
