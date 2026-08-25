//! Print every Vulkan physical device this host exposes, with the probed
//! compute-feature bits.
//!
//! ```text
//! cargo run -p ironaccelerator-vulkan --example list_devices
//! ```

use ironaccelerator_core::Backend;
use ironaccelerator_vulkan::{drv, VULKAN_BACKEND};

fn main() {
    if !VULKAN_BACKEND.is_available() {
        println!("vulkan: unavailable on this host (no ICD or no device)");
        return;
    }

    for pd in drv::enumerate() {
        println!("[{}] {}", pd.ordinal, pd.name);
        // Decode `VK_MAKE_API_VERSION`: major in bits 22..29, minor in 12..21.
        let (major, minor) = (
            (pd.api_version >> 22) & 0x7f,
            (pd.api_version >> 12) & 0x3ff,
        );
        println!(
            "     vendor={:04x} type={:?} api=vk{major}.{minor}",
            pd.vendor_id, pd.device_type,
        );
        println!(
            "     vram={} MiB compute_queue={:?} subgroup={}",
            pd.heap_size_bytes / (1024 * 1024),
            pd.compute_queue_family,
            pd.subgroup_size,
        );
        println!(
            "     fp16={} int16={} int8={} coop_matrix={}",
            pd.shader_float16, pd.shader_int16, pd.shader_int8, pd.cooperative_matrix,
        );
        match VULKAN_BACKEND.capabilities(pd.ordinal) {
            Ok(f) => println!("     flags={f:?}"),
            Err(e) => println!("     flags: {e}"),
        }
    }
}
