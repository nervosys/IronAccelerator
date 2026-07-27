//! Print every D3D12 adapter this host exposes, with the probed feature bits.
//!
//! ```text
//! cargo run -p ironaccelerator-dx12 --example list_adapters
//! ```

use ironaccelerator_core::Backend;
use ironaccelerator_dx12::{drv, DX12_BACKEND};

fn main() {
    if !DX12_BACKEND.is_available() {
        println!("d3d12: unavailable on this host");
        return;
    }

    for a in drv::enumerate() {
        println!("[{}] {}", a.ordinal, a.name);
        println!(
            "     vendor={:04x} device={:04x} fl={:#x} uma={}",
            a.vendor_id, a.device_id, a.feature_level, a.uma
        );
        println!(
            "     vram={} MiB shared={} MiB",
            a.dedicated_video_memory / (1024 * 1024),
            a.shared_system_memory / (1024 * 1024)
        );
        println!(
            "     wave_ops={} lanes={}..{} total={} fp16={} fp64={} int64={}",
            a.wave_ops,
            a.wave_lane_count_min,
            a.wave_lane_count_max,
            a.total_lane_count,
            a.native_16bit_ops,
            a.fp64,
            a.int64_shader_ops
        );
        match DX12_BACKEND.capabilities(a.ordinal) {
            Ok(f) => println!("     flags={f:?}"),
            Err(e) => println!("     flags: {e}"),
        }
    }
}
