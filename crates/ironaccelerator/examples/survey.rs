//! Survey every device reachable from this process.
//!
//! ```text
//! cargo run -p ironaccelerator --example survey --features all
//! ```

use ironaccelerator::prelude::*;

fn main() {
    let rt = ironaccelerator::init();

    let backends = rt.available_backends();
    if backends.is_empty() {
        println!("no backend located its runtime libraries on this host");
        return;
    }
    println!(
        "available backends: {}",
        backends
            .iter()
            .map(|b| b.name())
            .collect::<Vec<_>>()
            .join(", ")
    );

    for d in rt.devices() {
        println!("\n[{}:{}] {}", d.id.backend.name(), d.id.ordinal, d.name);
        println!(
            "  vendor={:?} arch={} tier={:?}",
            d.vendor, d.arch, d.capability.tier
        );
        if d.total_memory_bytes > 0 {
            println!("  memory={} MiB", d.total_memory_bytes / (1024 * 1024));
        }
        println!("  flags={:?}", d.capability.flags);
    }

    // Pure hardware filtering — no workload vocabulary involved.
    let fp16 = rt.devices_with(CapabilityFlags::FP16);
    println!("\n{} device(s) report FP16", fp16.len());
}
