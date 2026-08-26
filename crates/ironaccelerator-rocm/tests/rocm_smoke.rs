//! Live-GPU smoke test for the ROCm depth path: HIPRTC compile, module load,
//! kernel launch, cache identity, and the `MemPool` upload/download round-trip.
//!
//! Skips cleanly (not fails) when no HIP device is present, so it is a no-op on
//! GPU-less CI and the real validation on a self-hosted AMD runner. The gate is
//! deliberately double: a HIP device **and** `IRON_RUN_GPU_TESTS=1`, matching
//! the opt-in the `hardware.yml` self-hosted workflow sets — so it stays inert
//! if a developer merely happens to have ROCm installed locally.
//!
//! This is the test that turns the ROCm depth from "compiles clean" into
//! "validated" the moment an AMD GPU is registered as a runner; until then it
//! reports skipped, as it does on this Windows + NVIDIA host.

use std::ffi::c_void;
use std::sync::Arc;

use ironaccelerator_rocm::drv::{Device, LaunchCfg, Stream};
use ironaccelerator_rocm::kernel::{get_or_compile, CompileOptions};
use ironaccelerator_rocm::pool::MemPool;

fn gpu_enabled() -> bool {
    std::env::var("IRON_RUN_GPU_TESTS").is_ok()
        && ironaccelerator_rocm::drv::is_available()
        && Device::count().unwrap_or(0) > 0
}

const SRC: &str = r#"
extern "C" __global__ void double_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] * 2.0f;
}
"#;

#[test]
fn hiprtc_compile_launch_and_pool_roundtrip() {
    if !gpu_enabled() {
        eprintln!("rocm_smoke: no HIP device or IRON_RUN_GPU_TESTS unset — skipping");
        return;
    }

    let dev = Device::open(0).expect("open device 0");
    dev.bind().unwrap();
    let stream = Stream::new(dev.clone()).expect("stream");

    // MemPool: allocate, upload, and confirm a plain round-trip first.
    const N: usize = 1024;
    let pool = MemPool::new(stream.clone());
    let mut buf = pool.alloc::<f32>(N).expect("pool alloc");
    let input: Vec<f32> = (0..N).map(|i| i as f32).collect();
    buf.copy_from_host(&input).expect("h2d");

    // Compile a doubling kernel through the HIPRTC cache and launch it.
    let k = get_or_compile(&dev, SRC, "double_kernel", &CompileOptions::default())
        .expect("hiprtc compile");
    let mut d_ptr = buf.device_ptr();
    let mut n: i32 = N as i32;
    let mut argv: [*mut c_void; 2] = [
        &mut d_ptr as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (N as u32).div_ceil(block);
    unsafe {
        k.function
            .launch_raw(LaunchCfg::linear(grid, block), &stream, argv.as_mut_ptr())
            .expect("launch");
    }
    stream.synchronize().unwrap();

    let mut out = vec![0f32; N];
    buf.copy_to_host(&mut out).expect("d2h");
    for (i, v) in out.iter().enumerate() {
        assert!(
            (v - (i as f32 * 2.0)).abs() < 1e-6,
            "element {i}: got {v}, want {}",
            i as f32 * 2.0
        );
    }

    // Cache identity: an identical key returns the same module (and the
    // double-checked insert holds under a second compile).
    let k2 = get_or_compile(&dev, SRC, "double_kernel", &CompileOptions::default()).unwrap();
    assert!(
        Arc::ptr_eq(&k.module, &k2.module),
        "kernel cache miss on identical key"
    );

    eprintln!("rocm_smoke: HIPRTC compile + launch + MemPool round-trip over {N} f32 — OK");
}
