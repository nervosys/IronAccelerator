//! End-to-end SAXPY using the cudarc-shaped surface.
//!
//! Run with:
//!     cargo run --release -p ironaccelerator-cuda --example saxpy_cudarc_style
//!
//! Demonstrates the canonical port shape:
//!
//!     use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync};   // before
//!     use cudarc::nvrtc::compile_ptx;
//!
//!     use ironaccelerator_cuda::cudarc_compat::*;                 // after
//!
//! Everything below is identical to what you'd write against cudarc 0.19.

use ironaccelerator_cuda::cudarc_compat::*;

const SAXPY_SRC: &str = r#"
extern "C" __global__
void saxpy(const float* x, float* y, float a, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Open device 0 and grab its default stream.
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no CUDA device available: {e}");
            return Ok(());
        }
    };
    let stream = device.default_stream();
    println!("device: {}", device.name()?);
    let (maj, min) = device.compute_capability()?;
    println!("compute capability: {maj}.{min}");
    let (free, total) = device.mem_get_info()?;
    println!(
        "memory: {} MiB free / {} MiB total",
        free >> 20,
        total >> 20
    );

    // 2. Compile the kernel via NVRTC. Cached on disk after the first run.
    let ptx = compile_ptx(SAXPY_SRC)?;
    let module = ironaccelerator_cuda::drv::Module::load(device.raw().clone(), &ptx)?;
    let saxpy = module.function("saxpy")?;

    // 3. Push host inputs to the GPU, allocate the output.
    const N: usize = 1 << 20;
    let a: f32 = 2.5;
    let x_host: Vec<f32> = (0..N).map(|i| i as f32 * 0.001).collect();
    let y_host: Vec<f32> = (0..N).map(|i| i as f32 * 0.01).collect();

    let x: CudaSlice<f32> = stream.htod_copy(x_host.clone())?;
    let mut y: CudaSlice<f32> = stream.htod_copy(y_host.clone())?;

    // 4. Launch — `LaunchAsync` mirrors cudarc's tuple-arg ergonomics, with
    //    the stream passed explicitly.
    let cfg = ironaccelerator_cuda::drv::LaunchCfg::for_elements(N as u32, 256);
    saxpy.launch_async(cfg, &stream, (x.view(), y.view_mut(), a, N as u32))?;

    // 5. Pull the result back. `dtoh_sync_copy` syncs the stream then copies.
    let out: Vec<f32> = stream.dtoh_sync_copy(&y)?;

    // 6. Verify against the CPU reference. FP32 accumulates rounding, so we
    //    check relative error against the running magnitude rather than a
    //    fixed absolute bound.
    let mut max_rel: f32 = 0.0;
    for i in 0..N {
        let expected = a * x_host[i] + y_host[i];
        let denom = expected.abs().max(1.0);
        max_rel = max_rel.max((out[i] - expected).abs() / denom);
    }
    println!("SAXPY over {N} elements: max relative error = {max_rel:.3e}");
    assert!(max_rel < 1e-5, "SAXPY accuracy regression: {max_rel:e}");
    println!("OK");
    Ok(())
}
