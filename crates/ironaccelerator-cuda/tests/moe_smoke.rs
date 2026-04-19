//! Live-GPU MoE smoke test — exercises `FlashMoePlan::execute` end-to-end
//! on the real GPU for both FP16 and BF16 activation paths.
//!
//! Skipped cleanly on GPU-less runners.

use ironaccelerator_cuda::blas::BlasLt;
use ironaccelerator_cuda::drv::{Device, DeviceBuf, Stream};
use ironaccelerator_cuda::moe::{
    FlashMoePlan, MoeActivation, MoeDType, MoeParams, MoeScratch,
};

fn have_cuda() -> bool {
    matches!(Device::count(), Ok(n) if n > 0)
}

/// Pack a slice of f32s into f16 bit patterns (u16) on the host.
fn f32_to_f16_bits(xs: &[f32]) -> Vec<u16> {
    xs.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect()
}
fn f32_to_bf16_bits(xs: &[f32]) -> Vec<u16> {
    xs.iter().map(|&v| half::bf16::from_f32(v).to_bits()).collect()
}
fn f16_bits_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect()
}
fn bf16_bits_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&b| half::bf16::from_bits(b).to_f32()).collect()
}

fn run_moe_once(dtype: MoeDType) {
    if !have_cuda() {
        eprintln!("moe_smoke: no CUDA — skipping");
        return;
    }

    // Tiny but non-trivial shape — small enough to finish in a second, big
    // enough that every kernel path (softmax-topk / count / scan / permute /
    // silu / combine / router-GEMM / expert-GEMM) sees real work.
    const T: u32 = 32;
    const H: u32 = 64;
    const I: u32 = 128;
    const E: u32 = 4;
    const K: u32 = 2;

    let params = MoeParams {
        num_tokens: T, hidden: H, inter: I,
        num_experts: E, top_k: K,
        dtype, activation: MoeActivation::Silu,
    };

    let dev = Device::open(0).unwrap();
    dev.bind().unwrap();
    let stream = Stream::new(dev.clone()).unwrap();
    let blaslt = BlasLt::new(dev.clone()).unwrap();

    let plan = FlashMoePlan::new(dev.clone(), blaslt, params, 4 << 20)
        .expect("FlashMoePlan::new");
    let mut scratch = MoeScratch::new(stream.clone(), &params, 64 << 20).unwrap();

    // Build deterministic small inputs on host, push to GPU.
    let mk = |seed: u32, n: usize, scale: f32| -> Vec<f32> {
        (0..n).map(|i| {
            let x = ((seed.wrapping_mul(2654435761).wrapping_add(i as u32)) % 10007) as f32;
            (x / 10007.0 - 0.5) * 2.0 * scale
        }).collect()
    };
    let x_h       = mk(1, (T*H) as usize,     1.0);
    let w_gate_h  = mk(2, (H*E) as usize,     0.1);
    let w_up_h    = mk(3, (E*H*I) as usize,   0.05);
    let w_down_h  = mk(4, (E*I*H) as usize,   0.05);

    let (x_bits, wg_bits, wu_bits, wd_bits) = match dtype {
        MoeDType::F16 => (
            f32_to_f16_bits(&x_h),
            f32_to_f16_bits(&w_gate_h),
            f32_to_f16_bits(&w_up_h),
            f32_to_f16_bits(&w_down_h),
        ),
        MoeDType::Bf16 => (
            f32_to_bf16_bits(&x_h),
            f32_to_bf16_bits(&w_gate_h),
            f32_to_bf16_bits(&w_up_h),
            f32_to_bf16_bits(&w_down_h),
        ),
    };

    let mut x:     DeviceBuf<u16> = DeviceBuf::alloc(stream.clone(), x_bits.len()).unwrap();
    let mut wg:    DeviceBuf<u16> = DeviceBuf::alloc(stream.clone(), wg_bits.len()).unwrap();
    let mut wu:    DeviceBuf<u16> = DeviceBuf::alloc(stream.clone(), wu_bits.len()).unwrap();
    let mut wd:    DeviceBuf<u16> = DeviceBuf::alloc(stream.clone(), wd_bits.len()).unwrap();
    let y:         DeviceBuf<u16> = DeviceBuf::alloc_zeros(stream.clone(), (T*H) as usize).unwrap();

    x.copy_from_host(&x_bits).unwrap();
    wg.copy_from_host(&wg_bits).unwrap();
    wu.copy_from_host(&wu_bits).unwrap();
    wd.copy_from_host(&wd_bits).unwrap();

    unsafe {
        plan.execute(
            &stream,
            x.device_ptr(),
            wg.device_ptr(),
            wu.device_ptr(),
            wd.device_ptr(),
            y.device_ptr(),
            &mut scratch,
        ).expect("moe execute");
    }

    let mut y_bits = vec![0u16; (T*H) as usize];
    y.copy_to_host(&mut y_bits).unwrap();
    stream.synchronize().unwrap();

    let y_f32 = match dtype {
        MoeDType::F16 => f16_bits_to_f32(&y_bits),
        MoeDType::Bf16 => bf16_bits_to_f32(&y_bits),
    };

    // Output must be finite and not all-zero (something happened).
    let mut any_nonzero = false;
    for (i, v) in y_f32.iter().enumerate() {
        assert!(v.is_finite(), "{:?}: non-finite y[{i}] = {v}", dtype);
        if v.abs() > 1e-6 { any_nonzero = true; }
    }
    assert!(any_nonzero, "{:?}: output was all zeros — kernels likely didn't run", dtype);

    eprintln!("moe_smoke: {:?} OK — {} tokens × {} hidden, y[0..4] = {:?}",
        dtype, T, H, &y_f32[..4]);
}

// Both dtypes run back-to-back in a single test to avoid contending for the
// GPU under cargo's default parallel test executor.
#[test]
fn moe_forward_fp16_and_bf16() {
    run_moe_once(MoeDType::F16);
    run_moe_once(MoeDType::Bf16);
}
