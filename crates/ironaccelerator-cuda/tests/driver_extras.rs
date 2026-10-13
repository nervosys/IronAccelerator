//! Live-GPU smoke tests for the "completeness" driver primitives added to
//! `drv`: device UUID, function attributes, occupancy query, module global
//! lookup, cooperative-groups launch, and peer memcpy.

use ironaccelerator_cuda::drv::{Device, DeviceBuf, LaunchCfg, Module, Stream};
use ironaccelerator_cuda::kernel::{compile, CompileOptions};
use ironaccelerator_cuda::sys;

fn have_cuda() -> bool {
    matches!(Device::count(), Ok(n) if n > 0)
}

/// Serialises the tests in this file. `graph_capture_honours_every_mode`
/// captures in `CU_STREAM_CAPTURE_MODE_GLOBAL`, which the driver invalidates
/// if *any* thread makes an unsafe call (e.g. a synchronous allocation) during
/// the capture — so it cannot share the process with concurrent GPU tests.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn device_uuid_is_stable_and_nonzero() {
    let _gpu = gpu_lock();
    if !have_cuda() {
        return;
    }
    let dev = Device::open(0).unwrap();
    let u1 = dev.uuid().unwrap();
    let u2 = dev.uuid().unwrap();
    assert_eq!(u1, u2, "UUID must be stable across calls");
    assert!(u1.bytes.iter().any(|&b| b != 0), "UUID looks empty");
}

const TRIVIAL_SRC: &str = r#"
extern "C" __global__
void noop(float* x, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}
"#;

#[test]
fn function_attribute_round_trip() {
    let _gpu = gpu_lock();
    if !have_cuda() {
        return;
    }
    let dev = Device::open(0).unwrap();
    let (maj, min) = dev.compute_capability().unwrap();
    let arch = format!("compute_{maj}{min}");
    let ptx = compile(TRIVIAL_SRC, &arch, &CompileOptions::default()).unwrap();
    let module = Module::load(dev.clone(), &ptx).unwrap();
    let f = module.function("noop").unwrap();

    // Every kernel has a max-threads-per-block; cuFuncGetAttribute reports it.
    let max_tpb = f
        .attribute(sys::driver::CUfunction_attribute::MaxThreadsPerBlock)
        .unwrap();
    assert!(max_tpb >= 64, "MaxThreadsPerBlock suspicious: {max_tpb}");

    // Setting MaxDynamicSharedSizeBytes to 0 is a valid no-op every SM accepts.
    f.set_attribute(
        sys::driver::CUfunction_attribute::MaxDynamicSharedSizeBytes,
        0,
    )
    .unwrap();
}

#[test]
fn occupancy_query_returns_plausible_block_count() {
    let _gpu = gpu_lock();
    if !have_cuda() {
        return;
    }
    let dev = Device::open(0).unwrap();
    let (maj, min) = dev.compute_capability().unwrap();
    let arch = format!("compute_{maj}{min}");
    let ptx = compile(TRIVIAL_SRC, &arch, &CompileOptions::default()).unwrap();
    let module = Module::load(dev.clone(), &ptx).unwrap();
    let f = module.function("noop").unwrap();

    let blocks_per_sm = f.occupancy_max_active_blocks_per_sm(256, 0).unwrap();
    assert!(
        blocks_per_sm >= 1,
        "trivial kernel should fit at least 1 block/SM, got {blocks_per_sm}"
    );
    assert!(
        blocks_per_sm <= 64,
        "occupancy looks too high to be real: {blocks_per_sm}"
    );
}

const GLOBAL_SRC: &str = r#"
__device__ unsigned int my_global = 0xdeadbeefu;
extern "C" __global__ void noop(){}
"#;

#[test]
fn module_global_lookup_returns_pointer_and_size() {
    let _gpu = gpu_lock();
    if !have_cuda() {
        return;
    }
    let dev = Device::open(0).unwrap();
    let (maj, min) = dev.compute_capability().unwrap();
    let arch = format!("compute_{maj}{min}");
    let ptx = compile(GLOBAL_SRC, &arch, &CompileOptions::default()).unwrap();
    let module = Module::load(dev.clone(), &ptx).unwrap();
    let (ptr, bytes) = module.global("my_global").unwrap();
    assert!(ptr != 0);
    assert_eq!(bytes, std::mem::size_of::<u32>());
}

#[test]
fn cooperative_launch_runs() {
    let _gpu = gpu_lock();
    if !have_cuda() {
        return;
    }
    let dev = Device::open(0).unwrap();
    let stream = Stream::new(dev.clone()).unwrap();
    let (maj, min) = dev.compute_capability().unwrap();
    let arch = format!("compute_{maj}{min}");
    let ptx = compile(TRIVIAL_SRC, &arch, &CompileOptions::default()).unwrap();
    let module = Module::load(dev.clone(), &ptx).unwrap();
    let f = module.function("noop").unwrap();

    let mut buf: DeviceBuf<f32> = DeviceBuf::alloc(stream.clone(), 1024).unwrap();
    buf.copy_from_host(&vec![0f32; 1024]).unwrap();
    let n = 1024u32;
    // Single block: trivially within the cooperative-launch concurrency limit
    // on any device that supports cooperative groups.
    f.launch_cooperative(
        LaunchCfg::for_elements(n, 256),
        &stream,
        (buf.device_ptr(), n),
    )
    .unwrap();
    stream.synchronize().unwrap();

    let mut out = vec![0f32; 1024];
    buf.copy_to_host(&mut out).unwrap();
    stream.synchronize().unwrap();
    for (i, v) in out.iter().enumerate() {
        assert_eq!(*v, 1.0, "cooperative launch missed index {i}");
    }
}

#[test]
fn copy_from_peer_async_round_trips_when_two_devices_exist() {
    let _gpu = gpu_lock();
    if !have_cuda() {
        return;
    }
    if Device::count().unwrap() < 2 {
        eprintln!("driver_extras: only one CUDA device — skipping peer test");
        return;
    }
    let dev_a = Device::open(0).unwrap();
    let dev_b = Device::open(1).unwrap();
    if !dev_a.can_access_peer(&dev_b).unwrap() {
        eprintln!("driver_extras: devices can't peer — skipping");
        return;
    }
    dev_a.enable_peer_access(&dev_b).unwrap();
    dev_b.enable_peer_access(&dev_a).unwrap();

    let stream_a = Stream::new(dev_a.clone()).unwrap();
    let stream_b = Stream::new(dev_b.clone()).unwrap();

    let mut src: DeviceBuf<u32> = DeviceBuf::alloc(stream_a.clone(), 256).unwrap();
    let mut dst: DeviceBuf<u32> = DeviceBuf::alloc(stream_b.clone(), 256).unwrap();
    let host_in: Vec<u32> = (0..256).collect();
    src.copy_from_host(&host_in).unwrap();
    stream_a.synchronize().unwrap();

    dst.copy_from_peer_async(&src).unwrap();
    stream_b.synchronize().unwrap();

    let mut host_out = vec![0u32; 256];
    dst.copy_to_host(&mut host_out).unwrap();
    stream_b.synchronize().unwrap();
    assert_eq!(host_out, host_in);
}

#[test]
fn graph_capture_honours_every_mode() {
    let _gpu = gpu_lock();
    use ironaccelerator_cuda::graph::{capture, CaptureMode};
    if !have_cuda() {
        return;
    }
    let dev = Device::open(0).unwrap();
    let stream = Stream::new(dev.clone()).unwrap();
    let buf = DeviceBuf::<u8>::alloc(stream.clone(), 256).unwrap();
    stream.synchronize().unwrap();
    for (mode, value) in [
        (CaptureMode::Global, 0x11u8),
        (CaptureMode::ThreadLocal, 0x22),
        (CaptureMode::Relaxed, 0x33),
    ] {
        let exec = capture(&stream, dev.clone(), mode, |s| {
            Ok(s.memset_d8_async(buf.device_ptr(), value, 256)?)
        })
        .unwrap_or_else(|e| panic!("capture in {mode:?} failed: {e}"));
        exec.launch(&stream).unwrap();
        stream.synchronize().unwrap();
        let mut host = [0u8; 256];
        buf.copy_to_host_sync(&mut host).unwrap();
        assert!(host.iter().all(|&b| b == value), "{mode:?} replay wrong");
    }
}

#[test]
fn explicit_mem_pool_accounting_cap_and_trim() {
    let _gpu = gpu_lock();
    use ironaccelerator_cuda::drv::AsyncMemPool;
    if !have_cuda() {
        return;
    }
    const MIB: usize = 1 << 20;
    let dev = Device::open(0).unwrap();
    let stream = Stream::new(dev.clone()).unwrap();
    let pool = AsyncMemPool::new(dev.clone(), None).unwrap();

    // Round-trip data through a pool-backed buffer.
    let src: Vec<u32> = (0..MIB as u32 / 4).collect();
    let mut buf = DeviceBuf::<u32>::alloc_from_pool(stream.clone(), &pool, src.len()).unwrap();
    buf.copy_from_host(&src).unwrap();
    let mut back = vec![0u32; src.len()];
    buf.copy_to_host_sync(&mut back).unwrap();
    assert_eq!(src, back);
    assert!(
        pool.used_bytes().unwrap() >= MIB as u64,
        "pool must account the buffer"
    );

    // Freed memory returns to *this* pool and, with the default MAX
    // threshold, stays reserved until trimmed.
    drop(buf);
    stream.synchronize().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert!(pool.reserved_bytes().unwrap() > 0);
    assert!(pool.used_high_bytes().unwrap() >= MIB as u64);
    pool.trim_to(0).unwrap();
    assert_eq!(
        pool.reserved_bytes().unwrap(),
        0,
        "trim_to(0) releases everything"
    );

    // max_size caps the reservation.
    let capped = AsyncMemPool::new(dev.clone(), Some(32 * MIB)).unwrap();
    let err = DeviceBuf::<u8>::alloc_from_pool(stream.clone(), &capped, 64 * MIB);
    assert!(err.is_err(), "allocation beyond max_size must fail");
    stream.synchronize().unwrap();

    // A pool on another device is rejected before reaching the driver.
    if Device::count().unwrap() > 1 {
        let other = AsyncMemPool::new(Device::open(1).unwrap(), None).unwrap();
        let e = DeviceBuf::<u8>::alloc_from_pool(stream.clone(), &other, 16).err();
        assert!(matches!(
            e,
            Some(ironaccelerator_cuda::drv::Error::Precondition { .. })
        ));
    }
}

#[test]
fn device_open_sets_default_pool_retain_on_free() {
    let _gpu = gpu_lock();
    if !have_cuda() {
        return;
    }
    let dev = Device::open(0).unwrap();
    dev.bind().unwrap();
    let d = sys::driver::fns().unwrap();
    let mut pool = sys::driver::CUmemPool::default();
    let mut cu_dev = 0;
    let mut threshold: u64 = 0;
    unsafe {
        assert_eq!(
            (d.cuDeviceGet)(&mut cu_dev, 0),
            sys::driver::CUresult::Success
        );
        assert_eq!(
            (d.cuDeviceGetDefaultMemPool)(&mut pool, cu_dev),
            sys::driver::CUresult::Success
        );
        assert_eq!(
            (d.cuMemPoolGetAttribute)(
                pool,
                sys::driver::CUmemPool_attribute::ReleaseThreshold,
                &mut threshold as *mut u64 as *mut std::ffi::c_void,
            ),
            sys::driver::CUresult::Success
        );
    }
    assert_eq!(threshold, u64::MAX, "default pool must retain freed memory");
}

#[test]
fn cublaslt_fast_accum_and_raw_attr_are_accepted() {
    use ironaccelerator_cuda::blas::{ComputeType, DType, MatmulDesc};
    let _gpu = gpu_lock();
    let Ok(mut desc) = MatmulDesc::new(ComputeType::F32, DType::R32F) else {
        eprintln!("SKIPPED cublaslt_fast_accum_and_raw_attr_are_accepted: cuBLASLt not available");
        return;
    };
    desc.set_fast_accum(true).unwrap();
    desc.set_fast_accum(false).unwrap();
    // CUBLASLT_MATMUL_DESC_SCALE_TYPE (1) is a 4-byte attribute.
    unsafe { desc.set_attr_raw_u32(1, DType::R32F as u32) }.unwrap();
}
