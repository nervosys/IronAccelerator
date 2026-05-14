//! Live-GPU smoke tests for the "completeness" driver primitives added to
//! `drv`: device UUID, function attributes, occupancy query, module global
//! lookup, cooperative-groups launch, and peer memcpy.

use ironaccelerator_cuda::drv::{Device, DeviceBuf, LaunchCfg, Module, Stream};
use ironaccelerator_cuda::kernel::{compile, CompileOptions};
use ironaccelerator_cuda::sys;

fn have_cuda() -> bool {
    matches!(Device::count(), Ok(n) if n > 0)
}

#[test]
fn device_uuid_is_stable_and_nonzero() {
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
