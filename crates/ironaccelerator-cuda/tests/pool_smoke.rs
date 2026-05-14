//! Live-GPU smoke test for the per-stream `MemPool`.

use ironaccelerator_cuda::drv::{Device, Stream};
use ironaccelerator_cuda::pool::MemPool;

fn have_cuda() -> bool {
    matches!(Device::count(), Ok(n) if n > 0)
}

#[test]
fn pool_recycles_buffers() {
    if !have_cuda() {
        eprintln!("pool_smoke: no CUDA — skipping");
        return;
    }
    let dev = Device::open(0).unwrap();
    let stream = Stream::new(dev).unwrap();
    let mut pool = MemPool::new(stream.clone());

    // First alloc: must hit the driver (bucket empty). Record the ptr.
    let buf = pool.alloc::<u32>(1024).unwrap();
    let first_ptr = buf.device_ptr();
    assert_eq!(buf.len(), 1024);
    drop(buf);

    // Second alloc of the same size: must come from the bucket. Pointer
    // identity is the smoking gun that we skipped cuMemAllocAsync.
    let buf2 = pool.alloc::<u32>(1024).unwrap();
    assert_eq!(
        buf2.device_ptr(),
        first_ptr,
        "pool should recycle the pointer"
    );
    assert_eq!(buf2.len(), 1024);
    drop(buf2);

    // Different size class — different pointer.
    let buf3 = pool.alloc::<u32>(64 * 1024).unwrap();
    assert_ne!(buf3.device_ptr(), first_ptr);
    drop(buf3);

    // Round-trip data through a pooled buffer to prove it's actually usable.
    let mut buf4 = pool.alloc::<u32>(8).unwrap();
    buf4.copy_from_host(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let mut out = vec![0u32; 8];
    buf4.copy_to_host(&mut out).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(out, [1, 2, 3, 4, 5, 6, 7, 8]);
    drop(buf4);

    pool.shrink(); // releases cached blocks back to driver
    stream.synchronize().unwrap();
}

#[test]
fn pool_bypasses_huge_allocations() {
    if !have_cuda() {
        return;
    }
    let dev = Device::open(0).unwrap();
    let stream = Stream::new(dev).unwrap();
    let pool = MemPool::new(stream.clone());

    // > 256 MiB bypasses the bucket and goes straight to the driver.
    // 512 MiB allocation must succeed and be freeable.
    let huge = pool.alloc::<u8>(512 << 20).unwrap();
    assert_eq!(huge.len(), 512 << 20);
    drop(huge);
    stream.synchronize().unwrap();
}
