//! Correctness of the pinned staging path for host→device copies.
//!
//! `DeviceBuf::copy_from_host` routes anything at or above 256 KiB through a
//! chunked pinned ring. These sizes deliberately straddle the threshold and the
//! 4 MiB chunk boundary, including sizes that are not a multiple of it, because
//! an off-by-one in the chunk loop would corrupt only the tail.

use ironaccelerator_cuda::drv::{Device, DeviceBuf, Stream};

fn sizes() -> Vec<usize> {
    let chunk = 4 << 20;
    vec![
        255 << 10, // just below the staging threshold
        256 << 10, // exactly at it
        (256 << 10) + 1,
        chunk - 1,         // one short of a chunk
        chunk,             // exactly one chunk
        chunk + 1,         // spills into a second
        chunk * 2,         // exactly fills the ring once
        chunk * 2 + 12345, // wraps the ring with a ragged tail
        chunk * 5 + 7,     // several wraps
    ]
}

#[test]
fn staged_copies_round_trip_exactly() {
    let Ok(dev) = Device::open(0) else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    dev.bind().expect("bind");
    let stream = Stream::new(dev).expect("stream");

    for n in sizes() {
        // A byte pattern where every position differs from its neighbours, so a
        // misplaced chunk cannot coincidentally compare equal.
        let src: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let mut buf: DeviceBuf<u8> = DeviceBuf::alloc(stream.clone(), n).expect("alloc");
        buf.copy_from_host(&src).expect("copy_from_host");

        let mut out = vec![0u8; n];
        buf.copy_to_host(&mut out).expect("copy_to_host");
        stream.synchronize().expect("sync");

        assert_eq!(out.len(), src.len());
        if out != src {
            let first = out
                .iter()
                .zip(src.iter())
                .position(|(a, b)| a != b)
                .expect("vectors differ but no differing index");
            panic!("size {n}: first mismatch at byte {first}");
        }
    }
}

#[test]
fn repeated_staged_copies_reuse_the_ring() {
    let Ok(dev) = Device::open(0) else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    dev.bind().expect("bind");
    let stream = Stream::new(dev).expect("stream");

    // Enough iterations to wrap the two-slot ring many times, which is where a
    // missing event wait would show up as torn data.
    let n = (4 << 20) + 4096;
    let mut buf: DeviceBuf<u8> = DeviceBuf::alloc(stream.clone(), n).expect("alloc");
    for round in 0u8..8 {
        let src: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_add(round)).collect();
        buf.copy_from_host(&src).expect("copy_from_host");
        let mut out = vec![0u8; n];
        buf.copy_to_host(&mut out).expect("copy_to_host");
        stream.synchronize().expect("sync");
        assert!(out == src, "round {round} corrupted");
    }
}

#[test]
fn blocking_readback_matches_staged_write() {
    let Ok(dev) = Device::open(0) else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    dev.bind().expect("bind");
    let stream = Stream::new(dev).expect("stream");

    // Spans several staging chunks with a ragged tail.
    let n = (2 << 20) * 3 + 517;
    let src: Vec<u8> = (0..n).map(|i| (i % 199) as u8).collect();
    let mut buf: DeviceBuf<u8> = DeviceBuf::alloc(stream.clone(), n).expect("alloc");
    buf.copy_from_host(&src).expect("staged write");

    // No explicit synchronize: copy_to_host_blocking must drain the stream
    // itself, including staged chunks still in flight.
    let mut out = vec![0u8; n];
    buf.copy_to_host_blocking(&mut out).expect("blocking read");
    assert!(out == src, "blocking readback did not observe staged write");
}
