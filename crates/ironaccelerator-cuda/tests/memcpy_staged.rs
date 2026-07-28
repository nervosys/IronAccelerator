//! Correctness of the pinned staging path for host→device copies.
//!
//! `DeviceBuf::copy_from_host` routes multi-chunk transfers through a pinned
//! ring of 4 × 2 MiB chunks.

use ironaccelerator_cuda::drv::{Device, DeviceBuf, Stream};

/// Staging engages at two chunks. These sizes straddle the threshold and the
/// chunk boundary, including sizes that are not a multiple of it, because an
/// off-by-one in the chunk loop would corrupt only the tail.
const CHUNK: usize = 2 << 20;
const THRESHOLD: usize = 2 * CHUNK;

fn sizes() -> Vec<usize> {
    vec![
        1,                 // degenerate
        CHUNK,             // one chunk: below the threshold, direct path
        THRESHOLD - 1,     // one byte short of staging
        THRESHOLD,         // exactly at it
        THRESHOLD + 1,     // ragged third chunk
        CHUNK * 4,         // exactly fills the ring
        CHUNK * 4 + 12345, // wraps the ring with a ragged tail
        CHUNK * 9 + 7,     // several wraps
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

/// `copy_from_host_sync` dispatches between the blocking copy and the staged
/// pipeline by size; both branches must produce identical bytes and leave the
/// stream idle without the caller synchronising.
#[test]
fn sync_copy_selects_a_correct_path_at_every_size() {
    let Ok(dev) = Device::open(0) else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    dev.bind().expect("bind");
    let stream = Stream::new(dev).expect("stream");

    for n in sizes() {
        let src: Vec<u8> = (0..n).map(|i| (i % 241) as u8).collect();
        let mut buf: DeviceBuf<u8> = DeviceBuf::alloc(stream.clone(), n).expect("alloc");
        buf.copy_from_host_sync(&src).expect("copy_from_host_sync");

        // Deliberately no synchronize: the blocking readback must be enough.
        let mut out = vec![0u8; n];
        buf.copy_to_host_blocking(&mut out).expect("blocking read");
        if out != src {
            let first = out
                .iter()
                .zip(src.iter())
                .position(|(a, b)| a != b)
                .expect("vectors differ but no differing index");
            panic!("size {n} (threshold {THRESHOLD}): first mismatch at byte {first}");
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

/// `Stream::new_legacy_ordered` gives cudarc-identical stream semantics: the
/// stream is sequenced against the legacy stream, so the synchronous copy paths
/// skip their drain. Correctness must be identical to the non-blocking default.
///
/// Kept as an opt-in rather than the default because measurement showed no
/// throughput benefit, and blocking streams do not overlap with each other.
#[test]
fn legacy_ordered_stream_round_trips_identically() {
    let Ok(dev) = Device::open(0) else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    dev.bind().expect("bind");
    let stream = Stream::new_legacy_ordered(dev.clone()).expect("legacy-ordered stream");
    assert!(stream.is_legacy_ordered());
    assert!(!Stream::new(dev).expect("stream").is_legacy_ordered());

    for n in sizes() {
        let src: Vec<u8> = (0..n).map(|i| (i % 233) as u8).collect();
        let mut buf: DeviceBuf<u8> = DeviceBuf::alloc(stream.clone(), n).expect("alloc");
        buf.copy_from_host_sync(&src).expect("sync write");
        let mut out = vec![0u8; n];
        buf.copy_to_host_sync(&mut out).expect("sync read");
        assert!(out == src, "size {n} corrupted on a legacy-ordered stream");
    }
}
