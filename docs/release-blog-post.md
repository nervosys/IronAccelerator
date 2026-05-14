# IronAccelerator: a faster, agent-first drop-in for cudarc

## TL;DR

We just shipped **`ironaccelerator-cuda`** — a Rust safe-CUDA wrapper layer that's a drop-in replacement for `cudarc::driver` and `cudarc::nvrtc`. Same API shape, same idioms, ported in a one-line `use` swap. Faster on every host-side hot path we measured.

On an RTX 3090 Ti against cudarc 0.19:

| Operation | cudarc | IronAccelerator | Speedup |
|---|---|---|---|
| **`MemPool` alloc + free (any size, warm)** | ~740 ns | **~10 ns** | **~75×** |
| async alloc + free (plain `DeviceBuf`) | ~910 ns | **~445 ns** | **2.04×** |
| stream synchronize (empty) | ~109 ns | **~85 ns** | **1.29×** |
| stream create + destroy | ~999 ns | **~888 ns** | **11%** |
| kernel launch | ~5.5 µs | ~5.5 µs | parity (cuLaunchKernel floor) |
| bulk memcpy (1 MB → 16 MB) | parity | parity | PCIe DMA floor |

If you're running a serving loop, an autotuner, or an agent dispatch system that churns thousands of small buffers per second, the alloc/free numbers are the ones that matter.

## What it is

`ironaccelerator-cuda` is a low-level, hardware-agnostic Rust interface over the CUDA driver. It targets CUDA Toolkit 13.2, loads every vendor library (cuBLAS, cuBLASLt, cuDNN, cuRAND, cuSPARSE, cuSOLVER, cuFFT, cuTENSOR, NCCL, NVTX, CUPTI) via `libloading` at first use, and exposes safe Rust wrappers over the driver primitives.

It is **not** a kernel library. There's no FlashAttention, no FP8 recipes, no workload planners, no autotuners. It's the substrate underneath all of that — what cudarc has always been, but with the wrapper overhead cut by 2–75× depending on the operation.

Specifically:

- **`drv`** — Device, Stream, Event, DeviceBuf, PinnedBuf, Module, Function. The full safe surface over `libcuda.so` / `nvcuda.dll`.
- **`kernel`** — NVRTC compile with both in-memory and on-disk PTX caches, keyed by source hash + arch.
- **`cudarc_compat`** — a one-line drop-in for cudarc 0.19 users.
- **`blas`, `cudnn`, `fft`, `nccl`, `cusparse`, `cusolver`, `cutensor`, `rng`** — per-library handle plumbing.
- **`advanced`** — VMM, green contexts, multicast teams, conditional graph nodes.
- **`pool`** — opt-in recycling allocator (the 75× win).
- **`sys`** — re-export of the raw `iron_cuda_sys` FFI for callers that need to drop a level.

## The 30-second port

```rust
// before
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, LaunchAsync};
use cudarc::nvrtc::compile_ptx;

// after — one import line
use ironaccelerator_cuda::cudarc_compat::{
    CudaDevice, CudaSlice, CudaStream, LaunchAsync, compile_ptx,
};

// Everything below is identical to what you'd write against cudarc 0.19:
let dev    = CudaDevice::new(0)?;
let stream = dev.default_stream();
let xs     = stream.htod_copy(vec![1.0f32, 2.0, 3.0])?;
let out    = stream.dtoh_sync_copy(&xs)?;
```

We cover the cudarc 0.19 driver/nvrtc surface that real users actually reach for: `CudaDevice::{new, count, ordinal, name, total_mem, mem_get_info, compute_capability, default_stream, new_stream, new_stream_with_priority, synchronize}`, `CudaSlice::{len, num_bytes, is_empty, byte_len, device_ptr, stream, ordinal, try_clone, view, view_mut}`, `CudaStreamExt::{htod_copy, htod_sync_copy, alloc, alloc_zeros, dtoh_sync_copy, dtoh_sync_copy_into, record_event, wait, join}`, `CudaEvent`, `CudaTimingEvent`, `CudaModule`, `CudaFunction::launch`, `LaunchAsync::launch_async`, `DevicePtr::device_ptr`, `compile_ptx{,_with_opts}`.

Module docs in `cudarc_compat` ship a side-by-side coverage map so an agent can answer "where did cudarc's `foo` go" without reading source.

## Why it's faster

We hold a `Rust + cudarc` style wrapper to four rules:

**1. Cache the driver function table inline.** Every `Device`, `Stream`, `Event`, `Module`, `Function`, and `PinnedBuf` stores `&'static DriverFns` resolved once at construction. A wrapped op (`stream.synchronize()`, `buf.copy_from_host(...)`, `module.function(...)`) reaches the function table via a struct-field load — no atomic, no LazyLock walk, no thread-context rebind. cudarc reaches for `cuCtxGetCurrent` on every call to verify the thread's bound context; we trust the bind that happened at `Device::open`. That's worth ~30 ns per sync.

**2. Cold-path every error.** `check()` is `#[inline(always)]` and the failure branch is moved to a `#[cold] #[inline(never)] check_err()` companion. The success path compiles to load/test/branch with no `Error`-enum materialisation. Same treatment for overflow helpers.

**3. Kill eager allocations on the hot path.** The old `ok_or(Error::Precondition { msg: "size overflow".into() })?` pattern was heap-allocating a `String` *on every successful alloc* — `.into()` is `String::from`, which is eager. Replaced with `let-else` + a cold helper. That alone took 64 KB alloc from 491 ns → 340 ns.

**4. Inline the public surface.** Every `CudaStreamExt` method, every `DeviceBuf` accessor, every `Drop`, every `Stream::synchronize` and `wait_for` is `#[inline]`. With the cached function pointer, the entire alloc path inlines to ~20 instructions before the FFI call.

The headline ~75× number comes from the **opt-in `MemPool`**:

## `MemPool`: the 75× win

For workloads that allocate and free hundreds or thousands of buffers per second (KV-cache slots, per-token scratch, agent tool churn), the `cuMemAllocAsync` FFI round-trip itself becomes the bottleneck. `MemPool` is a per-stream Rust-side freelist that caches recycled `DeviceBuf` allocations in power-of-two byte buckets (1 KiB through 256 MiB). The warm path bypasses the driver entirely:

```rust
use ironaccelerator_cuda::pool::MemPool;

let pool = MemPool::new(stream.clone());
for _ in 0..N {
    let buf = pool.alloc::<f32>(1024)?;   // pop bucket, no FFI
    // ... use buf in kernels exactly like any DeviceBuf ...
    drop(buf);                             // push bucket, no FFI
}
```

`PooledBuf<'p, T>: Deref<Target = DeviceBuf<T>>` — every existing method (including all `cudarc_compat`) works unchanged through deref. `into_inner()` escapes back to a plain `DeviceBuf` when you want a buffer to outlive the pool.

Under the hood it's a three-tier cache:

1. **Per-thread, per-bucket front cache** (4-deep fixed array, no lock, `UnsafeCell` access). This is the warm path. ~10 ns end-to-end.
2. **Shared `parking_lot::Mutex<Vec>` back cache**, bounded by `max_per_bucket`. Spills here when the front fills or a different thread allocs. ~20 ns.
3. **Driver** (`cuMemAllocAsync`). Hit when both caches miss. ~500 ns.

The bench's single-thread alloc-then-free loop stays entirely in tier 1, so we measure the floor of safe-Rust dispatch: `RefCell`-free `UnsafeCell` access + fixed-array index + branch + balanced `Arc<Stream>` ref-traffic for stream lifetime. **Cost ≈ 10 ns regardless of allocation size.**

The pool is bounded (32 blocks per bucket default; tunable), big requests (>256 MiB) bypass entirely, and `MemPool::shrink()` drains everything back to the driver between epochs.

Default `DeviceBuf::alloc` still goes straight to `cuMemAllocAsync` — the cudarc drop-in story is unaffected by the pool's existence. The pool is opt-in for workloads that need it.

## Designed for agents

Each of these is a deliberate trade-off, not aesthetic:

- **Errors name the operation.** `Error::Driver { op: "cuMemAllocAsync", code }` — no anonymous `CUDA_ERROR_*` payloads. An agent reading a panic message can grep the source for the op string and land directly on the call site. No stack-trace archaeology.
- **Crate-level docs ship a task→method table.** `ironaccelerator_cuda`'s `lib.rs` doc opens with "If you want to do X, use Y" mapping the 10 most common dispatch points (port cudarc / allocate / compile NVRTC / launch / time / capture graph / call vendor lib / drop to FFI). An LLM agent loading the rendered docs sees the canonical entry points in one screenful.
- **One way to do each thing.** No five flavors of allocation, no two stream types, no three launch APIs. The surface is small enough for an agent to hold the whole thing in context.
- **A runnable example you can copy whole.** `examples/saxpy_cudarc_style.rs` is ~80 lines, runs end-to-end on a live GPU (verified 1.19 × 10⁻⁷ relative error vs CPU reference), and uses only `cudarc_compat::*` — exactly what an LLM porting a cudarc app would produce.
- **`cudarc_compat` module doc is a coverage map**, not prose. Each cudarc 0.19 API → IronAccelerator equivalent in a single table, plus an explicit "differences worth knowing" list.

## What we deliberately don't ship

This is the bottom of the stack. We don't ship kernels, planners, FP8 recipes, attention/MoE implementations, or workload autotuners. The surface is small enough (drv + kernel + per-library handles + cudarc_compat + opt-in pool) that downstream libraries — inference servers, training runtimes, agent frameworks — can layer on top without fighting our abstractions or carrying dead weight.

This is by design. cudarc is the proven shape; we're keeping it.

## How the gap holds up at each layer

The optimization isn't one trick — it's a stack:

- **`AtomicPtr<DriverFns>` fast-path cache** in `iron_cuda_sys::driver::fns()` collapses two `OnceLock` acquires into one acquire load + null check.
- **Per-handle cached `&'static DriverFns`** — `Device`, `Stream`, `Event`, `Module`, `Function`, `PinnedBuf`, `CapturedGraph`, `GraphExec` each store the function-table reference at construction. Zero atomic ops per call in the steady state.
- **Cached stream priority range** on `Device` (`OnceCell<(i32, i32)>`) amortizes `cuStreamGetPriorityRange` across every `Stream::new`.
- **Cold-path error construction** so the success branch is `test/jne/ret`.
- **Killed the eager `String::from`** in `DeviceBuf::alloc`'s overflow path.
- **`#[inline]` everywhere it matters**, with `#[cold] #[inline(never)]` siblings for the unhappy paths.
- **Borrowed pool lifetime** (`PooledBuf<'p, T>`) so `MemPool` doesn't need `Arc<PoolInner>` traffic.
- **Thread-local front cache via `UnsafeCell`** wrapped in a manually-`Sync` newtype — safe because `thread_local::ThreadLocal` hands out one cell per thread.

Each step shaved nanoseconds. Compounded, they're a 75× gap on the hottest path in any serving loop.

## Reference numbers

RTX 3090 Ti, CUDA 13.2 driver 596.36, release build, x86_64 Windows:

```
vs_cudarc/alloc/pooled_alloc_free/ironaccelerator_pool/1KB    9.77 ns
vs_cudarc/alloc/pooled_alloc_free/cudarc_no_pool/1KB         726 ns

vs_cudarc/alloc/pooled_alloc_free/ironaccelerator_pool/64KB   9.76 ns
vs_cudarc/alloc/pooled_alloc_free/cudarc_no_pool/64KB        756 ns

vs_cudarc/alloc/pooled_alloc_free/ironaccelerator_pool/1MB    9.77 ns
vs_cudarc/alloc/pooled_alloc_free/cudarc_no_pool/1MB         732 ns

vs_cudarc/alloc/pooled_alloc_free/ironaccelerator_pool/16MB   9.86 ns
vs_cudarc/alloc/pooled_alloc_free/cudarc_no_pool/16MB        739 ns
```

Reproduce yourself:

```bash
cargo bench -p ironaccelerator-cuda --bench vs_cudarc
```

Or just run the end-to-end SAXPY example:

```bash
cargo run --release -p ironaccelerator-cuda --example saxpy_cudarc_style
```

## What it costs you

Two real trade-offs versus cudarc, written down honestly:

1. **Per-call thread-context bind.** cudarc verifies `cuCtxGetCurrent` on every driver call (~50–100 ns). We bind once at `CudaDevice::new` and trust the binding to persist. If you manually switch contexts elsewhere, call `dev.raw().bind()` before resuming. For the typical "one device, one process" case (every serving stack we've measured), this is free correctness; in the rare multi-context case it's something to know.
2. **`PooledBuf<'p, T>` carries a lifetime.** That's what lets the hot path skip `Arc` traffic. For long-lived buffers, `PooledBuf::into_inner()` detaches as a plain `DeviceBuf` and you keep cudarc-shape ownership semantics. For dispatch-loop scratch buffers, the lifetime is unannotated and behaves like any other scoped value.

The `Drop` semantics are simpler than cudarc's — we just call `cuMemFreeAsync`, no per-buffer event-fence tracking. If you need fence-tracking, build it on top.

## What's on the box

- 45 live-GPU tests (lib + cudarc_compat + gpu_smoke + pool_smoke).
- 38 workspace test suites, including 9 cudarc_compat round-trip tests against a real driver.
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` clean across all 16 crates.
- `cargo clippy --workspace --all-targets --release` clean.
- `cargo fmt --all -- --check` clean.
- Zero `TODO` / `FIXME` / `unimplemented!` in CUDA src.
- Per-crate `repository`, `homepage`, `documentation`, `keywords`, `categories` metadata — discoverable on crates.io.
- `RELEASE.md` runbook with the exact `cargo publish` ordering for all 16 workspace crates.
- AGPL-3.0-or-later, with a commercial license available.

## When to use it, when not to

**Use it when:** you're writing Rust against the CUDA driver, you currently use cudarc or are weighing it, you have a dispatch loop that allocates/frees small buffers per call, you want low-overhead access to cuBLASLt/cuDNN/NCCL/etc. without a vendor SDK linked at build time, you're building infrastructure that an LLM agent might modify in the loop.

**Don't use it when:** you need cudarc's exact thread-context-bind-on-every-call paranoia (rare), you need cudarc's per-buffer event-fence Drop (also rare), you want a kernel library (you want a layer on top of this, not this).

## Try it

```toml
[dependencies]
ironaccelerator-cuda = "1.1"
```

Source at https://github.com/nervosys/IronAccelerator. The cudarc migration map is in [`crates/ironaccelerator-cuda/src/cudarc_compat.rs`](https://github.com/nervosys/IronAccelerator/blob/master/crates/ironaccelerator-cuda/src/cudarc_compat.rs). Performance numbers are reproducible with the `vs_cudarc` Criterion benchmark in the same crate.

Built so it can be picked up and shipped — by humans or agents.
