# IronAccelerator

A high-performance, **low-level hardware-agnostic** Rust interface over NVIDIA, AMD, Apple, Qualcomm, Intel, Google, and AWS accelerators plus open cross-vendor APIs (Vulkan / OpenGL / WebGPU). **Agent-first**: predictable shapes, terse APIs that an LLM can reason about without docs, errors that name the operation that failed.

> **Scope.** IronAccelerator is a *driver substrate*, not a kernel library.
> Each backend crate wraps the vendor driver/runtime (devices, streams,
> events, memory, kernel compile + cache, handle plumbing for vendor
> libraries like cuBLAS / cuDNN / NCCL / cuFFT). It does **not** ship
> kernels, planners, FP8 recipes, attention/MoE implementations, or
> workload autotuners — those belong to libraries layered on top.

## 30-second drop-in for `cudarc` users

```rust
// before
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync};
use cudarc::nvrtc::compile_ptx;

// after — one import line
use ironaccelerator_cuda::cudarc_compat::{CudaDevice, CudaSlice, LaunchAsync, compile_ptx};

let dev    = CudaDevice::new(0)?;
let stream = dev.default_stream();
let xs     = stream.htod_copy(vec![1.0f32, 2.0, 3.0])?;
let out    = stream.dtoh_sync_copy(&xs)?;
```

Full coverage map and migration notes live in the module docs at
[`ironaccelerator_cuda::cudarc_compat`](crates/ironaccelerator-cuda/src/cudarc_compat.rs) — start there if you're porting an existing cudarc codebase. A runnable end-to-end example using NVRTC compile + kernel launch + H↔D copy is at [`examples/saxpy_cudarc_style.rs`](crates/ironaccelerator-cuda/examples/saxpy_cudarc_style.rs):

```bash
cargo run --release -p ironaccelerator-cuda --example saxpy_cudarc_style
```

**Why switch:** on the host-side hot path we're **1.4–2× faster than cudarc 0.19**
([benchmarks](#vs-cudarc-drop-in-replacement)). cudarc rebinds the
thread-context on every driver call; we cache it once. cudarc tracks per-buffer
event fences on `Drop`; we just call `cuMemFreeAsync`. The wins compound at
high-frequency dispatch loops.

## Backend support matrix

| Backend    | Vendor / API                       | Enumerate | Driver wrappers | Vendor-lib handles | Min SDK / runtime               |
| ---------- | ---------------------------------- | --------- | --------------- | ------------------ | ------------------------------- |
| CUDA       | NVIDIA                             | ✅         | ✅               | cuBLAS / cuBLASLt / cuDNN / cuFFT / cuSPARSE / cuSOLVER / cuTENSOR / NCCL / cuRAND / NVTX / CUPTI | CUDA 12.5+ driver (13.x tested) |
| ROCm       | AMD                                | ✅         | ✅               | hipBLASLt          | ROCm 6.2+                       |
| Metal      | Apple                              | ✅         | ✅               | MPS                | macOS 14+ / iOS 17+             |
| QNN        | Qualcomm Hexagon NPU               | ✅         | ⚠️ HDK needed    | —                  | QNN SDK 2.22+                   |
| Vulkan     | cross-vendor GPU compute           | ✅         | ✅               | —                  | Vulkan 1.3 ICD                  |
| OpenGL     | legacy / embedded GPU fallback     | ✅ ctx     | ✅               | —                  | GL 4.3+ compute                 |
| WebGPU     | native (Vk/Metal/DX12) + browser   | ✅         | ✅               | —                  | wgpu 22 / Chrome 113+           |
| TPU (PJRT) | Google TPU v4 / v5 / v6e           | ✅ env     | ⏳ PJRT client   | —                  | PJRT plugin (`libtpu.so`)       |
| Level Zero | Intel GPU (Arc / Flex / PVC) + NPU | ✅         | ✅               | —                  | `ze_loader` from Intel compute  |
| AWS Neuron | Trainium / Inferentia              | ✅ cores   | ⏳ NEFF load     | —                  | `libnrt` (Neuron SDK 2.x)       |

Every backend is loaded via `libloading` at first use — the workspace builds on any host, present or missing vendor SDK. Missing runtimes surface as typed `NotAvailable` errors, not link failures.

Per-backend build prerequisites and runtime environment variables live in [`docs/backends/`](docs/backends/).

## Why

LLM serving stacks, training runtimes, and agent dispatch loops all want the same thing at the bottom of the stack: a Rust API over the vendor driver that's *fast*, *correct*, and *unsurprising*. cudarc is the de-facto choice on CUDA, but its per-call thread-bind verification and per-buffer event-fence Drop add ~500 ns of overhead per allocation and ~30 ns per sync. At hundreds of allocs/sec in a hot dispatch loop, that adds up.

IronAccelerator gives the same surface area, faster, and lets you drop down to `iron_cuda_sys` raw FFI without giving up the safe wrappers when you need a knob the wrapper doesn't expose.

## Design principles

- **Zero link-time deps on vendor SDKs.** Every CUDA library is loaded
  through `libloading` at first use. The workspace builds on any machine,
  with or without CUDA installed; missing libraries surface as typed
  `NotAvailable` errors, not link failures.
- **Safe by default, fast by construction.** RAII wrappers over opaque
  handles; typed `Result` everywhere; `#[inline]` on every hot driver call.
- **No domain code.** This crate stops at the driver layer. Kernels,
  planners, recipes, autotuners all belong in downstream libraries. The
  surface area is small enough that an LLM agent can hold the whole API
  in context.
- **Cached driver pointers.** `Device`, `Stream`, `Event`, `Module`,
  `Function` each carry a `&'static DriverFns` reference resolved once at
  construction. Hot paths reach the function table via a struct-field load,
  not an atomic.
- **Process-global handle caches.** cuBLASLt, cuDNN, cuSOLVER, cuSPARSE,
  cuTENSOR handles are cached per-device-ordinal behind `Mutex<HashMap>`
  and stream-rebound on each borrow.
- **Errors name the operation.** `Error::Driver { op: "cuMemAllocAsync", code }`
  — no anonymous `CUDA_ERROR_*` payloads. An agent can grep for the op string
  to locate the call site.
- **Host-side overhead is measured, not assumed.** Every hot path is
  benchmarked against cudarc and the raw driver.

## Workspace layout

```shell
crates/
  ironaccelerator-core/       # cross-backend types: Error, BackendKind, dtype, …
  ironaccelerator-cuda-sys/   # clean-room CUDA 13.2 FFI + dynamic loader
  ironaccelerator-cuda/       # safe CUDA wrappers + cudarc_compat drop-in surface
  ironaccelerator-rocm-sys/   # ROCm/HIP FFI scaffold
  ironaccelerator-rocm/       # ROCm safe wrappers
  ironaccelerator-metal/      # Apple Metal/MPS scaffold
  ironaccelerator-qnn-sys/    # Qualcomm Hexagon NPU FFI scaffold
  ironaccelerator-qnn/        # Qualcomm Hexagon NPU wrappers
  ironaccelerator-vulkan/     # cross-vendor Vulkan compute
  ironaccelerator-opengl/     # legacy GL 4.3+ compute fallback
  ironaccelerator-webgpu/     # wgpu (native + browser)
  ironaccelerator-tpu/        # PJRT plugin loader
  ironaccelerator-levelzero/  # Intel oneAPI / Level Zero
  ironaccelerator-neuron/     # AWS Trainium / Inferentia
```

For CUDA users the only two crates that matter are `ironaccelerator-cuda` (safe wrappers + `cudarc_compat`) and `ironaccelerator-cuda-sys` (raw FFI re-exported as `ironaccelerator_cuda::sys`).

## CUDA backend — what's implemented

| Module                              | Coverage                                                       |
| ----------------------------------- | -------------------------------------------------------------- |
| `drv`                               | Device, Stream, Event, DeviceBuf, PinnedBuf, Module, Function, launch |
| `cudarc_compat`                     | Drop-in surface for cudarc 0.19 users (see [migration map](crates/ironaccelerator-cuda/src/cudarc_compat.rs)) |
| `kernel`                            | NVRTC compile + in-memory & on-disk PTX cache (keyed by src-hash + arch) |
| `graph`                             | Stream capture → `CUgraphExec` execute / replay                |
| `blas`                              | cuBLASLt handle cache + MatmulDesc / MatrixLayout / matmul     |
| `cudnn`                             | Handle cache + v9 backend-graph `BackendDescr` (finalize+exec) |
| `cusolver` / `cusparse` / `cutensor` / `cufft` / `curand` | Per-library handle plumbing                      |
| `nccl`                              | Single-process multi-GPU + multi-process collectives           |
| `advanced`                          | VMM (`cuMemCreate` / `cuMemMap`), green contexts, multicast teams, conditional graph nodes |
| `cupti` / `nvtx`                    | Profiler hooks                                                 |
| `alloc` / `pinned` / `streams` / `peer` / `events` / `launch` | Driver-level primitives        |

All vendor calls flow through typed `Result<T, Error>`; there are no panics in the hot path, and no allocations on success. Each error names the underlying CUDA op (`Error::Driver { op: "cuMemAllocAsync", code }`) so an agent can grep the source for what failed without reading a stack trace.

## Performance posture

### vs cudarc (drop-in replacement)

IronAccelerator's `cudarc_compat` module re-exports cudarc-shaped types and methods so existing code can switch with a `use` swap. Same reference machine (RTX 3090 Ti, CUDA 13.2 / driver 596.36, release build, 2026-05-13):

| Path                                  | IronAccelerator | cudarc 0.19  | Winner                |
| ------------------------------------- | --------------- | ------------ | --------------------- |
| Stream synchronize (empty)            | **85 ns**       | 109 ns       | Iron **1.29× faster** |
| Stream create + destroy               | **888 ns**      | 999 ns       | Iron **11 % faster**  |
| Async alloc + free (1 KB)             | **424 ns**      | 1022 ns      | Iron **2.41× faster** |
| Async alloc + free (64 KB)            | **470 ns**      | 952 ns       | Iron **2.02× faster** |
| Async alloc + free (1 MB)             | **423 ns**      | 905 ns       | Iron **2.14× faster** |
| Async alloc + free (16 MB)            | **435 ns**      | 916 ns       | Iron **2.11× faster** |
| Kernel launch (noop, 1024 thr)        | ~5.5 µs         | ~5.5 µs      | parity (FFI-bound)    |
| H→D round-trip (64 KB)                | **23.0 µs**     | 28.4 µs      | Iron **19 % faster**  |
| H→D round-trip (1 MB)                 | 135 µs          | 136 µs       | parity                |
| H→D round-trip (16 MB)                | 1.81 ms         | 1.82 ms      | parity                |
| D→H round-trip (1 KB)                 | **16.2 µs**     | 20.7 µs      | Iron **22 % faster**  |
| D→H round-trip (1 MB)                 | 300 µs          | 301 µs       | parity                |
| D→H round-trip (16 MB)                | **3.95 ms**     | 4.66 ms      | Iron **15 % faster**  |

The big wins are on control-plane operations a serving loop runs at high frequency: async alloc/free is **~2× faster** across every size, stream sync is **1.87× faster**. Four things compound:
1. An `AtomicPtr<DriverFns>` fast-path in `iron_cuda_sys::driver::fns()` collapses two `OnceLock` acquires into one acquire atomic load + null check.
2. Every `Device`, `Stream`, and `Event` caches `&'static DriverFns` at construction; `synchronize`, `alloc`, `copy_*`, `Drop`, `bind`, `attribute`, `Event::record/synchronize`, etc. dereference the cached pointer directly — zero atomic ops per call. `Device` also caches the stream priority range (one FFI amortized across every `Stream::new`).
3. Every wrapped driver call is `#[inline]`; the error-return branch is `#[cold] #[inline(never)]`, so the success path is a load/test/branch sequence with no spurious `Error`-enum materialisation.
4. `DeviceBuf::alloc` no longer heap-allocates a `String` for the overflow message on every call (the `ok_or(Error { msg: ".".into() })` pattern eagerly built a `String` the optimiser couldn't kill).
5. **What cudarc does that we don't:** `cudarc::driver::CudaStream::synchronize` calls `bind_to_thread()` first, which runs a `cuCtxGetCurrent` FFI on every call — a paranoid context check. IronAccelerator binds the device once at `Device::open` and trusts the binding to persist; if you need cudarc's behavior, call `device.bind()` explicitly.

Bulk memcpy and kernel launches sit at the FFI floor — both libraries bottleneck on the same `cuMemcpyHtoDAsync_v2` / `cuLaunchKernel` driver entries, so the wrapper differences don't surface above bench noise. The wins concentrate where the wrapper *is* the cost: control-plane sync, alloc/free, and short copies.

Reproduce:

```bash
cargo bench -p ironaccelerator-cuda --bench vs_cudarc
```

### Live-GPU functional coverage

`tests/gpu_smoke.rs` exercises the full CUDA pipeline end-to-end on the host's actual GPU (cleanly skipped on GPU-less CI):

- Device enumeration + bind across every visible ordinal
- H→D / D→H / D→D memcpy with byte-for-byte value integrity
- Event record + sync
- Raw-driver ↔ wrapper parity on memset + memcpy
- 8 concurrent streams in flight (verifies no hidden global mutex)
- **NVRTC compile + launch of a real SAXPY kernel over 1 M elements**,
  result verified against a CPU reference
- NVRTC cache identity (same source+arch → same `Arc<Module>`)

Pure-CPU hot paths (host overhead, no GPU involvement):

| Path                              | Time        |
| --------------------------------- | ----------- |
| `LaunchArgs::pack` (12 args, max) | ~2 ns       |
| FNV-1a source hash (kernel cache key) | ~1.25 GiB/s |

The wrapper overhead vs. calling the driver directly is **single-digit nanoseconds per op** — it never shows up in a kernel-launch profile.

Run the benches yourself:

```bash
cargo bench -p ironaccelerator-cuda --bench host_hot_path
cargo bench -p ironaccelerator-cuda --bench gpu_vs_cudart   # needs a GPU
cargo bench -p ironaccelerator-cuda --bench vs_cudarc       # needs a GPU
```

## Building

CUDA / ROCm / Metal / QNN toolkits are **not** required at build time —
every vendor library is loaded dynamically at first use. Missing libraries
return `Error::NotAvailable`; present ones Just Work.

```bash
# Full workspace
cargo build --release

# Just the CUDA backend
cargo build --release -p ironaccelerator-cuda
```

If the driver lives somewhere non-standard, set `IRON_CUDA_LIBDIR` to prepend a search path before the platform default.

## Roadmap

Driver-substrate work only — kernels, planners, and workload abstractions belong in downstream libraries.

- [ ] `cuMemPool` direct wrappers + per-stream pool config (currently the driver's default pool is used).
- [ ] Optional Rust-side small-buffer free list to skip `cuMemFreeAsync` round-trips at very high alloc churn.
- [ ] HIP FFI + safe wrapper for `ironaccelerator-rocm` matching the CUDA shape.
- [ ] Metal/MPS bindings via `objc2`.
- [ ] QNN SDK FFI + safe wrappers.
- [ ] Level Zero / oneAPI tighter capability probe.

## License

IronAccelerator is dual-licensed:

- **AGPL-3.0-or-later** — see [`LICENSE`](LICENSE).
- **Commercial license** — available from Nervosys for embedded, proprietary, or closed-source SaaS use cases. See [`LICENSING.md`](LICENSING.md) for the rationale and contact procedure.
