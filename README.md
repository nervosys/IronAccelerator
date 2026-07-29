<p align="center">
  <img src="media/banner.png" alt="IronAccelerator" width="100%">
</p>

A high-performance, **low-level hardware-agnostic** Rust interface over NVIDIA, AMD, Apple, Qualcomm, Intel, Google, and AWS accelerators plus the platform and cross-vendor APIs (Vulkan / Direct3D 12 / OpenGL / WebGPU). **Agent-first**: predictable shapes, terse APIs that an LLM can reason about without docs, errors that name the operation that failed.

> **Scope.** IronAccelerator is a *driver substrate*, not a kernel library.
> Each backend crate wraps the vendor driver/runtime (devices, streams,
> events, memory, kernel compile + cache, handle plumbing for vendor
> libraries like cuBLAS / cuDNN / NCCL / cuFFT). It does **not** ship
> kernels, planners, FP8 recipes, attention/MoE implementations, workload
> autotuners, workload/strategy descriptors, tensor descriptors,
> quantization schemes, CPU reference ops, or an accelerator ontology —
> those all belong to libraries layered on top, and in our stack that means
> [IronWorks](https://github.com/nervosys/ironworks).

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

**Why switch:** on the host-side hot path we're **faster than cudarc 0.19** across alloc, sync, and host→device transfer — CI-confirmed at every transfer size from 256 B to 64 MiB, up to **1.29×**, with the opt-in `MemPool` at ~70× on alloc/free ([numbers below](#performance-posture)). Device→host is at parity; both libraries are at the driver's floor there, and we say so rather than rounding it up. cudarc rebinds the thread-context on every driver call; we cache it once. cudarc tracks per-buffer event fences on `Drop`; we just call `cuMemFreeAsync`. The wins compound at high-frequency dispatch loops.

**Production user:** [IronWorks](https://github.com/nervosys/ironworks) (a Rust LLM inference engine) completed a full cudarc → IronAccelerator migration on 2026-05-15, dropping cudarc from `Cargo.lock` entirely. ~300 call sites migrated; zero kernel regressions; tg128 on Llama-3.2-1B Q4_K_M unchanged at ~522 tok/s (within ±1.3% of the cudarc baseline — iwx is kernel-bound, so the wrapper-level wins amortise to noise). The migration validated that every iwx CUDA-driver call site has a native IA equivalent.

## Backend support matrix

Honest current state. **CUDA is the only backend that's production-ready today.** Everything else compiles, registers, and enumerates devices where the SDK is present — and five of them (Vulkan, D3D12, OpenGL, Metal, Level Zero) also run compute through the unified [`ComputeDevice` trait](#cross-vendor-compute--one-trait-five-drivers) — but the per-backend hot path has not yet had the same optimization sprint that pushed the CUDA `MemPool` to ~70× faster than cudarc on alloc/free. Detailed gap analysis per backend lives in [`docs/backends/STATUS.md`](docs/backends/STATUS.md).

| Backend    | Vendor / API                       | Driver wrappers | Runtime kernel compile | cudarc-shaped compat | `MemPool` equivalent | Live-GPU tests | Min SDK / runtime               |
| ---------- | ---------------------------------- | --------------- | ---------------------- | -------------------- | -------------------- | -------------- | ------------------------------- |
| **CUDA**   | NVIDIA                             | ✅ full          | ✅ NVRTC + disk cache  | ✅ `cudarc_compat`   | ✅ `MemPool` (~70× cudarc) | ✅ 45 tests | CUDA 12.5+ driver (13.x tested) |
| ROCm       | AMD                                | ✅ HIP full      | ⏳ HIPRTC pending      | ❌                   | ❌                   | ❌ no AMD GPU on CI host | ROCm 6.2+                |
| Metal      | Apple                              | ✅ enumerate + compute | ❌ bring your own metallib | ❌            | ❌                   | ⏳ cross-checks; needs macOS to run | macOS 14+ / iOS 17+ |
| Vulkan     | cross-vendor GPU compute           | ✅ enumerate + compute | ❌ bring your own SPIR-V | ❌              | ❌                   | ✅ dispatch on 3 devices | Vulkan 1.3 ICD          |
| QNN        | Qualcomm Hexagon NPU               | ⚠️ scaffold      | n/a (QNN is AOT)       | ❌                   | ❌                   | ❌ needs SDK + device | QNN SDK 2.22+              |
| OpenGL     | legacy / embedded GPU fallback     | ✅ enumerate + compute | ⏳              | ❌                   | ❌                   | ✅ dispatch (WGL 4.3) | GL 4.3+ compute            |
| **D3D12**  | Windows, all vendors               | ✅ enumerate + compute | ❌ bring your own DXIL | ❌              | ❌                   | ✅ dispatch on 3 adapters | Windows 10 1507+ |
| WebGPU     | browser / WASM only                | ✅ host-bound adapter   | n/a (host owns device) | ❌              | ❌                   | ⏳ needs browser harness | Chrome 113+ / Safari 17.4+ |
| TPU (PJRT) | Google TPU v4 / v5 / v6e           | ⚠️ env probe     | n/a (PJRT plugin AOT)  | ❌                   | ❌                   | ❌ needs TPU VM | PJRT plugin (`libtpu.so`)      |
| Level Zero | Intel GPU (Arc / Flex / PVC) + NPU | ✅ enumerate + compute | ❌ bring your own SPIR-V | ❌          | ❌                   | ⏳ builds; needs Intel GPU to run | `ze_loader` from Intel compute |
| AWS Neuron | Trainium / Inferentia              | ⚠️ cores probe   | n/a (NEFF AOT)         | ❌                   | ❌                   | ❌ needs trn/inf instance | `libnrt` (Neuron SDK 2.x)  |

Legend:
- ✅ shipped and exercised against a real device (or the closest equivalent — Vulkan ICD probe, WGSL compile, etc.)
- ⚠️ scaffold compiles and registers, does device enumeration only
- ⏳ noted in code, work pending
- ❌ not present
- n/a not meaningful for this backend (e.g. Metal Shading Language compile happens offline by design)

Every backend is loaded via `libloading` at first use — the workspace builds on any host, with or without the vendor SDK. Missing runtimes surface as typed `NotAvailable` errors, not link failures.

If you're shopping for "drop-in cudarc replacement", use the CUDA backend; that's the whole point of the project today. If you need ROCm/Metal/Vulkan/QNN parity with the CUDA backend's posture, [`STATUS.md`](docs/backends/STATUS.md) lists what remains per backend — none of the non-CUDA backends are at production-ready quality yet, and most need their target hardware in CI to validate further work.

Per-backend build prerequisites and runtime environment variables live in [`docs/backends/`](docs/backends/).

## Cross-vendor compute — one trait, five drivers

Beyond enumeration, the Vulkan, D3D12, OpenGL, Metal, and Level Zero backends
share a single compute-submission surface: the `ComputeDevice` trait in
`ironaccelerator-core`. Write the submission logic once and it runs on any of
them — only the shader bytecode differs (SPIR-V, DXIL, GLSL, or a `.metallib`,
whichever the driver consumes; there is no translation layer, matching the
driver-line scope).

```rust
use ironaccelerator::prelude::*;

// Backend-agnostic: identical body for Vulkan, D3D12, OpenGL, Metal, or Level Zero.
fn double_in_place<C: ComputeDevice>(dev: &C, code: &[u8]) -> Result<Vec<f32>, C::Error> {
    let input: Vec<u8> = (0..256u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let buf = dev.upload(&input)?;                     // host → device-local
    let pipe = dev.pipeline(code, 1)?;                 // one storage buffer at slot 0
    dev.dispatch(&pipe, &[&buf], [256 / 64, 1, 1])?;   // run + wait
    let mut out = vec![0u8; input.len()];
    dev.download(&buf, &mut out)?;                     // device → host
    Ok(out.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}
```

The trait uses associated `Buffer` / `Pipeline` / `Error` types, so it stays
zero-cost — no boxing, no vtable — and `no_std`-clean. A single generic routine
is verified doubling a buffer on real hardware across Vulkan, D3D12, and OpenGL
(`unified_compute` over Vulkan + D3D12 on 2× RTX 3090 Ti + AMD iGPU + D3D12
WARP; `live_compute` on an OpenGL 4.3 context). The Metal and Level Zero impls
compile-check (Metal cross-checked for `aarch64-apple-darwin`, Level Zero built
natively) but are not run here — this workspace has no Apple or Intel-GPU host.
WebGPU intentionally sits the trait out: its `GPUDevice` is owned by the host
page and driven asynchronously from JS (buffer readback is `mapAsync` → a
Promise), so it cannot satisfy the synchronous trait without an async variant.

Metal and Level Zero note: both set threadgroup size at dispatch, not in the
shader, so the trait's `dispatch` assumes a 1-D group of 64 threads;
`metal::Context::dispatch_sized` / `levelzero::Kernel::set_group_size` +
`Context::launch` take an explicit size for other geometries.

## Why

LLM serving stacks, training runtimes, and agent dispatch loops all want the same thing at the bottom of the stack: a Rust API over the vendor driver that's *fast*, *correct*, and *unsurprising*. cudarc is the de-facto choice on CUDA, but its per-call thread-bind verification and per-buffer event-fence Drop add ~350 ns of overhead per allocation and ~30 ns per sync. At hundreds of allocs/sec in a hot dispatch loop, that adds up.

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
  ironaccelerator-metal/      # Apple Metal compute (ComputeDevice)
  ironaccelerator-qnn-sys/    # Qualcomm Hexagon NPU FFI scaffold
  ironaccelerator-qnn/        # Qualcomm Hexagon NPU wrappers
  ironaccelerator-vulkan/     # cross-vendor Vulkan compute
  ironaccelerator-opengl/     # legacy GL 4.3+ compute fallback
  ironaccelerator-dx12/       # Direct3D 12 (Windows, all vendors)
  ironaccelerator-webgpu/     # browser/WASM path, host-bound
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

**Recent additions (2026-05-15) to support the IronWorks migration:**

- **cuBLAS classic perf primitives**: `cublasSetMathMode` (required to enable tensor-core HGEMM on the FP16 prefill hot path — without it cuBLAS falls back to CUDA cores) + `cublasSetWorkspace_v2` (lets you bind a persistent workspace so cuBLAS picks better algorithms).
- **Graph exec update fast path**: `cuGraphExecUpdate_v2` for in-place re-application of a re-captured graph to an existing exec (~µs vs ~10× slower full re-instantiate). Plus `cuGraphGetNodes` / `cuGraphNodeGetType` for topology inspection.
- **NVRTC fast-math options** on `kernel::CompileOptions`: `ftz`, `prec_div`, `prec_sqrt`, `fmad`, `use_fast_math`, `maxrregcount` as structured fields that map to the corresponding NVRTC flags. (~5-10 % kernel speedup on Ampere from `ftz=true` + `prec_div=false`.)
- **PTX wrapper semantics**: `Ptx::from_src` wraps already-compiled PTX text without re-invoking NVRTC (matches cudarc 0.12 / 0.19 behaviour) — required for on-disk PTX cache reload paths.
- **Default-mempool retain-on-free**: `Device::open` sets `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = u64::MAX` on the default stream-ordered pool. Matches PyTorch/cudarc behaviour.
- **cudarc-style associated constants** on every CUDA enum that bindgen would produce with `_t`/`_enum` suffixes: `CUBLAS_OP_N`, `CUDA_R_32F`, `CU_FUNC_ATTRIBUTE_*`, `CU_DEVICE_ATTRIBUTE_*`, `CU_STREAM_CAPTURE_MODE_*`, `CU_GRAPH_NODE_TYPE_*`, etc. Lets migration code keep the bindgen-style paths.
- **`KernelArg for &DeviceBuf<T>`**: pass slice references directly as kernel arguments (cudarc-style ergonomics).
- **`DeviceBuf::slice(Range<usize>)`**: cudarc-style range slicing into a `DeviceView`.
- **`Function: Clone`**: enables function-handle caches (module loader maps `(module, fn_name) → Function`).

All vendor calls flow through typed `Result<T, Error>`; there are no panics in the hot path, and no allocations on success. Each error names the underlying CUDA op (`Error::Driver { op: "cuMemAllocAsync", code }`) so an agent can grep the source for what failed without reading a stack trace.

## Performance posture

`cudarc_compat` re-exports cudarc-shaped types and methods so existing code
switches with a `use` swap — and it is faster on the host-side hot path. All
numbers below are on an **RTX 3090 Ti, CUDA 13.2**, against **cudarc 0.19.6**.

### Data transfer — paired method, 95% CI

Transfer benchmarks are dominated by machine state (GPU clock float, PCIe power
state, contention). Measuring each library in its own block — the criterion
default — lets that drift land on one side and masquerade as a code difference;
early runs of this suite swung a single path between 0.66× and 1.98× on
identical code. [`examples/ab_vs_cudarc.rs`](crates/ironaccelerator-cuda/examples/ab_vs_cudarc.rs)
instead samples the two libraries **back-to-back** and reports the median
**per-pair ratio** with a bootstrap 95% confidence interval, so shared drift
cancels. A win is only claimed when the whole interval sits above 1.0.

Ratios are cudarc ÷ IronAccelerator — **above 1.0 means IronAccelerator is
faster.** The table sweeps 256 B → 64 MiB, three transfer shapes per size, and
every figure below reproduced across independent runs on the idle GPU (the
`h2d` verdicts were CI-confirmed in every run):

| operation                | 256 B | 1 KiB | 64 KiB | 256 KiB | 1 MiB | 4 MiB | 16 MiB | 64 MiB |
| ------------------------ | ----: | ----: | -----: | ------: | ----: | ----: | -----: | -----: |
| **host→device**          | **1.29×** | **1.29×** | **1.20×** | **1.10×** | **1.08×** | **1.12×** | **1.12×** | **1.28×** |
| device→host (alloc)      | 1.00× | 0.99× | 1.00×  | 1.00×   | 1.00× | 1.00× | 0.98×  | 1.00×  |
| device→host (into buf)   | 0.98× | 0.98× | 0.99×  | 1.00×   | 1.00× | 1.00× | 1.00×  | 1.00×  |

**Host→device wins at every size, CI-confirmed.** Two mechanisms drive the
curve. At the small end (≤1 KiB, ~1.29×) the win is pure per-call overhead:
one cached-pointer copy call versus cudarc's clone-then-synchronize path. The
curve dips to ~1.08–1.10× in the 256 KiB–1 MiB PCIe-bound band, where the
transfer itself dominates and there is little wrapper left to shave. Above that
(~1.12× through 16 MiB, ~1.28× at 64 MiB) the pinned-staging path takes over:
the driver cannot DMA out of pageable memory on a non-null stream, so it stages
internally, and its internal path is slower than doing it ourselves.
`copy_from_host` routes multi-chunk transfers through four pinned 2 MiB chunks,
overlaps each chunk's host copy with the previous chunk's DMA, and spreads the
host copy across a small worker pool. 16 MiB H→D dropped from ~36 ms (pageable)
to ~1.6 ms on a quiet device.

**Device→host is at parity across the whole sweep, and we say so rather than
round it up.** Both the allocating (`dtoh_sync_copy`) and buffer-reusing
(`dtoh_sync_copy_into`) paths land in a 0.97–1.02× band — measurement noise
around the driver floor, not a code difference. Both libraries issue one
ordering check and one blocking `cuMemcpyDtoH_v2` against a single async copy
engine (`ASYNC_ENGINE_COUNT = 1` on this part), so there is no wrapper work left
to remove and no second engine to parallelise over. Seven approaches — pinned
staging, host-side registration, a legacy-ordered stream, the buffer-reusing
API — were implemented and measured before concluding this; each is documented
at its call site so it is not re-attempted. (A single contended run once showed
a spurious 1.27× at 16 MiB; the paired method flags such rows, and it did not
reproduce on the idle device.)

Reproduce (prints device state and flags any row whose CI is too wide to trust;
pin to the idle GPU to keep drift out of the absolute timings):

```bash
CUDA_VISIBLE_DEVICES=1 cargo run --release -p ironaccelerator-cuda --example ab_vs_cudarc
```

### Control plane — where the wrapper *is* the cost

Alloc/free, sync, and stream/event lifecycle are where a thin wrapper wins or
loses, because the driver call underneath is cheap and the overhead is the
wrapper. Criterion medians, idle GPU, cudarc 0.19.6 (these are unpaired, so
treat them as indicative rather than to two significant figures):

| Path                                      | IronAccelerator | cudarc 0.19 | Speedup       |
| ----------------------------------------- | --------------: | ----------: | ------------- |
| **`MemPool` alloc+free (warm)**           |      **~10 ns** |    ~0.7 µs  | **~70×**      |
| Async alloc + free (1 KB)                 |      **365 ns** |      692 ns | **1.9×**      |
| Async alloc + free (1 MB)                 |      **367 ns** |      744 ns | **2.0×**      |
| Async alloc + free (16 MB)                |      **358 ns** |      724 ns | **2.0×**      |
| Stream synchronize (empty)                |       **58 ns** |       86 ns | **1.5×**      |
| Event create+record+sync+destroy         |      **215 ns** |      326 ns | **1.5×**      |
| Kernel launch (noop, 1024 thr)            |      ~4.7 µs    |    ~4.6 µs  | ~1.0× (parity) |
| Stream create + destroy                   |      **699 ns** |      754 ns | **1.08×**     |

The headline is the optional [`MemPool`](crates/ironaccelerator-cuda/src/pool.rs):
a per-stream freelist recycling `DeviceBuf` allocations into power-of-two
buckets. The hot path is a thread-local front cache — `UnsafeCell` access plus a
fixed-array index, **no FFI, no mutex** — at ~10 ns per alloc/free cycle.
Cross-thread overflow falls through to a `parking_lot::Mutex`-guarded shared
cache; over-cap allocations go to `cuMemFreeAsync`. For a dispatch loop churning
thousands of small buffers per second (KV-cache slots, per-token scratch), that
is **~70× cudarc** while keeping the same `DeviceBuf` API through `Deref`.
Plain `DeviceBuf::alloc` still goes straight to `cuMemAllocAsync`.

The control-plane wins come from four compounding choices:

1. An `AtomicPtr<DriverFns>` fast path in `iron_cuda_sys::driver::fns()` — one
   acquire load + null check instead of two `OnceLock` acquires.
2. Every `Device` / `Stream` / `Event` caches `&'static DriverFns` at
   construction, so `synchronize` / `alloc` / `copy_*` / `Drop` dereference a
   cached pointer with zero atomics per call.
3. Every wrapped driver call is `#[inline]` with a `#[cold]` error branch, so
   the success path is load / test / branch with no `Error` materialisation.
4. **What cudarc does that we don't:** `CudaStream::synchronize` calls
   `bind_to_thread()` (`cuCtxGetCurrent`) on every call. We bind once at
   `Device::open` and trust it to persist; call `device.bind()` if you switch
   contexts manually.

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

- [x] Scope cut completed: workload/strategy descriptors, tensor descriptors,
      quantization + CPU reference ops, the heuristic planner, and the
      `ironaccelerator-ontology` crate moved to
      [IronWorks](https://github.com/nervosys/ironworks). `Backend` is now a
      discovery-only trait and the facade `Runtime` is a device survey.
- [x] `cuMemPool` direct wrappers + default-pool retention (`Device::open` sets `ReleaseThreshold = u64::MAX` on the default pool — retains memory across free/alloc, matching PyTorch/cudarc behaviour). Per-stream custom pools still TODO.
- [ ] Optional Rust-side small-buffer free list to skip `cuMemFreeAsync` round-trips at very high alloc churn.
- [x] Unified `ComputeDevice` trait (`ironaccelerator-core`) implemented across
      Vulkan, D3D12, OpenGL, Metal, and Level Zero — one submission surface,
      backend-native bytecode, no translation layer.
- [x] Metal compute bindings via the `metal` crate (`objc2`). The MPS-backed
      GEMM was removed as workload-level — that belongs above the driver line.
- [ ] HIP FFI + safe wrapper for `ironaccelerator-rocm` matching the CUDA shape.
- [ ] QNN SDK FFI + safe wrappers.
- [ ] Level Zero / oneAPI tighter capability probe (COMPUTE queue-group query).

## License

IronAccelerator is dual-licensed:

- **AGPL-3.0-or-later** — see [`LICENSE`](LICENSE).
- **Commercial license** — available from NERVOSYS for embedded, proprietary, or closed-source SaaS use cases. See [`LICENSING.md`](LICENSING.md) for the rationale and contact procedure.
