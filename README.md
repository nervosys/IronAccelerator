# IronAccelerator

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

**Why switch:** on the host-side hot path we're **faster than cudarc 0.19** across alloc, sync, and host→device transfer — CI-confirmed at every transfer size, up to **1.31×**, with the opt-in `MemPool` at ~90× on alloc/free ([numbers below](#vs-cudarc-drop-in-replacement)). Device→host is at parity; both libraries are at the driver's floor there, and we say so rather than rounding it up. cudarc rebinds the thread-context on every driver call; we cache it once. cudarc tracks per-buffer event fences on `Drop`; we just call `cuMemFreeAsync`. The wins compound at high-frequency dispatch loops.

**Production user:** [IronWorks](https://github.com/nervosys/ironworks) (a Rust LLM inference engine) completed a full cudarc → IronAccelerator migration on 2026-05-15, dropping cudarc from `Cargo.lock` entirely. ~300 call sites migrated; zero kernel regressions; tg128 on Llama-3.2-1B Q4_K_M unchanged at ~522 tok/s (within ±1.3% of the cudarc baseline — iwx is kernel-bound, so the wrapper-level wins amortise to noise). The migration validated that every iwx CUDA-driver call site has a native IA equivalent.

## Backend support matrix

Honest current state. **CUDA is the only backend that's production-ready today.** Everything else compiles, registers, and enumerates devices where the SDK is present — but the per-backend hot path has not yet had the same optimization sprint that pushed CUDA to ~75× faster than cudarc. Detailed gap analysis per backend lives in [`docs/backends/STATUS.md`](docs/backends/STATUS.md).

| Backend    | Vendor / API                       | Driver wrappers | Runtime kernel compile | cudarc-shaped compat | `MemPool` equivalent | Live-GPU tests | Min SDK / runtime               |
| ---------- | ---------------------------------- | --------------- | ---------------------- | -------------------- | -------------------- | -------------- | ------------------------------- |
| **CUDA**   | NVIDIA                             | ✅ full          | ✅ NVRTC + disk cache  | ✅ `cudarc_compat`   | ✅ `MemPool` (~75× cudarc) | ✅ 45 tests | CUDA 12.5+ driver (13.x tested) |
| ROCm       | AMD                                | ✅ HIP full      | ⏳ HIPRTC pending      | ❌                   | ❌                   | ❌ no AMD GPU on CI host | ROCm 6.2+                |
| Metal      | Apple                              | ⚠️ scaffold      | n/a (MSL is offline)   | ❌                   | ❌                   | ❌ needs macOS | macOS 14+ / iOS 17+              |
| Vulkan     | cross-vendor GPU compute           | ✅ enumerate + compute | ❌ bring your own SPIR-V | ❌              | ❌                   | ⏳ device probe only | Vulkan 1.3 ICD             |
| QNN        | Qualcomm Hexagon NPU               | ⚠️ scaffold      | n/a (QNN is AOT)       | ❌                   | ❌                   | ❌ needs SDK + device | QNN SDK 2.22+              |
| OpenGL     | legacy / embedded GPU fallback     | ✅ enumerate     | ⏳                     | ❌                   | ❌                   | ⏳ context probe only | GL 4.3+ compute            |
| **D3D12**  | Windows, all vendors               | ✅ enumerate + compute | ❌ bring your own DXIL | ❌              | ❌                   | ✅ dispatch on 3 adapters | Windows 10 1507+ |
| WebGPU     | browser / WASM only                | ✅ host-bound adapter   | n/a (host owns device) | ❌              | ❌                   | ⏳ needs browser harness | Chrome 113+ / Safari 17.4+ |
| TPU (PJRT) | Google TPU v4 / v5 / v6e           | ⚠️ env probe     | n/a (PJRT plugin AOT)  | ❌                   | ❌                   | ❌ needs TPU VM | PJRT plugin (`libtpu.so`)      |
| Level Zero | Intel GPU (Arc / Flex / PVC) + NPU | ✅ enumerate + compute | ⏳ SPIR-V        | ❌                   | ❌                   | ⏳ device probe only | `ze_loader` from Intel compute |
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

### vs cudarc (drop-in replacement)

IronAccelerator's `cudarc_compat` module re-exports cudarc-shaped types and methods so existing code can switch with a `use` swap. Same reference machine (RTX 3090 Ti, CUDA 13.2 / driver 596.36, release build, 2026-05-15 — measured against cudarc 0.19.6, criterion 0.5, 100 samples × 5 s each):

| Path                                  | IronAccelerator | cudarc 0.19  | Winner                |
| ------------------------------------- | --------------- | ------------ | --------------------- |
| **`MemPool` alloc+free (any size, warm)** | **~10 ns** | ~1 µs        | Iron **~100× faster** |
| Stream synchronize (empty)            | **84 ns**       | 124 ns       | Iron **1.47× faster** |
| Stream create + destroy               | **902 ns**      | 1.03 µs      | Iron **1.14× faster** |
| Async alloc + free (1 KB)             | **461 ns**      | 1.05 µs      | Iron **2.27× faster** |
| Async alloc + free (64 KB)            | **465 ns**      | 1.03 µs      | Iron **2.21× faster** |
| Async alloc + free (1 MB)             | **463 ns**      | 767 ns       | Iron **1.66× faster** |
| Async alloc + free (16 MB)            | **470 ns**      | 1.04 µs      | Iron **2.21× faster** |
| Event create + record + sync + destroy| **26.2 µs**     | 26.3 µs      | Iron **edge** ¹       |
| Kernel launch (noop, 1024 thr)        | **5.78 µs**     | 6.15 µs      | Iron **1.06× faster** |
| H→D round-trip (1 KB)                 | 33.6 µs         | 28.9 µs      | cudarc 1.16× (noise)  |
| H→D round-trip (64 KB)                | **23.3 µs**     | 32.1 µs      | Iron **1.38× faster** |
| H→D round-trip (1 MB)                 | **145.5 µs**    | 146.8 µs     | Iron edge             |
| H→D round-trip (16 MB)                | **1.96 ms**     | 2.39 ms      | Iron **1.22× faster** |
| D→H round-trip (1 KB)                 | **18.2 µs**     | 31.3 µs      | Iron **1.72× faster** |
| D→H round-trip (64 KB)                | 40.4 µs         | 29.7 µs      | cudarc 1.36× ²        |
| D→H round-trip (1 MB)                 | **335.7 µs**    | 358.5 µs     | Iron **1.07× faster** |
| D→H round-trip (16 MB)                | 4.46 ms         | 4.33 ms      | cudarc 1.03× (noise)  |

### 2.0.0 re-measurement (paired method)

The table above is a criterion run, which measures each library in its own
contiguous block — so machine drift lands on one side and shows up as a
difference that is not in the code. On a shared desktop that effect exceeded the
differences being measured. 2.0.0 adds
[`examples/ab_vs_cudarc.rs`](crates/ironaccelerator-cuda/examples/ab_vs_cudarc.rs),
which samples the two libraries **back-to-back** and reports the median
per-pair ratio with a bootstrap 95% CI, so shared contention cancels. Verdicts
are only issued when the interval excludes 1.0.

| operation | 1 KiB | 64 KiB | 1 MiB | 16 MiB |
|-----------|------:|-------:|------:|-------:|
| host→device | **1.31×** | **1.26×** | **1.08×** | **1.30×** |
| device→host | 0.99× | 1.00× | 1.00× | 1.02× |

H2D improved across the board in 2.0.0 because large pageable copies now stage
through pinned memory — see the CHANGELOG's *Performance* section. **Device→host
is at parity, not a win**, and that looks structural: both libraries issue one
ordering check and one blocking `cuMemcpyDtoH_v2` against a single async copy
engine, so there is no wrapper work left to remove. Seven approaches were
implemented and measured before concluding that.

**16 of 20 IA wins in the 1.2.0 criterion run above.** The 4 cudarc-leaning rows are all bandwidth-bound transfers where the wrapper layer is sub-dominant; ratios fluctuate ±15 % run-to-run due to Windows GPU clock float (1740–2100 MHz, no NVAPI lock). cudarc 0.19 source ([`core.rs:1550`](https://github.com/coreylowman/cudarc/blob/main/src/driver/safe/core.rs)) issues identical `cuMemAllocAsync` + `cuMemcpy*Async_v2` + `cuStreamSynchronize` calls on these paths — both libraries are at the same hardware ceiling.

¹ Event lifecycle was 1.14× cudarc before 2026-05-15. Removed a redundant `device.bind()` (`cuCtxSetCurrent`) in `Event::new_impl` since IA's invariant guarantees binding persists from `Device::open`. Cut from 36 µs → 18 µs and now beats cudarc.

² D→H 64 KB occasionally hits a single-run outlier; median across multiple runs is 28-32 µs IA vs 29-32 µs cudarc — statistical tie.

**Recent wrapper-perf improvements (2026-05-15):**
- `Event::new_impl` skips per-event `cuCtxSetCurrent` (event lifecycle 36 µs → 26 µs)
- `dtoh_sync_copy` switched from `cuMemcpyDtoHAsync_v2 + cuStreamSynchronize` to the synchronous `cuMemcpyDtoH_v2` (D→H 1 MB 438 µs → 335 µs)
- `Device::open` auto-sets the default mempool's `ReleaseThreshold = u64::MAX` — retains memory across free/alloc cycles, +1-5 % on tight alloc/free loops

The headline win is the optional [`MemPool`](crates/ironaccelerator-cuda/src/pool.rs): a per-stream Rust-side freelist that recycles `DeviceBuf` allocations into power-of-two size buckets. The hot path is a thread-local front cache — `UnsafeCell` access + a fixed-array index — **no FFI, no mutex**, landing at ~10 ns per alloc/free cycle on the common single-thread case. Cross-thread overflow falls through to a `parking_lot::Mutex`-guarded shared back cache; over-cap allocations go all the way to `cuMemFreeAsync`. For a dispatch loop that churns thousands of small buffers per second (KV-cache slots, per-token scratch, agent tool churn), that's roughly **75× faster than cudarc** while preserving the same `DeviceBuf` API through `Deref`. Default `DeviceBuf::alloc` still goes straight to `cuMemAllocAsync` for one-off cases. The other wins on the control plane (alloc, sync, create) come from these compounding optimizations: Four things compound:
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

## Telemetry

The `ironaccelerator` facade builds with an **opt-out** OpenTelemetry exporter,
enabled by the default `telemetry` feature. It is **off unless you configure a
destination**, and it ships **no endpoint and no credential** — nothing is
transmitted anywhere by default, and nothing happens at build or install time
(there is no build script).

When `ironaccelerator::init()` / `Runtime::new()` runs, it installs an OTLP
span exporter **only if** `OTEL_EXPORTER_OTLP_ENDPOINT` is set in the process
environment. Endpoint, auth header, protocol, and service name are read at
runtime from the standard OpenTelemetry variables — you point it at your own
collector with your own token:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otel.your-org.example/otlp"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer <your-token>"
export OTEL_SERVICE_NAME="your-service"
```

Turn it off completely, three ways:

- **Remove the dependency:** `ironaccelerator = { version = "2", default-features = false, features = [...] }` drops the exporter and its whole dependency tree.
- **Disable at runtime:** `IRONACCEL_TELEMETRY=off`.
- **Leave `OTEL_EXPORTER_OTLP_ENDPOINT` unset** — the default — and it never opens a connection.

It exports to whatever destination the process operator configures, and to no
other. It does not report to the crate authors, and there is no destination it
can reach that you did not set yourself.

## Roadmap

Driver-substrate work only — kernels, planners, and workload abstractions belong in downstream libraries.

- [x] Scope cut completed: workload/strategy descriptors, tensor descriptors,
      quantization + CPU reference ops, the heuristic planner, and the
      `ironaccelerator-ontology` crate moved to
      [IronWorks](https://github.com/nervosys/ironworks). `Backend` is now a
      discovery-only trait and the facade `Runtime` is a device survey.
- [x] `cuMemPool` direct wrappers + default-pool retention (`Device::open` sets `ReleaseThreshold = u64::MAX` on the default pool — retains memory across free/alloc, matching PyTorch/cudarc behaviour). Per-stream custom pools still TODO.
- [ ] Optional Rust-side small-buffer free list to skip `cuMemFreeAsync` round-trips at very high alloc churn.
- [ ] HIP FFI + safe wrapper for `ironaccelerator-rocm` matching the CUDA shape.
- [ ] Metal/MPS bindings via `objc2`.
- [ ] QNN SDK FFI + safe wrappers.
- [ ] Level Zero / oneAPI tighter capability probe.

## License

IronAccelerator is dual-licensed:

- **AGPL-3.0-or-later** — see [`LICENSE`](LICENSE).
- **Commercial license** — available from NERVOSYS for embedded, proprietary, or closed-source SaaS use cases. See [`LICENSING.md`](LICENSING.md) for the rationale and contact procedure.
