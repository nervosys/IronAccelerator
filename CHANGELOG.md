# Changelog

All notable changes to **IronAccelerator** are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org).

## [Unreleased]

### Docs

- Rewrote the README performance section into a single coherent cudarc
  comparison. The stale 1.2.0 criterion transfer table — which pre-dated the
  2.0.0 pinned-staging work and showed host→device as a cudarc win on some
  sizes — is replaced by the paired-method results (host→device 1.08–1.31×
  CI-confirmed, device→host at parity). Control-plane wins (MemPool ~90×,
  alloc/free ~2×, sync ~1.5×) kept and relabelled by measurement method. Fixed
  the broken `#performance-posture` anchor and unified the MemPool figure to
  ~90× throughout (was a mix of ~75× and ~90×).

## [2.0.0] - 2026-07-28

### Removed (breaking)

The 1.2.0 scope cut removed the kernels; this one removes the *front end* that
sat above the driver. Everything below has moved to
[IronWorks](https://github.com/nervosys/ironworks), the inference engine that
consumes IronAccelerator. IronAccelerator is now a driver substrate top to
bottom: devices, capability bits, memory, streams, events, kernel launch
geometry, runtime compile, and vendor-library handle plumbing — nothing above
that line.

**`ironaccelerator-core` — six modules deleted:**

- `workload` — `Workload`, `WorkloadKind`, `WorkloadShape`, `Precision`,
  `Phase`.
- `strategy` — `Strategy` (all 17 variants), `StrategyHint`, `StrategyScore`,
  `FlashVariant`.
- `tensor` — `TensorDesc`, `Layout`.
- `quant` — `QuantScheme`, `QuantGranularity`, `QuantParams`, `CalibStats`,
  `Fp8Format`, `fp8_scale_from_absmax`, `fp8_scale_from_history`, and the CPU
  reference quant/dequant entry points (INT8 per-tensor/per-channel sym+asym,
  packed-nibble INT4 per-group).
- `simd` — runtime-dispatched AVX2 / NEON `quant_i8_row` / `dequant_i8_row`.
- `activation` — `silu`, `silu_inplace`, `swiglu`, `swiglu_interleaved`.

**`Backend` trait narrowed to discovery.** `fn plan(&self, device, &Workload)
-> Result<Strategy>` and `fn score(&self, device, &Workload) -> f32` are gone;
the trait is now `kind` / `is_available` / `enumerate` / `capabilities`. Every
backend crate's `plan` impl was removed with it. Implementors get a smaller
trait to satisfy; callers that dispatched through `plan` must select in their
own planner.

**`ironaccelerator-ontology` deleted.** The whole crate — `Ontology`,
`HardwareNode`, `WorkloadClass`, `StrategyClass`, `Optimization`, `Edge`,
`Relation`, `Recommendation`, `FilterSpec`, `RankBy`, `Explanation`, the
`dump` binary, and the compiled-in hardware/workload/strategy graph. Ranking
kernel strategies per part is planner work. The facade's `ontology` feature
and the `agent_plan` example are gone too; `ironaccelerator/all` no longer
implies `ontology`.

**Also removed:**

- `ironaccelerator_vulkan::shader` — `wgsl_to_spirv` and `ShaderError`, along
  with the direct `naga` dependency. Shader translation is a toolchain
  concern, not a driver one; `naga` also now lives inside `wgpu` rather than
  standing alone. Build pipelines from SPIR-V you supply, via
  `compute::ComputePipeline` — the same bring-your-own-image contract the CUDA
  backend has with PTX. (`naga` still appears in `Cargo.lock` as a transitive
  dep of `wgpu`, which `ironaccelerator-webgpu` uses; that is wgpu's own
  internal use, not ours.)
- `ironaccelerator_cuda::blas::epilogue_for` and
  `ironaccelerator_rocm::blas::epilogue_for` — mapped a `Strategy` to an
  epilogue tag. Set the epilogue directly via `MatmulDesc::set_epilogue_raw`.
- `Runtime::plan`, `Runtime::plan_with`, and `Plan` on the facade.

### Added

- **`ironaccelerator-dx12` — new Direct3D 12 backend.** Covers the one GPU
  API on Windows no other backend here reached; every vendor exposes D3D12
  on Windows whether or not a Vulkan ICD is installed, and on Arm-based
  Windows devices it is often the only path. `d3d12.dll` / `dxgi.dll` load
  via `libloading` — never linked — so the crate builds on every target and
  reports unavailable where they are absent. COM is hand-written (vtables
  for `IDXGIFactory1` / `IDXGIAdapter1` / `ID3D12Device`); `windows-sys`
  dropped its Direct3D bindings in 0.60 and the `windows` crate would link
  rather than load. Enumerates hardware adapters (software/WARP filtered
  out), probes feature level, FP64, native 16-bit ops, UMA, and wave ops,
  and hands back an owned `ID3D12Device` from `drv::open`. Enabled with the
  `dx12` feature; included in `all`. New `BackendKind::Dx12`.
- **`ironaccelerator_dx12::compute` — D3D12 compute submission.**
  `Context` owns a COMPUTE queue, allocator, command list, and fence.
  Committed buffers in the DEFAULT / UPLOAD / READBACK heaps, staged
  `upload` / `download` with barriers and fence waits handled, root
  signatures (`root_signature_with_uavs`, or `root_signature_from_blob` to
  take a shader's embedded one), `compute_pipeline` from DXIL, and
  `dispatch`. Buffers bind as root UAVs rather than through descriptor
  tables, which avoids the `GetCPUDescriptorHandleForHeapStart`
  by-value-return ABI trap entirely; textures and samplers would need the
  heap path and are not supported. Shaders are not compiled here — bring
  DXIL, as the CUDA backend takes PTX. Verified end to end on 2× RTX 3090 Ti
  and an AMD integrated GPU.
- `Runtime::devices_with(CapabilityFlags)` — hardware-only filter over the
  device survey, replacing the capability half of what `plan` did.
- `Runtime::available_backends()` and `Runtime::capabilities(backend, device)`
  for live per-device capability queries.
- CI jobs for two configurations nothing was building: `no_std`
  `ironaccelerator-core` (three feature combinations) and
  `wasm32-unknown-unknown`. Both had rotted undetected — see *Fixed*.
- `ironaccelerator-webgpu` binding tests that compile for both native and
  `wasm32`, runnable in a headless browser via `wasm-bindgen-test`.
- `ironaccelerator` gains a `survey` example, replacing the `agent_plan`
  example that the planner removal deleted.

### Performance

Measured with paired back-to-back samples against cudarc 0.19 and a bootstrap
95% CI on the median per-pair ratio (`examples/ab_vs_cudarc.rs`). Ratios are
cudarc ÷ IronAccelerator, so above 1.0 is faster.

| operation | 1 KiB | 64 KiB | 1 MiB | 16 MiB |
|-----------|------:|-------:|------:|-------:|
| host→device | **1.31×** | **1.26×** | **1.08×** | **1.30×** |
| device→host | 0.99× | 1.00× | 1.00× | 1.02× |

- **Large host→device copies stage through pinned memory.** The driver cannot
  DMA out of pageable memory on a non-null stream, so it stages the transfer
  internally, and on large transfers that path is roughly an order of magnitude
  slower than doing it ourselves. cudarc never hit this because its
  `default_stream` is the legacy NULL stream, which takes the driver's
  optimised blocking path — so the earlier comparison was measuring stream
  semantics, not wrapper overhead. `DeviceBuf::copy_from_host` now routes
  multi-chunk transfers through a per-stream ring of four pinned 2 MiB chunks,
  with each chunk's host copy overlapping the previous chunk's DMA. 16 MiB went
  from ~36 ms to ~1.4 ms.
- **The staging `memcpy` is spread across a small worker pool.** Once the
  pipeline is deep enough, that copy — not the DMA — bounds the transfer: one
  core moves ~6 GB/s against a link that takes over 11 GiB/s. At most three
  extra threads, created lazily, page-aligned so no two touch the same page.
  This took 16 MiB H2D from parity to a confirmed win.
- **Synchronous copies pick their path by size.** Below two chunks there is no
  second chunk for a DMA to overlap, so staging's `memcpy` is pure cost;
  `copy_from_host_sync` uses the blocking driver copy there instead
  (211 µs vs 300 µs staged vs 477 µs pageable-async, at 1 MiB).
- **`Stream::drain` polls with `cuStreamQuery`** and only falls through to
  `cuStreamSynchronize` when the driver reports outstanding work. Sound where a
  "dirty" flag would not be: work enqueued on our stream by a vendor library
  holding our raw handle is invisible to us but visible to the driver.
- **Large device→host destinations are prefaulted in parallel.** A `Vec` handed
  back by `dtoh_sync_copy` is freshly mapped, so the copy otherwise takes
  thousands of page faults serialised against the transfer.

Device→host is at parity, and that appears to be structural rather than
unfinished. Both libraries issue one ordering check and one blocking
`cuMemcpyDtoH_v2` against a single async copy engine
(`CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT` is 1 on the test part), so there is
no wrapper work left to remove and no parallelism the hardware offers. Seven
approaches were implemented and measured before concluding this: cached and
uncached `cuMemHostRegister`, chunked D2H staging with serial and with parallel
host legs, a legacy-ordered default stream, a contiguous pinned scratch buffer,
and the buffer-reusing `dtoh_sync_copy_into` path. The reasoning for each is
recorded at its call site so it is not re-attempted.

### Removed (housekeeping)

- `ironaccelerator-core`'s `half` dependency and its `f16` feature, plus its
  `thiserror` dependency. None were referenced anywhere in the crate — the
  error type is hand-written. `ironaccelerator-cuda` keeps its own `half` and
  its own self-contained `f16` feature, which is what the CUDA `f16` paths
  actually use; nothing routed through core's copy.

### Changed (breaking)

- **`ironaccelerator-webgpu` is the browser path only, and no longer depends
  on `wgpu`.** On native it reached nothing Vulkan / Metal / D3D12 / OpenGL
  do not, while costing 98 transitive dependencies and placing a portability
  layer between this workspace and the drivers it exists to wrap. The crate
  now has *no dependencies* beyond `ironaccelerator-core`.

  Because WebGPU adapter negotiation is asynchronous and `Backend` is not,
  the host now negotiates and registers the result via
  `drv::bind_adapter(AdapterInfo)`; the `GPUDevice` stays with the host.
  `AdapterInfo` is plain data in WebGPU's own vocabulary (`vendor` and
  `architecture` are strings — WebGPU deliberately does not expose PCI IDs)
  and no longer re-exports `wgpu` types.

  Removed with it: `compute::Context`, `compute::ComputePipeline`,
  `compute::dispatch`, `Context::storage_buffer{,_init}`, and
  `drv::bind_device`. Buffers and pipelines need the live `GPUDevice`, which
  the host owns — call WebGPU directly from there.

### Fixed

- **`Backend::capabilities(device)` ignored its argument in six backends.**
  Vulkan, Level Zero, Metal, OpenGL, TPU, and Neuron returned a constant, so
  `Runtime::capabilities(kind, n)` could disagree with
  `devices()[n].capability.flags` for the same device on the same host — on a
  machine with a Vulkan ICD, `capabilities` reported `FP32 | FP16 |
  MULTI_STREAM` while `enumerate` reported `FP32 | FP16 | INT8 | WMMA |
  MULTI_STREAM`. Vulkan and Level Zero now derive both from one shared
  `flags_for`, and all six reject an ordinal they never enumerated instead of
  answering for it. A cross-backend test in `runtime_smoke` pins the agreement.
- **Vulkan over-reported `TENSOR_CORES`.** Presence of
  `VK_KHR_cooperative_matrix` set both `WMMA` and `TENSOR_CORES`. The
  extension can be implemented on general ALUs, so it does not establish
  dedicated matrix silicon; only `WMMA` is set now. Level Zero's blanket
  `TENSOR_CORES` for every Xe GPU is dropped for the same reason.
- `ironaccelerator-core` now actually compiles with
  `--no-default-features` (`no_std`). Removing the six std-heavy front-end
  modules cut this from 43 errors to 2 — missing `alloc::boxed::Box` imports in
  `memory.rs` and `stream.rs` — which are now fixed. The crate's
  `#![cfg_attr(not(feature = "std"), no_std)]` was previously aspirational; no
  CI job built that configuration. Backend crates still require `std`.
- **`ironaccelerator-webgpu` could not compile for `wasm32` at all**, despite
  being the WASM backend. `compute.rs` called `Instance::enumerate_adapters`,
  which `wgpu` gates behind `#[cfg(native)]`, with no `cfg` of its own.
  `drv.rs` gated it correctly; `compute.rs` did not. The rewrite removes the
  call entirely, and a CI job now builds the target.
- **`drv::bind_device` never affected `Context`.** Both module docs stated
  that `Context::new` would use a pre-bound device when one was present, but
  `Context::new` never read the `BOUND` cell — it always ran adapter
  negotiation. The documented WASM path was unreachable. Binding is now the
  only path, so the two cannot diverge again.
- `ironaccelerator-cuda` is clippy-clean under `-D warnings` again: elided two
  redundant `KernelArg` lifetimes, simplified the `Ptx::from_src` NUL check and
  the `from_file` UTF-8 conversion, and used `size_of_val` in
  `dtoh_sync_copy_into`. All behaviour-preserving; no signature changed.

### Migration

Consumers of the CUDA driver surface (`drv`, `kernel`, `pool`, `graph`,
`cudarc_compat`, the vendor-library modules, `sys`) are **unaffected** — no
symbol in that surface changed. This break only touches code that used
`ironaccelerator-core`'s workload/strategy vocabulary, the ontology, or the
facade's planner — pin `ironaccelerator-core = "1.2"` if you need that
vocabulary from here, or take it from IronWorks.

## [1.2.0] - 2026-06-04

### Highlights

- **Drop-in replacement for `cudarc` 0.19**, with measurably lower wrapper
  overhead on every host-side hot path (~2× faster alloc/free, ~1.3–1.9×
  faster stream sync). See README and `cudarc_compat` module docs for the
  migration map and bench numbers.
- **New `MemPool` opt-in recycling allocator** for dispatch loops:
  ~10 ns per alloc+free cycle regardless of size, vs ~740 ns for cudarc —
  **~75× faster** by skipping the `cuMemAllocAsync` round-trip entirely
  on the hot path. See `crates/ironaccelerator-cuda/src/pool.rs`.

  Three-tier cache:
    1. **Per-thread, per-bucket front cache** (4-deep fixed array, no
       lock at all — `UnsafeCell` access via a manually-`Sync` wrapper
       in `thread_local::ThreadLocal`). This is the warm path.
    2. **Shared `parking_lot::Mutex<Vec>` back cache** per bucket,
       bounded by `max_per_bucket`. Spills here when the front fills or
       a different thread allocs.
    3. **Driver** (`cuMemAllocAsync`) when both tiers are empty/full.

  `PooledBuf<'p, T>` borrows the pool with a lifetime, so the pool
  doesn't need `Arc` traffic for buffer lifetime management. Use
  `PooledBuf::into_inner()` to detach a buffer for storage beyond the
  pool's lifetime.

  New public methods on `DeviceBuf` to support the pool:
    * `truncate(new_len)` — shrink logical length without re-allocating.
    * `zero_in_place()` — stream-ordered `cuMemsetD8Async`.
    * `unsafe from_raw_parts(stream, ptr, len, capacity_bytes)` —
      rebuild a buffer from an already-allocated pointer.
    * `unsafe detach_ptr() -> CUdeviceptr` — extract the device pointer
      and suppress the buffer's `Drop` for the pointer (keeps the
      `Arc<Stream>` field's `Drop` so the stream's refcount is balanced).

  Note: `MemPool::shrink()` takes `&mut self` (not `&self`) — required
  because draining the per-thread front caches via
  `ThreadLocal::iter_mut` needs unique access. `MemPool::Drop` calls
  `shrink()` so every cached block returns to the driver.
- **Scope tightened** to a pure driver substrate. The CUDA crate no longer
  ships kernels, planners, FP8 recipes, attention/MoE implementations, or
  workload autotuners — they belong to downstream libraries. The surface is
  small enough that an LLM agent can hold the whole API in context.

### Added (CUDA completeness pass)

Six driver primitives commonly needed for "full" CUDA support that the
crate didn't yet expose. Live-GPU smoke tests for each in
`tests/driver_extras.rs` (6 tests, all pass on the reference 2× RTX 3090
Ti box).

* **`Device::uuid()` → `CUuuid`** via `cuDeviceGetUuid_v2`. Stable
  16-byte identifier per physical GPU, useful for naming devices across
  enumerations and across MIG slices.
* **`Function::attribute(attr)` / `set_attribute(attr, v)`** via
  `cuFuncGetAttribute` / `cuFuncSetAttribute`. New
  `sys::driver::CUfunction_attribute` enum exposes the 14 attribute IDs.
  Most importantly: setting `MaxDynamicSharedSizeBytes` is the only way
  to opt in to >48 KiB of dynamic shared memory per block (required for
  most FlashAttention-class kernels), and the cluster-dim attributes
  drive Hopper+ thread-block clusters.
* **`Function::occupancy_max_active_blocks_per_sm(block_size, dyn_shmem)`**
  via `cuOccupancyMaxActiveBlocksPerMultiprocessor`. Returns the
  occupancy bound; multiply by the device's `MultiprocessorCount`
  attribute for occupancy-based grid sizing. Required input for
  `launch_cooperative`.
* **`Function::launch_cooperative(cfg, stream, args)`** via
  `cuLaunchCooperativeKernel`. Same shape as `launch()` but routes
  through the cooperative-groups entry so kernels can call
  `cooperative_groups::this_grid()::sync()`. The kernel must be
  compiled with `--cooperative-groups` and the grid must fit on the
  device concurrently — use the occupancy query above.
* **`Module::global(name)` → `(CUdeviceptr, usize)`** via
  `cuModuleGetGlobal_v2`. Returns the device pointer and byte size of a
  `__constant__` or `__device__` symbol so callers can `memcpy` constants
  into the kernel's address space without a full launch.
* **`DeviceBuf::copy_from_peer_async(&src)`** via `cuMemcpyPeerAsync`.
  Cross-device memcpy bounded by `dst.len == src.len`. Verified on the
  reference dual-GPU box (RTX 3090 Ti pair) with bytes round-tripping
  correctly between devices 0 and 1.

### Added (cudarc compatibility)

- `cudarc_compat::CudaDevice::mem_get_info()` — `(free, total)` device-memory
  bytes via new `cuMemGetInfo_v2` FFI binding.
- `cudarc_compat::CudaDevice::compute_capability()` and `::device_count()`
  aliases for cudarc parity.
- `cudarc_compat::CudaStreamExt::record_event()` — combines `Event::new + record`
  in a single call, mirroring cudarc's `stream.record_event(None)`.
- `cudarc_compat::CudaStreamExt::wait(&event)` — explicit fence helper.
- `cudarc_compat::CudaStreamExt::join(&other)` — cross-stream sync via
  auto-recorded event.
- Module docs in `cudarc_compat` rewritten as a side-by-side API coverage map
  (cudarc 0.19 → IronAccelerator) plus an explicit "differences worth knowing"
  list, agent-readable in one screenful.
- 3 new live-GPU tests in `tests/cudarc_compat.rs` covering the new methods.
- Kernel-launch micro-benchmark added to `benches/vs_cudarc.rs`.
- Runnable end-to-end example at `examples/saxpy_cudarc_style.rs` —
  NVRTC compile + kernel launch + H↔D copy + verification, written to be
  byte-identical to what a cudarc user would write.
- New `pool` module: `MemPool` + `PooledBuf` for per-stream recycling
  allocation. `PooledBuf` derefs to `DeviceBuf` so every existing method
  and trait impl works unchanged. `MemPool::shrink()` drains cached blocks
  back to the driver between epochs. Live-GPU smoke test
  (`tests/pool_smoke.rs`) verifies pointer recycling and >256 MiB bypass.
- `DeviceBuf::truncate`, `zero_in_place`, and `unsafe from_raw_parts` —
  required by `MemPool` to hand out partial views of bucket-rounded
  allocations and reconstruct `DeviceBuf`s for cached pointers.

### Performance

- **`AtomicPtr<DriverFns>` hot-path cache** in `iron_cuda_sys::driver::fns()`
  collapses two `OnceLock` acquires into one Acquire atomic load + null check;
  the cold first-call path keeps the existing init logic.
- **Cached `&'static DriverFns` on every handle** — `Device`, `Stream`,
  `Event`, `Module`, `Function`, `PinnedBuf`, `CapturedGraph`, `GraphExec`
  each store the function-table reference resolved once at construction.
  Hot ops reach the table via a struct-field load, not an atomic.
- **Cached stream priority range** on `Device` (`OnceCell<(i32, i32)>`)
  amortises `cuStreamGetPriorityRange` across every `Stream::new`.
- **Cold-path error construction** — `check()` is `#[inline(always)]` with a
  `#[cold] #[inline(never)] check_err()` companion; the success branch
  compiles to load/test/branch with no Error-enum materialisation.
  `alloc_overflow()` / `pinned_alloc_overflow()` follow the same pattern.
- **Killed an eager `String` allocation** in `DeviceBuf::alloc` — the
  `ok_or(Error { msg: "size overflow".into() })` pattern heap-allocated on
  every successful call. Replaced with `let-else` + a cold helper. **64 KB
  alloc dropped from 491 ns → 340 ns.**
- `#[inline]` on every hot driver call: `Stream::{synchronize, wait_for}`,
  `DeviceBuf::{alloc, alloc_zeros, from_host, copy_from_host, copy_to_host,
  copy_from_device}`, `DeviceBuf::Drop`, `Stream::Drop`, `Event::{record,
  synchronize, Drop}`, `Function::launch`, and every `CudaStreamExt` method.
- `#[inline]` on all 12 vendor library `fns()` accessors in
  `iron_cuda_sys` so cublas / cudnn / nvrtc / etc. lookups inline at the
  call site.

### Removed (scope cleanup)

- `ironaccelerator-cuda::attention`, `flash_attention`, `moe`, `fp8`,
  `fp8_gemm`, `grouped_gemm`, `tune`, `tensor`, `session`, `backend`,
  `memcpy` modules deleted — domain code that belongs in downstream
  libraries. The `Backend` trait registration for CUDA is gone; use the
  CUDA crate directly via `ironaccelerator_cuda::drv` or
  `cudarc_compat`.
- `tests/moe_smoke.rs` deleted.

### Security

- **Release-grade security audit** committed at `docs/SECURITY_AUDIT.md`:
  0 open CVEs across 186 dependencies (cargo audit clean), 0 leaked
  private data, 0 cryptographic primitives in scope (out of scope for
  FIPS 140-3 validation), 618 `unsafe` blocks across 56 files reviewed
  against MITRE ATT&CK supply-chain / memory-corruption / DLL-hijack
  patterns, and a full CMMC 2.0 Level 2 control mapping.
- **`SECURITY.md`** vulnerability-reporting policy added — private
  channel is `opensource@nervosys.ai` or a GitHub Security Advisory.
  Acknowledgement SLA: 3 business days. Fix SLA: 30 days for high.
- **`cargo audit` wired into CI** as a dedicated `audit` job via
  `rustsec/audit-check`. Closes CMMC SI.L2-3.14.3 / RA.L2-3.11.2 by
  scanning every push and PR against the RustSec advisory database.

### Internal

- `Device`/`Stream`/`Event`/etc. methods that took `&Session` now take
  `Device`/`Stream` references directly (Session was a workload-level
  abstraction).
- `fft`, `nccl`, `pinned`, `rng`, `streams`, `graph` modules refactored to
  take their hardware handles explicitly.

### ROCm

Brought the ROCm crate up to the same performance bar as CUDA:

- `iron_rocm_sys::hip::fns()` now uses an `AtomicPtr<HipFns>` hot-path cache
  matching the CUDA pattern (one acquire load + null check on the success
  path; cold first-call goes through the `OnceLock`).
- `Device`, `Stream`, `Event`, `Module`, `Function` each cache
  `&'static HipFns` at construction. Hot ops reach the function table via a
  struct-field load.
- `check()` and `hip()` are `#[inline(always)]` with `#[cold]` error helpers
  (`check_err`, `hip_load_err`).
- `DeviceBuf::alloc` no longer heap-allocates a `String` on every success —
  same eager-`ok_or` fix as CUDA, with a cold `alloc_overflow()` helper.
- `#[inline]` on every hot path: `Stream::{synchronize, wait_for, Drop}`,
  `Event::{record, synchronize, Drop}`, `Device::{bind, attribute}`,
  `Module::Drop`, `Function::launch_raw`, `DeviceBuf::{alloc,
  copy_from_host, copy_to_host, Drop}`.

### Reference benchmarks (RTX 3090 Ti, CUDA 13.2, release build)

| op | Iron | cudarc 0.19 | Δ |
|---|---|---|---|
| stream synchronize empty | ~85 ns | ~109 ns | **1.29×** |
| stream create+destroy | ~888 ns | ~999 ns | **11%** |
| async alloc+free (all sizes) | ~430 ns | ~960 ns | **~2.0×** |
| kernel launch (noop) | ~5.5 µs | ~5.5 µs | parity (FFI-bound) |
| bulk memcpy | parity | parity | PCIe-bound |

## [1.1.0] - 2026-04-18

### Added
- **Vulkan kernels** — `shader::wgsl_to_spirv` compiles WGSL to SPIR-V at
  runtime via `naga 22`; `kernels::axpy_f32` and `kernels::gemm_f32`
  are the first real Vulkan payloads, sharing source with the WebGPU
  backend.
- **WebGPU kernels** — `kernels::GEMM_F32_WGSL` + `kernels::gemm_f32`
  dispatch a naive tiled 16×16 GEMM usable on native (Vulkan / Metal /
  DX12 / GLES) and on WASM browser adapters.
- **OpenGL kernels** — `kernels::axpy_f32` implements SAXPY against the
  glow-loaded GL 4.3+ context as a reference payload for the fallback
  backend.
- **Level Zero compute + runtime** — `compute::Context` (context +
  compute queue + command list), `Context::alloc_device` /
  `alloc_shared`, `Context::load_spirv` / `Module::kernel`, and
  `Context::launch` wire up a full `zeMemAllocDevice` +
  `zeModuleCreate` + `zeKernelCreate` + `zeCommandListAppendLaunchKernel`
  path on Intel GPUs and NPUs.
- **AWS Neuron runtime** — `runtime::Model::{load, execute}` +
  `runtime::TensorSet` wrap `nrt_load` / `nrt_execute` /
  `nrt_allocate_tensor_set` / `nrt_add_tensor_to_tensor_set` so hosts
  can run a compiled NEFF on one or more NeuronCores.
- **Core activations** — `ironaccelerator_core::activation::{silu,
  swiglu, swiglu_interleaved}` — CPU reference for LLaMA-family FFN +
  MoE fused expert blocks.
- **FP8 calibration** — `ironaccelerator_core::quant::{Fp8Format,
  fp8_scale_from_absmax, fp8_scale_from_history}` for
  transformer-engine-style delayed scaling on CUDA + ROCm paths.
- **Asymmetric INT8 ops** — `quant_i8_per_tensor_asym` +
  `dequant_i8_per_tensor_asym` back the existing
  `QuantParams::int8_per_tensor_asym`.
- **NEON quant dispatch** — `aarch64` runtime-feature-gated
  `quant_i8_row_neon` / `dequant_i8_row_neon`.
- **Umbrella integration test** (`crates/ironaccelerator/tests/
  runtime_smoke.rs`) — `Runtime::new()` + reference GEMM planner
  smoke, tolerant of hosts without vendor runtimes.
- **Docs** — top-level `README.md` backend-support matrix with minimum
  vendor-SDK versions, plus a one-file-per-backend set under
  `docs/backends/`.

### Changed
- Test count: 62 → 79 passing (17 new).



### Added
- **TPU backend (`ironaccelerator-tpu`)** — Google Cloud TPU support via the
  **PJRT** C plugin interface. Dynamically loads
  `pjrt_c_api_tpu_plugin.so` / `libtpu.so`, resolves `GetPjrtApi`, and
  enumerates one `DeviceDescriptor` per chip using the Cloud-TPU-VM
  standard env vars (`TPU_ACCELERATOR_TYPE`, `TPU_NUM_DEVICES`,
  `TPU_CHIPS_PER_HOST`). Generation detection covers v4, v5, v5e, v5p,
  and v6e (Trillium) with appropriate capability flags (`BF16`, `INT8`,
  `HBM`, `TENSOR_CORES`, plus `INT4` on v5/v5p/v6e).
- **Level Zero backend (`ironaccelerator-levelzero`)** — oneAPI Level Zero
  backend covering Intel Arc / Flex / Battlemage GPUs **and** Intel NPUs
  (Meteor / Arrow / Lunar Lake VPU). Dynamically loads `ze_loader`,
  calls `zeInit` → `zeDriverGet` → `zeDeviceGet` →
  `zeDeviceGetProperties`, and surfaces GPU vs VPU via `ze_device_type_t`
  with distinct capability flags and compute tiers.
- **Neuron backend (`ironaccelerator-neuron`)** — AWS Trainium and
  Inferentia support via `libnrt`. Loads `nrt_init`,
  `nrt_get_total_nc_count`, and `nrt_get_version`; emits one descriptor
  per NeuronCore. Generation inferred from `NEURON_INSTANCE_TYPE` +
  runtime version, mapping inf1 → Neuron v1, trn1/inf2 → v2, trn2 → v3,
  with `FP8_E4M3` / `FP8_E5M2` + `INT4` for Trainium2.
- `BackendKind::{Tpu, LevelZero, Neuron}` and
  `Vendor::{Google, Aws}` added to the core registry.
- **Core SIMD + quantization primitives** (`ironaccelerator-core::{simd, quant}`).
  - `QuantScheme` with per-tensor / per-channel / per-group granularity and
    symmetric / asymmetric symmetry; `CalibStats` for min-max calibration;
    `QuantParams` for scale + zero-point tables. CPU reference
    `quant_i8_per_channel_sym` / `dequant_i8_per_channel_sym` and packed-nibble
    `quant_u4_per_group_sym` / `dequant_u4_per_group_sym` as the oracle GPU
    dequant kernels are validated against.
  - Runtime-dispatched AVX2 `quant_i8_row` / `dequant_i8_row` with scalar
    fallback on non-x86 or pre-AVX2 hosts.
- **Vulkan backend (`ironaccelerator-vulkan`)** — `ash`-based Vulkan 1.3
  compute backend. Process-wide instance, live physical-device enumeration
  with subgroup size, FP16/INT8 shader features, and `VK_KHR_cooperative_matrix`
  detection. Vendor-ID → `Vendor` mapping; discrete/integrated → `ComputeTier`.
  Feature-gated off on `wasm32`.
- **OpenGL backend (`ironaccelerator-opengl`)** — `glow`-based GL 4.3+
  compute-shader fallback. `bind_current_context(loader)` adopts the calling
  thread's GL context; backend reports unavailable until then. Probes
  `MAX_COMPUTE_WORK_GROUP_*` limits and vendor string.
- **WebGPU backend (`ironaccelerator-webgpu`)** — `wgpu` 22 based backend
  usable both natively (routes to Vulkan/Metal/DX12/GLES) and on WASM as the
  primary browser compute path. Enumerates every adapter; `bind_device` lets
  WASM hosts preselect. Subgroup support surfaced via `CapabilityFlags::WMMA`.
- `BackendKind::{Vulkan, OpenGl, WebGpu}` variants in the core registry.

## [1.0.0] - 2026-04-18

### Added
- **QNN backend (`ironaccelerator-qnn`, `ironaccelerator-qnn-sys`)** — live
  provider-interface loader covering HTP / GPU / CPU / DSP; safe wrapper
  `Backend` / `Device` / `Context` / `Graph` with binary-blob serialise +
  rehydrate; backend enumerates each available target as its own device.
- **FlashAttention-3 via cuDNN v9** (`ironaccelerator-cuda::flash_attention`).
  Builds the full SDPA operation graph — matmul(Q,K^T) → scale → row-max →
  sub → exp → row-sum → div → matmul(P,V) — from raw
  `cudnnBackend*Descriptor*` calls on top of the v0.2 generic `BackendDescr`.
  `HEUR_MODE_A` routes Hopper to fused FA-3 kernels; Ampere to FA-2;
  pre-Ampere to composed kernels. Workspace size query + `VariantPack`
  encoding handled internally.
- **Metal backend** — `metal-rs` + `objc2` deps gated to `target_vendor =
  "apple"`. `drv::Device::all()` enumerates live `MTLDevice`s with Apple
  GPU-family detection (M3/M4 → family 9; M2 → 8; M1 → 7); `drv::Queue`,
  `drv::Buffer`, and `blas::gemm` (F32/F16 via `MPSMatrixMultiplication`)
  plus a reusable `GemmPlan`. Non-Apple hosts keep a probe-only stub.
- `cudnn::BackendDescr::get_attribute` / `get_i64` / `get_descriptor` +
  `adopt_descriptor` for externally-created descriptors.
- **Flash MoE** (`ironaccelerator-cuda::moe`) — fused Mixture-of-Experts
  forward pass. `FlashMoePlan` orchestrates: router GEMM (cuBLASLt) →
  softmax+top-K → histogram + exclusive scan → permute → per-expert
  up-proj / SiLU / down-proj GEMMs → weighted combine. NVRTC kernels
  compiled once via the process-wide cache; reusable `MoeScratch` owns all
  intermediate device buffers. FP16 I/O with FP32 accumulation; top-K ≤ 8.

### Changed
- Workspace version bumped to **1.0.0**. API surface is considered frozen
  per SemVer going forward; breaking changes require a major bump.

## [0.2.0] - 2026-04-17

### Added
- **ROCm backend (`ironaccelerator-rocm`, `ironaccelerator-rocm-sys`)**
  - Hand-written FFI for HIP runtime (`libamdhip64`), hipBLAS, hipBLASLt, RCCL.
  - Safe driver wrapper with `Device`, `Stream`, `Event`, `DeviceBuf<T>`,
    `Module` / `Function` — API-parallel to the CUDA safe layer.
  - `hipblaslt` safe wrapper: `BlasLt`, `MatmulDesc`, `MatrixLayout`,
    `Preference`, heuristic + `matmul` with FP8 scale-pointer support for
    CDNA3 (`gfx942`).
  - Backend enumerates live devices via HIP and maps `gfx` arch codes to
    `ironaccelerator_core::Capability`.
- **Criterion benchmarks** comparing the wrapped driver path against raw
  `libcuda` calls (`crates/ironaccelerator-cuda/benches/gpu_vs_cudart.rs`).
  Wrapper overhead measured at 0% for bulk d2d copy and 1–5 % on
  microsecond-scale control-plane ops.

### Fixed
- **CUDA loader symbol lookup** (`ironaccelerator-cuda-sys::loader`). The
  `libloading::Library::get` contract requires null-terminated symbol bytes;
  the previous `str::as_bytes()` path silently failed on Windows. All symbol
  lookups now go through a null-terminating helper.
- **Windows symbol fallback** in the CUDA driver. `nvcuda.dll` exports
  `cuCtxGetStreamPriorityRange` instead of `cuStreamGetPriorityRange`;
  `load_fns` now falls back to the context-scoped name when the stream-scoped
  one is absent.

### Changed
- License changed from dual MIT/Apache-2.0 to **AGPL-3.0-or-later with a
  commercial option**. See `LICENSING.md`.

## [0.1.0] - 2026-04-16

Initial public release.

### Added
- Workspace skeleton and `ironaccelerator-core` trait surface: `Backend`,
  `BackendRegistry`, `DeviceDescriptor`, `Capability`, `Workload`, `Strategy`.
- **CUDA backend (`ironaccelerator-cuda`, `ironaccelerator-cuda-sys`)**
  - Driver + runtime FFI for CUDA 13.2 via `libloading` (no `cudarc`).
  - Safe wrapper: `Device`, `Stream`, `Event`, `DeviceBuf<T>`, `Module`,
    kernel launch, graph capture, pinned / managed memory.
  - cuBLASLt safe layer with FP8 delayed-scaling support.
  - cuDNN v9, cuSOLVER, cuSPARSE, cuTENSOR, and NCCL safe wrappers.
  - Kernel cache keyed by `(ordinal, image-hash, fn-name)`.
- Scaffolds for ROCm, Metal, and QNN backends.
- Ontology / strategy layer with heuristic scoring.

[2.0.0]: https://github.com/nervosys/IronAccelerator/compare/v1.2.0...v2.0.0
[1.2.0]: https://github.com/nervosys/IronAccelerator/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/nervosys/IronAccelerator/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/nervosys/IronAccelerator/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/nervosys/IronAccelerator/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/nervosys/IronAccelerator/releases/tag/v0.1.0
