# Changelog

All notable changes to **IronAccelerator** are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org) with the caveat that we remain pre-1.0 — minor
versions may break API.

## [Unreleased]

### Highlights

- **Drop-in replacement for `cudarc` 0.19**, with measurably lower wrapper
  overhead on every host-side hot path (~2× faster alloc/free, ~1.3–1.9×
  faster stream sync). See README and `cudarc_compat` module docs for the
  migration map and bench numbers.
- **Scope tightened** to a pure driver substrate. The CUDA crate no longer
  ships kernels, planners, FP8 recipes, attention/MoE implementations, or
  workload autotuners — they belong to downstream libraries. The surface is
  small enough that an LLM agent can hold the whole API in context.

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

[1.1.0]: https://github.com/nervosys/IronAccelerator/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/nervosys/IronAccelerator/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/nervosys/IronAccelerator/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/nervosys/IronAccelerator/releases/tag/v0.1.0
