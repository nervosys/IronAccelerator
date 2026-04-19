# Changelog

All notable changes to **IronAccelerator** are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org) with the caveat that we remain pre-1.0 — minor
versions may break API.

## [Unreleased]

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
