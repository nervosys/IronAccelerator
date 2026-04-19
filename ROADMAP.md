# IronAccelerator — Roadmap

Live progress tracker. Each section is a milestone; each bullet is a
deliverable with a status box. Move items to **Shipped** with the version
they land in. When all **Path to 1.1** boxes are ticked, we cut 1.1.

Last updated: **2026-04-18**

---

## Status snapshot

| backend        | crate                         | enumerate | compute scaffold | real kernels |
|----------------|-------------------------------|-----------|------------------|--------------|
| CUDA           | `ironaccelerator-cuda`        | ✅        | ✅               | ✅ (BLAS / cuDNN / FA-3 / MoE) |
| ROCm           | `ironaccelerator-rocm`        | ✅        | ✅               | ✅ (hipBLASLt) |
| Metal          | `ironaccelerator-metal`       | ✅ Apple  | ✅ Apple         | ✅ (MPS GEMM) |
| QNN (Hexagon)  | `ironaccelerator-qnn`         | ✅        | ⚠️ needs HDK     | — |
| Vulkan         | `ironaccelerator-vulkan`      | ✅        | ✅ (compute.rs)  | SAXPY (WGSL→SPIR-V via naga) |
| OpenGL 4.3+    | `ironaccelerator-opengl`      | ✅ ctx    | ✅ (compute.rs)  | SAXPY |
| WebGPU         | `ironaccelerator-webgpu`      | ✅        | ✅ (compute.rs)  | — |
| TPU (PJRT)     | `ironaccelerator-tpu`         | ✅ env    | ⏳ PJRT client   | — |
| Level Zero     | `ironaccelerator-levelzero`   | ✅        | ✅ (compute.rs)  | — |
| AWS Neuron     | `ironaccelerator-neuron`      | ✅ cores  | ⏳ NEFF load     | — |
| CPU SIMD       | `ironaccelerator-core::simd`  | n/a       | ✅ AVX2 quant    | scalar + quant |

Workspace: `cargo build --workspace --features ironaccelerator/all` clean;
`cargo test --workspace` = **79 passing, 0 failed** as of 2026-04-18
(workspace version `1.1.0`).

---

## Path to 1.1

Blocking items for the next minor. All checkboxes here must flip before
we tag `1.1.0`.

### Cross-backend plumbing

- [x] Umbrella `ironaccelerator::init()` registers every compiled-in
      backend (cuda / rocm / metal / qnn / vulkan / opengl / webgpu /
      tpu / levelzero / neuron).
- [x] Feature flags on the umbrella crate for every backend + `all`.
- [x] `BackendRegistry::describe_all()` — one-shot enumerate across
      every available backend.
- [x] `Strategy` variants for graphics-compute + NPU backends:
      `SpirvCompute`, `GlslCompute`, `Wgsl`, `Pjrt`, `Neuron`,
      `LevelZero`. Every new-backend `plan()` returns one of these
      (no more `Strategy::Custom` stubs).
- [x] Integration test that spins up `Runtime::new()` and asserts at
      least the CPU path returns a plan for a reference GEMM workload.
      (`crates/ironaccelerator/tests/runtime_smoke.rs`)

### Vulkan

- [x] Instance + physical-device enumeration (subgroup, FP16/INT8,
      cooperative-matrix).
- [x] `compute::Context` (logical device + compute queue + command
      pool + memory-type lookup).
- [x] `compute::Buffer` (device-local + host-visible) with
      map/unmap.
- [x] `compute::ComputePipeline` (SPIR-V + descriptor set + storage
      buffers).
- [x] One-shot `Context::dispatch` (submit + wait).
- [x] SAXPY kernel compiled from WGSL at runtime via `naga`, wired
      as `kernels::axpy_f32`. First real Vulkan payload.
- [x] WGSL→SPIR-V fallback via `naga` (`shader::wgsl_to_spirv`) so
      Vulkan and WebGPU share one kernel source.
- [ ] `GemmPlan` with `VK_KHR_cooperative_matrix` on discrete GPUs,
      fall-through tiled GEMM for integrated.

### WebGPU

- [x] Adapter enumeration across Vulkan/Metal/DX12/GLES on native,
      browser adapter on WASM.
- [x] `compute::Context` (adapter pick + device + queue).
- [x] `compute::ComputePipeline::from_wgsl` + `dispatch` helper.
- [x] `bind_device` for WASM pre-selected devices.
- [x] SAXPY WGSL kernel exposed as `webgpu::kernels::axpy_f32`.
- [x] Naive tiled GEMM in WGSL
      (`kernels::GEMM_F32_WGSL` + `kernels::gemm_f32`); shared with
      Vulkan backend (`ironaccelerator_vulkan::kernels::gemm_f32`).
      Subgroup-optimized variant parks until the WebGPU `SUBGROUP`
      feature is stable in browsers.
- [ ] WASM smoke-test harness using `wasm-bindgen-test`.

### OpenGL

- [x] `bind_current_context(loader)` + 4.3 probe.
- [x] `compute::Program` (compile + link a GLSL `#version 430`
      compute shader).
- [x] SSBO helpers (`glBufferStorage` + `glBindBufferBase`).
- [x] `dispatch` that wraps `glDispatchCompute` + `glMemoryBarrier`.
- [x] SAXPY GLSL kernel (`kernels::axpy_f32`) mirroring the Vulkan one.

### TPU (PJRT)

- [x] Plugin loader (`GetPjrtApi` symbol probe).
- [x] Env-driven topology (`TPU_ACCELERATOR_TYPE`, `TPU_NUM_DEVICES`).
- [ ] Real `PJRT_Client_Create` + `PJRT_Client_Devices` walk (replace
      env enumeration once the plugin is reachable on CI).
- [ ] StableHLO builder stub — accept a serialised HLO module and
      call `PJRT_Client_Compile`.
- [ ] `PJRT_LoadedExecutable_Execute` round-trip for a trivial
      add-one program.

### Level Zero

- [x] `ze_loader` dynamic load + `zeInit` + device walk.
- [x] `Context` (`zeContextCreate` + `zeCommandQueueCreate` +
      `zeCommandListCreate`).
- [x] Buffer allocation via `zeMemAllocDevice` / `zeMemAllocShared`
      (`Context::alloc_device`, `Context::alloc_shared`).
- [x] SPIR-V module load (`Context::load_spirv`) + `zeKernelCreate`
      + dispatch via `zeCommandListAppendLaunchKernel`
      (`Module::kernel` + `Context::launch`).
- [ ] NPU (VPU) path: verify the same pipeline works on Meteor Lake
      NPU driver.

### AWS Neuron

- [x] `libnrt` load + `nrt_get_total_nc_count` + generation detection.
- [x] `nrt_load` a NEFF binary into a NeuronCore
      (`runtime::Model::load`).
- [x] `nrt_execute` with tensor I/O (`runtime::Model::execute`,
      `runtime::TensorSet`), surfaced as `Strategy::Neuron`.
- [ ] Trn2 FP8 path documentation.

### QNN

- [x] Provider loader + `Backend`/`Device`/`Context`/`Graph` wrappers.
- [ ] Runtime `Graph::execute` tested end-to-end on a Hexagon HDK box.
- [ ] Serialise + rehydrate a compiled graph through a temp file
      round-trip in CI.

### Quant / SIMD / core

- [x] `QuantScheme` + `CalibStats` + `QuantParams`.
- [x] CPU INT8 per-channel + INT4 per-group reference + roundtrip
      tests.
- [x] AVX2 `quant_i8_row` / `dequant_i8_row` runtime dispatch.
- [x] NEON equivalents (`aarch64` runtime feature dispatch in
      `simd.rs`).
- [x] Asymmetric INT8 per-tensor quant/dequant
      (`quant_i8_per_tensor_asym` / `dequant_i8_per_tensor_asym`).
- [x] FP8 (E4M3 / E5M2) scale-calibration helper
      (`fp8_scale_from_absmax`, `fp8_scale_from_history`) for the CUDA
      + ROCm transformer-engine paths.

### Flash MoE

- [x] `FlashMoePlan` with per-expert matmul loop (CUDA).
- [ ] cuBLASLt **grouped GEMM** — single launch covering every expert;
      bump CUDA minimum to 12.5.
- [ ] BF16 and FP8 I/O variants.
- [x] SwiGLU (up + gate split) reference in
      `ironaccelerator_core::activation::{swiglu, swiglu_interleaved}`.
      CUDA fused variant tracked post-1.1.
- [ ] Device-side dispatch — eliminate the `offsets` D2H sync.

### Docs + release engineering

- [x] `README.md` matrix of backends + minimum vendor-SDK versions.
- [x] `docs/backends/` one file per backend with build-time + runtime
      prerequisites (10 backend files).
- [x] CI matrix: Linux (CUDA / ROCm / Level Zero / Vulkan / Neuron /
      TPU), Windows (CUDA / Vulkan / WebGPU / Level Zero), macOS
      (Metal / WebGPU / OpenGL) — `.github/workflows/ci.yml`
      `feature-matrix` job compiles the umbrella crate under each
      feature combination; no vendor SDK linked, every backend loads
      via `libloading`.
- [ ] Public-API review: no `#[doc(hidden)]` leaks, every `pub`
      item documented. Pre-existing broken intra-doc links in the
      0.1-era CUDA sys / metal / cuda crates must be fixed before we
      can re-enable `RUSTDOCFLAGS=-D warnings` in CI. (Tracked;
      non-blocking for 1.1.)
- [x] `CHANGELOG.md` `[1.1.0]` entry cut; workspace version bumped
      to **1.1.0**. (Git tag is operator-driven — push when ready.)

---

## Post-1.1 parking lot

Known-useful work that isn't blocking the next minor:

- Grouped-GEMM kernel replacing per-expert loop for MoE.
- CoreML / ANE bridge via `objc2-core-ml`.
- MLX-style JIT kernels for Apple Silicon.
- Python bindings (`pyo3`) once 1.1 API is frozen.
- `#[no_std]` feature for embedded NPU targets.
- MIOpen safe wrapper (ROCm convolution + fused ops).
- rocFFT / rocRAND / rocSOLVER / rocSPARSE safe wrappers.
- Composable Kernel template front-end (ROCm CUTLASS analogue).
- Triton-style kernel JIT for cross-backend fused attention.
- FlashAttention-3 BF16 + FP8 variants on Hopper.
- NVLink / xGMI / NVSwitch topology-aware collective planning.
- WASM compute benchmarks vs. native Vulkan + Metal.

---

## Shipped

### 1.1.0 — 2026-04-18

- Vulkan `kernels::axpy_f32` + `kernels::gemm_f32` driven by
  `shader::wgsl_to_spirv` (naga WGSL → SPIR-V).
- WebGPU `kernels::axpy_f32` + naive tiled `kernels::gemm_f32`; shared
  WGSL source with Vulkan.
- OpenGL `kernels::axpy_f32` (GLSL `#version 430`).
- Level Zero compute stack: context + queue + list + device/shared
  USM allocation + SPIR-V module load + kernel create + launch.
- AWS Neuron `runtime::Model::load` / `runtime::Model::execute`
  (NEFF binary → NeuronCore).
- Core `activation::{silu, swiglu, swiglu_interleaved}` CPU refs.
- Core `quant::{Fp8Format, fp8_scale_from_absmax,
  fp8_scale_from_history}` for transformer-engine FP8 calibration.
- Core `quant_i8_per_tensor_asym` / `dequant_i8_per_tensor_asym`.
- NEON (`aarch64`) runtime dispatch for `quant_i8_row` /
  `dequant_i8_row`.
- Umbrella `runtime_smoke` integration test.
- README backend matrix + minimum vendor-SDK versions;
  `docs/backends/` one file per backend.
- CI `feature-matrix` job that builds the umbrella crate per-OS
  against every feasible backend combination.

### 1.0.0 — 2026-04-18

- CUDA / ROCm / Metal / QNN backends enumerate live devices.
- FlashAttention-3 via cuDNN v9 backend-descriptor graph.
- Metal GEMM via `MPSMatrixMultiplication`.
- Version bump + SemVer freeze.
- Flash MoE (`ironaccelerator-cuda::moe`).
- Core SIMD (AVX2 row-wise INT8 quant/dequant).
- Core quantization primitives (INT8 per-channel + per-tensor; INT4
  per-group symmetric with packed-nibble storage).
- Vulkan / OpenGL / WebGPU backends (enumeration + compute scaffold for
  Vulkan and WebGPU).
- Google TPU backend via PJRT plugin loader.
- Intel Level Zero backend (GPU + NPU).
- AWS Neuron backend (Trainium / Inferentia).
- Umbrella `ironaccelerator::init()` registers every compiled-in
  backend; feature-gated `all`.

### 0.2.0 — 2026-04-17

- ROCm sys + safe wrappers; hipBLASLt FP8 for gfx942.
- CUDA loader null-termination + Windows symbol fallback fixes.
- Criterion benches (wrapped vs raw driver paths).
- AGPL-3.0 + commercial dual license.

### 0.1.0 — 2026-04-16

- Workspace skeleton, `ironaccelerator-core` trait surface.
- CUDA sys + safe wrappers (driver / runtime / NVRTC / cuBLAS(Lt) /
  cuDNN / cuRAND / cuSPARSE / cuSOLVER / cuFFT / cuTENSOR / NCCL).
- Ontology + strategy + heuristic planner.
