# IronAccelerator — Roadmap

Live progress tracker. Each section is a milestone; each bullet is a
deliverable with a status box. Move items to **Shipped** with the version
they land in.

Last updated: **2026-10-13** — workspace at **2.4.0** (FPGA backend, capability
ontology, ROCm depth, the CUDA/ROCm/Intel/FPGA hardware lanes). See the
CHANGELOG.

> **Scope.** IronAccelerator is a driver substrate. Kernels, planners,
> workload/strategy types, quantization schemes, CPU reference ops, and the
> accelerator ontology all live in [IronWorks](https://github.com/nervosys/ironworks),
> the inference engine that consumes this crate. Roadmap items below are
> driver-level only.

---

## Status snapshot

Per-backend status lives in one place — [`docs/backends/STATUS.md`](docs/backends/STATUS.md)
— which tracks, for every backend, what the crate actually supplies, what has
run against a live device, and what is still scaffold. It is kept current with
the code; this file deliberately no longer carries a second copy of that table,
because the duplicate is what drifted.

The shape in one paragraph: **CUDA** is the complete driver substrate and the
reference target. **ROCm** is the closest follower (HIPRTC + `MemPool` landed;
device paths await an AMD runner). **Vulkan, D3D12, OpenGL, Metal, and Level
Zero** implement the unified `ComputeDevice` submission trait. **QNN, TPU,
Neuron, and FPGA** are probe/enumerate only — for FPGA that is the endpoint, not
a gap, since bitstreams are AOT.

Workspace: `cargo build --workspace --features ironaccelerator/all` clean;
`cargo test --workspace --all-features` green (114 passed, 7 `#[ignore]`d for
GPU, 0 failed) and `cargo clippy --workspace --all-features --all-targets
-- -D warnings` clean, as of 2026-09-09.

---

## Open driver-level work

What remains at the driver line, by backend. These were originally scoped as
"Path to 1.1" blockers; 1.1 shipped 2026-04-18 and the workspace is now at
2.2.0, so the unticked boxes below are simply the work still outstanding — no
longer a release gate.

### Cross-backend plumbing

- [x] Umbrella `ironaccelerator::init()` registers every compiled-in
      backend (cuda / rocm / metal / qnn / vulkan / opengl / webgpu /
      tpu / levelzero / neuron).
- [x] Feature flags on the umbrella crate for every backend + `all`.
- [x] `BackendRegistry::describe_all()` — one-shot enumerate across
      every available backend.
- [x] Integration test that spins up `Runtime::new()` and surveys devices
      + capability bits across every registered backend.
      (`crates/ironaccelerator/tests/runtime_smoke.rs`)
- [x] `Backend` trait reduced to discovery only — `kind`, `is_available`,
      `enumerate`, `capabilities`. Strategy selection moved to IronWorks.

### CUDA

- [x] Complete driver substrate — the reference target for every other
      backend. Device / Stream / Event / DeviceBuf / Module / Function,
      NVRTC compile + disk cache, 12 vendor libraries, `MemPool`,
      `cudarc_compat`.
- [x] Race-free NVRTC module cache and vendor-handle caches
      (double-checked insert; both were check-then-act).
- [x] Live hardware lane in CI — `hardware.yml` (lane: `cuda`) on a
      `[self-hosted, linux, cuda]` runner. Fails if `nvidia-smi` reports
      no GPU, so it cannot pass green having skipped, and runs the
      `#[ignore]`-gated `cudarc_compat` round-trips that execute nowhere
      else. **Needs an NVIDIA runner registered to actually run.**
- [x] CI's docs job enforces `RUSTDOCFLAGS=-D warnings` (see *Docs + release
      engineering*).

### ROCm

The closest follower to CUDA, and the largest open gap in the workspace.

- [x] HIP driver wrappers (Device / Stream / Event / DeviceBuf / Module).
- [x] `iron_rocm_sys::hiprtc` — HIPRTC FFI, the NVRTC analogue.
- [x] `kernel::get_or_compile` + in-memory module cache, race-free from
      the start.
- [x] `pool::MemPool` — recycling allocator over
      `hipMallocAsync`/`FreeAsync`, shared-freelist tier; bucket math
      unit-tested.
- [x] `rocm_smoke` end-to-end test, double-gated on a HIP device and
      `IRON_RUN_GPU_TESTS`, plus the `rocm` lane in `hardware.yml`.
- [ ] **Run it.** Every ROCm device path is structurally mirrored from
      the validated CUDA design but has never executed on an AMD GPU.
      Registering an AMD runner is the whole remaining step, and it is
      what promotes the `†` rows in `STATUS.md` to validated.
- [ ] Lock-free per-thread front cache for `MemPool` (CUDA's ~70× tier).
- [ ] `hip_compat` surface mirroring `cudarc_compat`.
- [ ] hipBLASLt beyond the current scaffold; MIOpen handle plumbing.

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

> The SAXPY/GEMM WGSL kernels and the `naga`-backed `shader::wgsl_to_spirv`
> that compiled them were removed — kernels in 1.2.0, the shader front-end
> in 2.0.0. Both sat above the driver line. `ComputePipeline` takes SPIR-V
> you supply.

### Direct3D 12

- [x] `libloading` probe of `d3d12.dll` + `dxgi.dll`, hand-written COM
      vtables, DXGI adapter walk with software adapters filtered out.
- [x] `CheckFeatureSupport` probes: feature level, FP64, native 16-bit
      ops, UMA, wave ops + lane counts.
- [x] `drv::open(ordinal)` → owned `ID3D12Device`, released on drop.
- [x] Verified on real hardware (2× RTX 3090 Ti + AMD iGPU), matching
      `Win32_VideoController` in count and order.
- [x] Command queue / allocator / list wrappers + fence sync.
- [x] Committed resources in the DEFAULT / UPLOAD / READBACK heaps, with
      staged `upload` / `download` and resource barriers.
- [x] Root signature (root UAVs) + compute pipeline from DXIL + dispatch.
- [x] Dispatch test against live adapters — verified doubling 1024 floats
      on 2× RTX 3090 Ti and an AMD iGPU.
- [ ] Descriptor heaps, for textures and samplers. Needs the
      `GetCPUDescriptorHandleForHeapStart` by-value-return ABI handled.
- [ ] `MemPool` equivalent; `Context` currently blocks per submission.

### WebGPU (browser only)

- [x] Host-bound `AdapterInfo` model — the host awaits `requestAdapter()`
      / `requestDevice()` and registers the result; the `GPUDevice` stays
      with the host.
- [x] Zero dependencies; builds for `wasm32-unknown-unknown` (it could
      not before 2.0.0) and is covered by a CI job.
- [x] Fallback (software) adapters recorded but never offered.
- [ ] WASM smoke-test harness using `wasm-bindgen-test`.

> The native `wgpu` path was removed in 2.0.0. On native it reached
> nothing Vulkan / Metal / D3D12 / OpenGL do not, at the cost of 98
> transitive dependencies and a layer of indirection above the driver.

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
      `runtime::TensorSet`).
- [ ] Trn2 FP8 path documentation.

### QNN

- [x] Provider loader + `Backend`/`Device`/`Context`/`Graph` wrappers.
- [ ] Runtime `Graph::execute` tested end-to-end on a Hexagon HDK box.
- [ ] Serialise + rehydrate a compiled graph through a temp file
      round-trip in CI.

### FPGA (XRT)

- [x] `libxrt_core` `libloading` probe + `xclProbe` enumeration, one
      descriptor per device; wired into the facade and the registry.
- [x] `fpga` lane in `hardware.yml` for a `[self-hosted, linux, fpga]`
      runner.
- [ ] Run the lane against a real Alveo / Versal card.

> Probe/enumerate is the **endpoint** here, not a gap. FPGA kernels are
> pre-synthesised bitstreams, so there is no runtime-compile path and no
> `ComputeDevice` impl to write; capability is bitstream-defined rather
> than silicon-defined and is reported empty.

### Moved out of scope

Shipped in 1.0/1.1, removed from IronAccelerator in the driver-substrate
cut. Tracked in IronWorks from here on — see the CHANGELOG's
`Removed (breaking)` entry for the full symbol list.

- Quantization schemes + calibration (`QuantScheme`, `CalibStats`,
  `QuantParams`, FP8 scale calibration).
- CPU reference / SIMD oracles (`core::simd`, `core::activation`).
- Workload + strategy descriptors and the heuristic planner.
- The accelerator ontology crate (`ironaccelerator-ontology`).
- Flash MoE, grouped GEMM, and every other kernel-level deliverable.
- WGSL → SPIR-V translation (`vulkan::shader`) and the `naga` dependency.
- The native `wgpu` WebGPU path; D3D12 now covers the Windows gap it
  was reaching for, directly.

### Docs + release engineering

- [x] `README.md` matrix of backends + minimum vendor-SDK versions.
- [x] `docs/backends/` one file per backend with build-time + runtime
      prerequisites (11 backend files).
- [x] CI matrix: Linux (CUDA / ROCm / Level Zero / Vulkan / Neuron /
      TPU), Windows (CUDA / Vulkan / OpenGL / D3D12 / Level Zero), macOS
      (Metal / WebGPU / OpenGL) — `.github/workflows/ci.yml`
      `feature-matrix` job compiles the umbrella crate under each
      feature combination; no vendor SDK linked, every backend loads
      via `libloading`.
- [x] `no-std` job building `ironaccelerator-core` without default
      features, and a `wasm32-unknown-unknown` job for the WebGPU
      backend. Both configurations had rotted undetected before 2.0.0
      because nothing in CI built them.
- [x] Public-API review: no `#[doc(hidden)]` anywhere; every `pub` item in
      the safe crates documented and `#![warn(missing_docs)]` enforced via
      CI clippy. `-sys` crates mirror vendor headers 1:1 and defer to them.
- [x] `docs/backends/CI-HARDWARE.md` — the three tiers of hardware path
      (free on hosted / self-hosted / no path GitHub can offer) and how
      to register a runner for each lane.
- [x] `docs/backends/STATUS.md` is the single source of per-backend
      status; this file points at it rather than duplicating it.
- [x] `CHANGELOG.md` `[1.1.0]` entry cut; workspace version bumped
      to **1.1.0**. (Git tag is operator-driven — push when ready.)
- [x] Cut **2.3.0** from the `[Unreleased]` entry — FPGA backend and
      capability ontology are additive. (crates.io publish is operator-driven.)

---

## Post-1.1 parking lot

Known-useful driver-level work that isn't blocking the next minor:

- CoreML / ANE bridge via `objc2-core-ml`.
- Python bindings (`pyo3`) once the API is frozen.
- `#[no_std]` feature for embedded NPU targets.
- MIOpen handle plumbing (ROCm convolution + fused ops).
- rocFFT / rocRAND / rocSOLVER / rocSPARSE safe wrappers.
- HIPRTC *disk* cache, matching the CUDA NVRTC shape (the runtime
  compile and in-memory cache shipped in the unreleased work).
- Per-stream custom `cuMemPool` wrappers (default-pool retention shipped).
- NVLink / xGMI / NVSwitch topology *reporting* (planning is a consumer
  concern).
- WASM compute benchmarks vs. native Vulkan + Metal.

Kernel- and planner-level ideas that used to live here (grouped GEMM,
FlashAttention variants, Triton-style JIT, MLX kernels) belong to IronWorks.

---

## Shipped

### 2.4.0 — 2026-10-13

- Public-API review: every safe-crate `pub` item documented,
  `missing_docs` enforced. cuDNN `AttrType` tags corrected; `graph::Capture`
  honours `CaptureMode`.

### 2.3.0 — 2026-10-13

- FPGA backend (XRT probe/enumerate), capability/cost ontology +
  `.well-known/ironstack.json`, ROCm HIPRTC + module cache + `MemPool`
  shared tier, self-hosted CUDA/ROCm/Intel/FPGA hardware lanes, Metal live
  on hosted CI, race-free CUDA caches. Full list in the CHANGELOG.

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
