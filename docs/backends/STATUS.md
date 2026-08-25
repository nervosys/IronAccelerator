# Backend hardware support — honest status

This doc tracks what each backend crate actually supplies, what's tested
against a live device, and what's still scaffold, as of **2.2.0**. The CUDA
crate is the reference target; "parity with CUDA" below means matching the same
driver-substrate surface shape (Device / Stream / Event / DeviceBuf / Module /
Function / kernel compile / memcpy / library handles).

Two distinct surfaces run through these crates, and a backend can supply one
without the other:

- **Full driver substrate** — the CUDA-shaped stack (Device/Stream/Event/
  DeviceBuf/Module/Function + vendor-library handles + runtime kernel compile).
  Only CUDA is complete here; ROCm is the closest follower.
- **Unified `ComputeDevice` trait** (`ironaccelerator-core`) — a
  least-common-denominator *compute-submission* surface (upload / pipeline /
  dispatch / download) implemented by the five backends that own a device
  handle and consume caller-supplied bytecode. See
  [Cross-vendor compute](#cross-vendor-compute) below.

| Backend            | Driver wrappers        | `ComputeDevice` | Vendor libs            | Runtime kernel compile      | Live-GPU tests                | cudarc-shaped compat | MemPool equivalent |
| ------------------ | ---------------------- | --------------- | ---------------------- | --------------------------- | ----------------------------- | -------------------- | ------------------ |
| **CUDA**           | ✅ full                | n/a (own stack) | ✅ 12 libs             | ✅ NVRTC + disk cache       | ✅ 52 test fns                | ✅ `cudarc_compat`   | ✅ `MemPool`       |
| **D3D12**          | ✅ enumerate + compute | ✅              | n/a                    | ❌ bring your own DXIL      | ✅ dispatch on 3 adapters      | ❌                   | ❌                 |
| **Vulkan**         | ✅ enumerate + compute | ✅              | n/a                    | ❌ bring your own SPIR-V    | ✅ dispatch on 3 devices       | ❌                   | ❌                 |
| **OpenGL**         | ✅ enumerate + compute | ✅              | n/a                    | ✅ GLSL (driver compiles)   | ✅ dispatch on WGL 4.3         | ❌                   | ❌                 |
| **Metal**          | ✅ enumerate + compute | ✅              | ⏳ (MPS dropped as workload-level) | ❌ bring your own `.metallib` | ⏳ compile-checked; needs macOS | ❌               | ❌                 |
| **Level Zero**     | ✅ enumerate + compute | ✅              | n/a                    | ❌ bring your own SPIR-V    | ⏳ builds; needs Intel GPU     | ❌                   | ❌                 |
| **ROCm**           | ✅ HIP full            | ❌              | ⏳ hipBLASLt scaffold  | ⏳ HIPRTC pending           | ❌ no AMD GPU on CI host       | ❌                   | ❌                 |
| **Qualcomm (QNN)** | ⚠️ AOT graph substrate | ❌ (AOT model)  | ⏳ QNN SDK FFI partial | n/a (QNN graphs are AOT)    | ❌ no SDK/device here          | ❌                   | ❌                 |
| **WebGPU**         | ✅ host-bound adapter  | ❌ (async, host-owned) | n/a             | n/a (host owns device)      | ⏳ needs browser harness       | ❌                   | ❌                 |
| **TPU (PJRT)**     | ⚠️ env/plugin probe    | ❌              | n/a                    | n/a (PJRT plugin is AOT)    | ❌ needs TPU VM                | ❌                   | ❌                 |
| **AWS Neuron**     | ⚠️ `libnrt` core probe | ❌              | n/a                    | n/a (NEFF is AOT)           | ❌ needs trn/inf instance      | ❌                   | ❌                 |

✅ = shipped and exercised against a real device (or the closest equivalent)
⚠️ = scaffold compiles and registers; does device enumeration / probe only
⏳ = present in code, work pending, or can't run on this host
❌ = not present, or not meaningful for this backend's model
n/a = not meaningful for this backend

## Cross-vendor compute

The one addition since the last revision of this doc. `ComputeDevice`
(`ironaccelerator-core/src/compute.rs`) is a single compute-submission trait —
`device_buffer` / `upload` / `pipeline` / `dispatch` / `download` — implemented
by **five** backends: Vulkan, D3D12, OpenGL, Metal, and Level Zero. A routine
written against `C: ComputeDevice` runs unchanged on any of them; only the
shader bytecode differs (SPIR-V / DXIL / GLSL / `.metallib`), because there is
no translation layer — each backend ingests exactly what its driver consumes.

The trait uses associated `Buffer` / `Pipeline` / `Error` types, so it is
zero-cost (no boxing, no vtable) and `no_std`-clean; it is deliberately not
`dyn`-safe. It carries only threadgroup *counts*, so the Metal and Level Zero
impls — which set group size at dispatch rather than in the shader — assume a
1-D group of 64 and expose a native call taking an explicit size for other
geometries.

Verified live on Vulkan (2× RTX 3090 Ti + AMD iGPU), D3D12 (same 3 adapters +
WARP), and OpenGL (WGL 4.3 context) — one generic doubling routine, byte-checked
readback. Metal and Level Zero implement the trait and compile-check (Metal
cross-checked for `aarch64-apple-darwin`, Level Zero built natively) but are not
run here: this workspace has no Apple or Intel-GPU host.

**Why WebGPU sits it out:** its `GPUDevice` is owned by the host page and driven
asynchronously from JS (readback is `mapAsync` → a Promise). There is no
synchronous device handle to hang the impl on, and wrapping it would re-introduce
the binding-crate dependency that backend exists to avoid.

## Per-backend detail

### CUDA — reference

The mature backend, and the reason to use the project today. Detailed coverage
in
[`crates/ironaccelerator-cuda/src/cudarc_compat.rs`](../../crates/ironaccelerator-cuda/src/cudarc_compat.rs)
and the [release blog post](../release-blog-post.md).

Drop-in for cudarc 0.19, faster on the host-side hot path (H→D CI-confirmed at
every size, up to ~1.29×; ~2× on plain `DeviceBuf::alloc`; ~70× via the opt-in
`MemPool` recycling allocator). Twelve vendor libraries plumbed (cuBLAS,
cuBLASLt, cuDNN, cuFFT, cuRAND, cuSOLVER, cuSPARSE, cuTENSOR, NCCL, NVRTC, CUPTI,
NVTX), NVRTC runtime compile with in-memory + on-disk PTX cache, stream-capture
graphs, VMM / green contexts / multicast in `advanced`. 52 test functions across
unit and live-GPU (`gpu_smoke`) suites.

### D3D12

Status: **enumeration, capability probing, and compute dispatch — all verified
on real hardware.** Implements `ComputeDevice`.

- `crates/ironaccelerator-dx12/src/drv.rs` — hand-written COM vtables for
  `IDXGIFactory1` / `IDXGIAdapter1` / `ID3D12Device`, `libloading` for
  `d3d12.dll` + `dxgi.dll`, adapter walk, `CheckFeatureSupport` probes, and
  `open()` returning an owned `ID3D12Device`.
- `crates/ironaccelerator-dx12/src/compute.rs` — COMPUTE queue, allocator,
  command list, fence, committed buffers in all three heaps, staged
  upload/download with barriers, root signatures, compute pipelines from DXIL,
  and dispatch.
- Verified against 2× RTX 3090 Ti + an AMD integrated part: 3 adapters, matching
  `Win32_VideoController` exactly in count and order, dispatch correctly doubling
  1024 floats on each.

**Missing for full parity with CUDA:** descriptor heaps (textures and samplers),
a `MemPool` equivalent, a cudarc-shaped surface, and async submission (`Context`
blocks on a fence per submit).

### Vulkan

Status: **driver substrate + compute dispatch, cross-vendor, live-tested.**
Implements `ComputeDevice`.

- `crates/ironaccelerator-vulkan/src/drv.rs` — physical-device enumeration,
  queue-family probe, compute-capability surfacing (subgroup size, FP16/INT8).
- `crates/ironaccelerator-vulkan/src/compute.rs` — `Context`, `Buffer`,
  `ComputePipeline`: load a SPIR-V module and dispatch a compute shader.
- `tests/dispatch.rs` verifies the generic `ComputeDevice` round-trip on the
  host's real Vulkan ICDs (3 devices).

**Removed for scope (2.0.0):** `src/kernels.rs` (shipped SAXPY/GEMM source) and
`src/shader.rs` (WGSL → SPIR-V via `naga`). Shader source and translation are
toolchain concerns, not driver ones; Vulkan ingests SPIR-V, so the backend takes
SPIR-V.

**Missing for full parity with CUDA:** a thinner cudarc-style facade (Vulkan's
compute API is more ceremonial than CUDA's), and a `MemPool` equivalent.

### OpenGL

Status: **legacy/embedded compute fallback, live-tested.** Implements
`ComputeDevice`.

- `crates/ironaccelerator-opengl/src/{drv,compute}.rs` — GL 4.3+ compute-shader
  context, buffers, and dispatch. GLSL source is handed straight to the driver,
  so this is the one `ComputeDevice` backend with a built-in runtime compile.
- `tests/live_compute.rs` runs the generic round-trip on a WGL 4.3 context.
- Lost its `kernels.rs` in the same 2.0.0 scope-cleanup pass as Vulkan.

### Metal

Status: **compute dispatch implemented; can't run live here.** Implements
`ComputeDevice`. (Previously listed as "scaffold only" — that is stale.)

- `crates/ironaccelerator-metal/src/{backend,drv,compute}.rs` — device
  enumeration and a `Context` implementing the trait: `upload` / `download` as a
  shared-storage `memcpy`, `pipeline` from a compiled `.metallib`, `dispatch`
  via a command buffer. Bindings behind `cfg(target_vendor = "apple")` on
  `objc2` + `metal-rs`.
- `dispatch_sized` takes an explicit threadgroup size for non-64-wide geometries.
- `tests/dispatch.rs` compiles an MSL kernel with `xcrun` and runs the trait
  round-trip — gated to `target_vendor = "apple"`, so it reports zero tests on
  this Windows CI and runs only on an actual Mac.

**Removed for scope:** the MPS-backed GEMM (workload-level, belongs above the
driver line).

**Missing for full parity with CUDA:** MTLHeap-based `MemPool`, a cudarc-shaped
surface, and live-GPU validation (needs a Mac).

### Level Zero

Status: **compute dispatch implemented; can't run live here.** Implements
`ComputeDevice`. (Newly added since the last revision of this doc.)

- `crates/ironaccelerator-levelzero/src/{drv,compute}.rs` — `ze_loader` probe,
  device enumeration, and a `Context` implementing the trait. Consumes
  OpenCL/SYCL-flavored (`Kernel`-model) SPIR-V — pointer args, not Vulkan's
  `GLCompute` model — so the two are not interchangeable.
- `Kernel::set_group_size` + `Context::launch` take an explicit group size.
- `tests/dispatch.rs` runs the round-trip when a device *and* an
  `IA_LEVELZERO_SHADER` SPIR-V binary are present; skips cleanly otherwise
  (every host without Intel compute, this CI included).

**Missing for full parity with CUDA:** tighter capability probe (COMPUTE
queue-group query), a `MemPool` equivalent, and live-GPU validation (needs an
Intel Arc/Flex/PVC GPU or NPU).

### ROCm

Status: **driver substrate is real; can't run live here.** Does *not* implement
`ComputeDevice` — it targets the full CUDA-shaped stack instead.

- `crates/ironaccelerator-rocm/src/drv.rs` — `Device`, `Stream`, `Event`,
  `DeviceBuf`, `Module`, `Function`, `launch_raw`. Same fast-path posture as
  CUDA: `AtomicPtr<HipFns>` hot-path cache, per-handle cached `&'static HipFns`,
  cold-error path, `#[inline]` on every wrapped op.
- `crates/ironaccelerator-rocm/src/blas.rs` — hipBLASLt handle plumbing scaffold.
- `crates/ironaccelerator-rocm-sys` — clean-room HIP FFI + dynamic loader
  (`hip`, `hipblas`, `hipblaslt`, `rccl`), same `AtomicPtr<HipFns>` hot-path
  cache.

**Missing for full parity with CUDA:**
- HIPRTC binding (runtime kernel compile, equivalent to NVRTC).
- `cudarc_compat`-shaped surface (`HipDevice`/`HipSlice`/…).
- `MemPool` recycling allocator (same three-tier design as CUDA).
- Live-GPU tests + bench (0 tests today). Workspace is Windows + NVIDIA; the AMD
  path needs dual-boot or a remote box.

This is the backend closest to a second production target: the driver substrate
is present, and the gap is bounded — HIPRTC, a compat surface, a MemPool port,
and hardware in CI.

### Qualcomm (QNN)

Status: **AOT graph substrate scaffold; no live device here.** Does not
implement `ComputeDevice` (QNN is an ahead-of-time graph model, not a
shader-dispatch one).

- `crates/ironaccelerator-qnn/src/drv.rs` — `Backend`, `Device`, `Context`,
  `Graph` with `new` / `from_binary` / `to_binary` / `finalize` and a
  `libloading` probe for `QnnHtp.dll` / `libQnnHtp.so`. More than enumeration:
  the binary-context cache path (serialize/deserialize a finalized context) is
  the shape QNN consumers actually need.
- `crates/ironaccelerator-qnn-sys` — partial FFI (`qnn`, `loader`); the full
  `QnnApi`/`QnnContext`/`QnnGraph`/`QnnHtp` surface and tensor/graph builders
  are still pending.

Per the [no-Huawei / Chinese-vendor rule](../../README.md) and the scope rule,
QNN stays at the driver-substrate layer — no graph optimizer, no quantization
recipes.

**Missing:** complete QNN SDK FFI, tensor/graph builders, HTP execution path,
live-NPU tests (needs Qualcomm SDK 2.22+ and a Snapdragon device or emulator).

### WebGPU

Status: **browser/WASM host-bound adapter only.** Intentionally not on the
`ComputeDevice` trait (see [Cross-vendor compute](#cross-vendor-compute)).

Host-bound, zero native dependencies — D3D12 now covers the Windows gap the old
`wgpu`-based native path had been reaching for. `tests/wasm_binding.rs` needs a
browser harness to run.

### TPU (PJRT) / AWS Neuron

Both stay at the probe/enumerate layer until there's a concrete consumer to
drive the work — and both target AOT compiled artifacts (PJRT plugin, NEFF), so
neither fits the `ComputeDevice` shader-dispatch model.

- **TPU** — `crates/ironaccelerator-tpu/src/drv.rs`: probes the Cloud TPU VM /
  GKE PJRT plugin paths and reads topology from `TPU_ACCELERATOR_TYPE` /
  `TPU_NUM_DEVICES` / `TPU_CHIPS_PER_HOST`. Needs a TPU VM to go further.
- **Neuron** — `crates/ironaccelerator-neuron/src/drv.rs`: `libnrt` loader
  (`nrt_init`, core count, version, generation detection) plus a `runtime::Model`
  stub. Needs a trn/inf instance.

## What "full hardware support" honestly requires

For a non-CUDA backend to reach CUDA-equivalent parity:

1. Runtime kernel-compile primitive where applicable: HIPRTC for ROCm;
   `metal::Library::newLibraryWithSource` for Metal; AOT-only for QNN / TPU /
   Neuron. Vulkan, D3D12, and Level Zero take pre-compiled bytecode by design —
   translating shader source is a toolchain job, not a driver one; OpenGL is the
   exception, since GL drivers compile GLSL themselves.
2. cudarc-shaped compatibility surface — same `Device`/`Slice`/`Stream` shapes
   and method names, so downstream code can portably target the right backend at
   compile time.
3. `MemPool` equivalent — the same three-tier design that gave CUDA the ~70×
   alloc/free win.
4. Live-GPU tests against a real device.
5. Reference benchmark vs the closest established Rust wrapper (hipBLAS-rs for
   ROCm, metal-rs for Metal, ash/wgpu for Vulkan).

That is months of dedicated work per backend, and most of it needs the target
hardware in the loop. The Windows + NVIDIA box that drove the CUDA optimization
sprint can't validate ROCm/Metal/QNN/Level Zero. ROCm and Vulkan are the most
tractable here (ROCm via dual-boot or a remote AMD box; Vulkan via the NVIDIA
Vulkan ICD already present, and already dispatching). Metal, Level Zero, and QNN
fundamentally need their target hardware.

## Summary

Use CUDA today — it's the production backend. Five backends (Vulkan, D3D12,
OpenGL, Metal, Level Zero) now share the unified `ComputeDevice` compute-
submission trait, three of them (Vulkan, D3D12, OpenGL) verified on real
hardware. ROCm has a real driver substrate but no runtime compile, compat
surface, or live tests. QNN is an AOT-graph scaffold; WebGPU is browser-bound;
TPU and Neuron are probe-only. Every backend loads its vendor runtime via
`libloading` at first use, so the workspace builds on any host and surfaces
missing runtimes as typed `NotAvailable` errors rather than link failures.
