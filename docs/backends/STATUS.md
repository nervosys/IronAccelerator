# Backend hardware support — honest status

This doc tracks what each backend crate actually supplies, what's tested
against a live device, and what's still scaffold. The CUDA crate is the
reference target; "parity with CUDA" below means matching the same
driver-substrate surface shape (Device / Stream / Event / DeviceBuf /
Module / Function / kernel compile / memcpy / library handles).

| Backend            | Driver wrappers | Vendor libs | NVRTC equivalent | Live-GPU tests | cudarc-shaped compat | MemPool equivalent |
| ------------------ | --------------- | ----------- | ---------------- | -------------- | -------------------- | ------------------ |
| **CUDA**           | ✅ full         | ✅ 11 libs  | ✅ NVRTC + disk cache | ✅ 45 tests | ✅ `cudarc_compat`   | ✅ `MemPool`        |
| **ROCm**           | ✅ HIP full     | ✅ hipBLASLt scaffold | ⏳ hiprtc binding pending | ❌ no GPU here | ❌                  | ❌                  |
| **Metal**          | ⚠️ scaffold     | ⏳ MPS scaffold | n/a (Metal Shading Language is offline) | ❌ no macOS here | ❌              | ❌                  |
| **Vulkan**         | ✅ enumerate+compute | n/a   | ✅ WGSL → SPIR-V via naga | ⏳ device probe only | ❌            | ❌                  |
| **Qualcomm (QNN)** | ⚠️ scaffold     | ⏳ QNN SDK FFI pending | n/a (QNN graphs are AOT) | ❌ no SDK here | ❌              | ❌                  |

✅ = shipped and exercised
⚠️ = scaffold compiles, does device enumeration only
⏳ = noted in code, work pending
❌ = not present
n/a = not meaningful for this backend

## Per-backend detail

### CUDA — reference

The mature backend. Detailed coverage in
[`crates/ironaccelerator-cuda/src/cudarc_compat.rs`](../../crates/ironaccelerator-cuda/src/cudarc_compat.rs)
and the [release blog post](../release-blog-post.md).

Drop-in for cudarc 0.19, ~2× faster on plain `DeviceBuf::alloc`,
~75× faster via the opt-in `MemPool` recycling allocator.

### ROCm

Status: **driver substrate is real; can't run live here.**

- `crates/ironaccelerator-rocm/src/drv.rs` — `Device`, `Stream`, `Event`,
  `DeviceBuf`, `Module`, `Function`, `launch_raw`. Same fast-path posture
  as CUDA: `AtomicPtr<HipFns>` hot-path cache, per-handle cached
  `&'static HipFns`, cold-error path, `#[inline]` on every wrapped op,
  killed eager `String` allocation in `DeviceBuf::alloc`.
- `crates/ironaccelerator-rocm/src/blas.rs` — hipBLASLt handle plumbing
  scaffold.
- `crates/ironaccelerator-rocm-sys` — clean-room HIP FFI + dynamic loader
  with the same `AtomicPtr<HipFns>` hot-path cache.

**Missing for full parity with CUDA:**
- HIPRTC binding (runtime kernel compile, equivalent to NVRTC).
- `cudarc_compat`-shaped surface (`HipDevice`/`HipSlice`/…).
- `MemPool` recycling allocator (same three-tier design as CUDA).
- Live-GPU tests + bench. Workspace is Windows + NVIDIA; can't exercise
  the AMD path without dual-boot or a remote box.

### Metal

Status: **scaffold only.**

- `crates/ironaccelerator-metal/src/{backend,blas,drv}.rs` — backend trait
  is registered, device enumeration stub returns empty list off macOS.
- `crates/ironaccelerator-metal/src/lib.rs` notes objc2 + metal-rs
  bindings gated behind `cfg(target_vendor = "apple")` as the next step.

**Missing for full parity with CUDA:**
- Metal command-queue / command-buffer wrappers.
- MTLBuffer wrappers + safe memcpy primitives.
- MPS handle plumbing for matmul / convolution.
- Live-GPU tests. Workspace is Windows; can't exercise the Metal path
  without a Mac.

### Vulkan

Status: **driver substrate present; cross-vendor.**

- `crates/ironaccelerator-vulkan/src/drv.rs` — physical-device enumeration,
  queue-family probe, compute-capability surfacing (subgroup size,
  FP16/INT8 support).
- `crates/ironaccelerator-vulkan/src/compute.rs` — `Context`, `Buffer`,
  `ComputePipeline` — the minimum to load a SPIR-V module and dispatch a
  compute shader.
- `crates/ironaccelerator-vulkan/src/shader.rs` — runtime WGSL → SPIR-V
  via `naga 22`. This is the kernel-compile primitive (driver-substrate,
  not a kernel itself).

**Recently removed:** `src/kernels.rs` shipped SAXPY/GEMM kernel source.
That violates the "no kernels in backend crates" scope rule we apply
across the workspace. Removed.

**Missing for full parity with CUDA:**
- `cudarc`-style ergonomic wrappers (Vulkan's compute API is more
  ceremonial than CUDA's — needs a thinner facade).
- Memory-pool equivalent of `MemPool`.
- Live-GPU integration test using the host NVIDIA Vulkan ICD.

### Qualcomm (QNN)

Status: **scaffold only.**

- `crates/ironaccelerator-qnn/src/{backend,drv}.rs` — backend trait
  registered, libloading probe for `QnnHtp.dll` / `libQnnHtp.so`.
- `crates/ironaccelerator-qnn-sys` — placeholder FFI module structure.

**Missing for full parity with CUDA:**
- `QnnApi.h` / `QnnContext.h` / `QnnGraph.h` / `QnnHtp.h` FFI surface.
- Tensor + graph builders.
- HTP execution path.
- Live-NPU tests. Requires Qualcomm SDK 2.22+ and a Snapdragon device or
  emulator.

Per the user's [no-Huawei feedback](../../README.md) and the scope rule,
QNN's eventual implementation must also stay at the driver-substrate
layer (no graph optimizer, no quantization recipes — those layer on top).

### WebGPU / OpenGL / TPU / Level Zero / Neuron

These are smaller cross-vendor or niche scaffolds. WebGPU + OpenGL both
just lost their `kernels.rs` files in the same scope-cleanup pass that
trimmed Vulkan. TPU / Level Zero / Neuron stay at the probe/enumerate
layer until there's a concrete consumer to drive the work.

## What "full hardware support" honestly requires

For each non-CUDA backend to reach CUDA-equivalent parity:

1. Runtime kernel compile primitive (NVRTC analogue, where applicable):
   HIPRTC for ROCm, `metal::Library::newLibraryWithSource_options_error`
   for Metal, naga (already done) for Vulkan, AOT-only for QNN.
2. cudarc-shaped compatibility surface — same `Device`/`Slice`/`Stream`
   shapes, same method names, so downstream code can portably target
   the right backend at compile time.
3. `MemPool` equivalent for each — same three-tier design that gave CUDA
   the ~75× alloc/free win.
4. Live-GPU tests against a real device.
5. Reference benchmark vs the closest established Rust wrapper for that
   backend (hipBLAS-rs for ROCm, metal-rs for Metal, ash/wgpu for Vulkan,
   no clear analogue for QNN).

That is months of dedicated work per backend, and most of it needs the
target hardware in the loop. The Windows + NVIDIA box that drove the
CUDA optimization sprint can't validate any of ROCm/Metal/QNN. ROCm and
Vulkan are tractable on this host (ROCm via dual-boot or remote AMD box;
Vulkan via the NVIDIA Vulkan ICD already present). Metal and QNN
fundamentally need their target hardware.

## What's been done in this pass

- Removed `kernels.rs` from `ironaccelerator-vulkan`, `-webgpu`, `-opengl`
  to enforce the same scope rule the CUDA crate already follows (no
  kernel implementations in backend crates).
- Wrote this status doc so callers can tell at a glance what's actually
  there and what's still scaffold.

The CUDA crate is the part to use today. The others compile clean,
register as backends, and enumerate devices where the SDK is present —
but the per-backend hot path is not yet at CUDA's level. The honest
"full hardware support" milestone needs target hardware in CI and
weeks-per-backend of focused work.
