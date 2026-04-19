# Vulkan backend

Crate: `ironaccelerator-vulkan`. Uses `ash 0.38` with the `loaded`
feature — no static link.

## Build time

Pulls in `naga 22` for WGSL → SPIR-V.

## Runtime

- Vulkan **1.3** ICD on the host (`vulkan-1.dll` / `libvulkan.so.1`).
- Any vendor: NVIDIA, AMD, Intel, Mesa, MoltenVK.
- Subgroup / FP16 / INT8 / cooperative-matrix features are probed and
  surfaced on `DeviceDescriptor`.

## Capabilities

- Instance + physical-device enumeration with capability flags.
- `compute::Context` (queue + command pool), `compute::Buffer`
  (device-local + host-visible), `compute::ComputePipeline` (SPIR-V +
  descriptor set).
- `kernels::axpy_f32` — SAXPY compiled from shared WGSL at runtime
  (`shader::wgsl_to_spirv`).
- `VK_KHR_cooperative_matrix` GEMM planned for 1.1.
