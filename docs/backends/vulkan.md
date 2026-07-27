# Vulkan backend

Crate: `ironaccelerator-vulkan`. Uses `ash 0.38` with the `loaded`
feature — no static link.

## Build time

Nothing beyond `ash`. Shader translation is not this crate's job — bring
your own SPIR-V, the same contract the CUDA backend has with PTX. The
`naga`-backed WGSL front-end was removed in 2.0.0.

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

Kernels are not provided; `compute::ComputePipeline` takes a SPIR-V
`&[u32]` and builds a pipeline from it.
