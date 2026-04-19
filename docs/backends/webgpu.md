# WebGPU backend

Crate: `ironaccelerator-webgpu`. Uses `wgpu 22` + `pollster 0.4`.

## Build time

Pulls in `wgpu 22`. On native, wgpu picks Vulkan / Metal / DX12 / GLES
at adapter request.

## Runtime

- **Native:** any wgpu adapter — Vulkan driver, Metal (macOS), DX12
  (Windows), or GLES fallback. Nothing extra to install.
- **WASM:** browser with WebGPU enabled (Chrome 113+, Safari 17.4+,
  Firefox Nightly behind flag). Host calls `drv::bind_device` with the
  pre-selected device from `navigator.gpu`.

## Capabilities

- Adapter + device enumeration via `AdapterInfo`.
- `compute::Context` (block_on adapter request) + `ComputePipeline`
  from WGSL + `dispatch`.
- `kernels::axpy_f32` SAXPY reference.
- Real GEMM WGSL + `wasm-bindgen-test` harness tracked in `ROADMAP.md`.
