# WebGPU backend

Crate: `ironaccelerator-webgpu`. No dependencies beyond
`ironaccelerator-core`.

This is the **browser path only**. On native hosts, Vulkan, Metal, D3D12,
and OpenGL reach the same hardware directly, so there is nothing for a
portability layer to add — see [`dx12.md`](dx12.md) for the Windows API
this replaced the `wgpu` native path with.

## Build time

Nothing. The crate is pure Rust with no bindings and no build script, and
compiles for every target including `wasm32-unknown-unknown` with no
feature flags and no `--cfg` requirements.

Before 2.0.0 this crate pulled `wgpu 22` + `pollster`, which was 98
transitive dependencies and, as it turned out, could not build for
`wasm32` at all.

## Runtime

- **WASM:** a browser with WebGPU enabled (Chrome 113+, Safari 17.4+,
  Firefox 141+).
- **Native:** reports unavailable. Use Vulkan / Metal / D3D12 / OpenGL.

## How binding works

WebGPU adapter negotiation is asynchronous — `requestAdapter()` and
`requestDevice()` both return promises — and `Backend` is a synchronous
trait. A synchronous wrapper would have to block, which a browser's main
thread does not permit.

So the host negotiates and this crate records the result:

```rust
use ironaccelerator_webgpu::{bind_adapter, AdapterInfo};

// once `requestDevice()` has resolved, from your binding layer:
bind_adapter(AdapterInfo {
    vendor: "nvidia".into(),          // GPUAdapterInfo.vendor
    architecture: "ampere".into(),    // GPUAdapterInfo.architecture
    shader_f16: true,                 // the "shader-f16" feature was granted
    max_buffer_size: 1 << 28,
    ..Default::default()
});
```

The `GPUDevice` stays with the host. Once bound, the adapter appears in
`Runtime::devices()` next to every native backend; until then the backend
reports unavailable. `unbind_adapter()` clears it.

Fallback adapters (`isFallbackAdapter`) are recorded but never offered —
a software rasteriser is a correctness reference, not an accelerator.

## Capabilities

- `FP32` always; `FP16` only when `"shader-f16"` was granted.
- Nothing above that: WebGPU has no INT8 dot product, no matrix engine,
  and no query for either.
- `total_memory_bytes` carries `maxBufferSize` — the largest single
  allocation the adapter will honour. WebGPU never reports a memory size,
  and this is the only capacity figure it does expose.

## Not provided

Buffers, pipelines, and dispatch. All three need the live `GPUDevice`,
which the host owns; wrapping them would mean re-exporting a binding
crate's types through this API, which is exactly the coupling that made
the pre-2.0.0 version so heavy. Call WebGPU directly from the host.
