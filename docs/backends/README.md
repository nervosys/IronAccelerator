# Backend notes

Per-backend build + runtime prerequisites. Every backend loads its
vendor runtime via `libloading` at first use, so the workspace builds
fine on hosts that don't have the SDK installed — the backend just
reports `BackendUnavailable` at runtime.

| Backend        | File                     |
|----------------|--------------------------|
| CUDA           | [`cuda.md`](cuda.md)     |
| ROCm           | [`rocm.md`](rocm.md)     |
| Metal          | [`metal.md`](metal.md)   |
| QNN (Hexagon)  | [`qnn.md`](qnn.md)       |
| Vulkan         | [`vulkan.md`](vulkan.md) |
| OpenGL         | [`opengl.md`](opengl.md) |
| WebGPU         | [`webgpu.md`](webgpu.md) |
| TPU (PJRT)     | [`tpu.md`](tpu.md)       |
| Level Zero     | [`levelzero.md`](levelzero.md) |
| AWS Neuron     | [`neuron.md`](neuron.md) |

## Feature flag convention

Every backend is an optional dependency on the umbrella `ironaccelerator`
crate. Enable exactly the ones you want:

```toml
ironaccelerator = { version = "1", default-features = false, features = ["cuda", "vulkan"] }
```

`features = ["all"]` enables every backend (heavy compile — only do this
when you want the full matrix).
