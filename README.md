# IronAccelerator

A high-performance, **agentic-first** Rust acceleration library spanning
NVIDIA, AMD, Apple, Qualcomm, Intel, Google, and AWS accelerators plus
open cross-vendor APIs (Vulkan / OpenGL / WebGPU).

> **Status: v1.0.** CUDA / ROCm / Metal / QNN backends ship live device
> enumeration + real kernels. FlashAttention-3 (cuDNN v9 backend graph)
> and Flash-MoE land in 1.0. Cross-vendor (Vulkan / OpenGL / WebGPU) and
> vendor-specific (Level Zero / TPU / AWS Neuron) backends compile
> clean, enumerate devices, and carry compute scaffolds; see
> [`ROADMAP.md`](ROADMAP.md) for what flips on the way to 1.1.

## Backend support matrix

| Backend        | Vendor / API                         | Enumerate | Compute scaffold | Real kernels              | Min SDK / runtime                |
|----------------|--------------------------------------|-----------|------------------|---------------------------|----------------------------------|
| CUDA           | NVIDIA                               | ✅        | ✅               | BLAS / cuDNN / FA-3 / MoE | CUDA 12.5+ driver (13.x tested)  |
| ROCm           | AMD                                  | ✅        | ✅               | hipBLASLt (FP8 on gfx942) | ROCm 6.2+                        |
| Metal          | Apple                                | ✅        | ✅               | MPS GEMM                  | macOS 14+ / iOS 17+              |
| QNN            | Qualcomm Hexagon NPU                 | ✅        | ⚠️ HDK needed    | —                         | QNN SDK 2.22+                    |
| Vulkan         | cross-vendor GPU compute             | ✅        | ✅               | SAXPY (WGSL→SPIR-V)       | Vulkan 1.3 ICD                   |
| OpenGL         | legacy / embedded GPU fallback       | ✅ ctx    | ✅               | SAXPY                     | GL 4.3+ compute                  |
| WebGPU         | native (Vk/Metal/DX12) + browser     | ✅        | ✅               | SAXPY WGSL                | wgpu 22 / Chrome 113+            |
| TPU (PJRT)     | Google TPU v4 / v5 / v6e             | ✅ env    | ⏳ PJRT client   | —                         | PJRT plugin (`libtpu.so`)        |
| Level Zero     | Intel GPU (Arc / Flex / PVC) + NPU   | ✅        | ✅               | —                         | `ze_loader` from Intel compute   |
| AWS Neuron     | Trainium / Inferentia                | ✅ cores  | ⏳ NEFF load     | —                         | `libnrt` (Neuron SDK 2.x)        |
| CPU SIMD       | AVX2 / NEON                          | n/a       | ✅               | row-wise INT8 quant       | none                             |

Every backend is loaded via `libloading` at first use — the workspace
builds on any host, present or missing vendor SDK. Missing runtimes
surface as typed `BackendUnavailable` errors, not link failures.

Per-backend build prerequisites and runtime environment variables live in
[`docs/backends/`](docs/backends/).

## Why

Modern AI workloads hit very different optimal kernels per hardware:
FlashAttention-3 + FP8 on Hopper, hipBLASLt on MI300X, MPSGraph on Apple
silicon, QNN-HTP INT8 on Snapdragon. IronAccelerator exposes one
[`Workload`] description and lets a planner — or an *agent* — pick the
right backend, device, and kernel strategy.

## Design principles

- **Zero link-time deps on vendor SDKs.** Every CUDA library is loaded
  through `libloading` at first use. The workspace builds on any machine,
  with or without CUDA installed; missing libraries surface as typed
  `NotAvailable` errors, not link failures.
- **Safe by default, fast by construction.** RAII wrappers over opaque
  handles; typed `Result` everywhere; `#[inline]` on the launch path.
- **Process-global handle caches.** cuBLASLt, cuDNN, cuSOLVER, cuSPARSE,
  cuTENSOR handles are cached per-device-ordinal behind `Mutex<HashMap>`
  and stream-rebound on each borrow.
- **Host-side overhead is measured, not assumed.** Every hot path has a
  Criterion bench. Planner dispatch, FP8-recipe validation, kernel-arg
  packing, and cache-key hashing all run in nanoseconds.

## Workspace layout

```
crates/
  ironaccelerator/            facade crate — runtime + planner + ontology
  ironaccelerator-core/       traits, types, capability flags, Workload/DType
  ironaccelerator-ontology/   machine-readable knowledge graph for agents
  ironaccelerator-cuda-sys/   clean-room CUDA 13.2 FFI + dynamic loader
  ironaccelerator-cuda/       safe CUDA backend (driver, BLAS, DNN, etc.)
  ironaccelerator-rocm/       ROCm/HIP backend scaffold
  ironaccelerator-metal/      Apple Metal/MPS/MLX backend scaffold
  ironaccelerator-qnn/        Qualcomm Hexagon NPU backend scaffold
```

## CUDA backend — what's implemented

| Module        | Coverage                                                       |
|---------------|----------------------------------------------------------------|
| `drv`         | Device, Stream, Event, DeviceBuf, PinnedBuf, Module, launch    |
| `graph`       | Stream capture → `CUgraphExec` execute                         |
| `blas`        | cuBLASLt MatmulDesc + MatrixLayout + heuristic + matmul        |
| `fp8`, `fp8_gemm` | FP8 recipe (Hopper/Blackwell MX) + FP8 GEMM front-end      |
| `cudnn`       | Handle cache + v9 backend-graph `BackendDescr` (finalize+exec) |
| `cusolver`    | getrf / getrs / potrf / geqrf — f32 and f64                    |
| `cusparse`    | DnMat, SpMatCsr, SpMM, SDDMM                                   |
| `cutensor`    | Tensor descriptors + contraction plan+execute                  |
| `cufft`       | 1D/2D/3D + plan-many, per-stream plan cache                    |
| `curand`      | Uniform / normal / u32 / u64 generators                        |
| `nccl`        | Single-process multi-GPU + multi-process collectives           |
| `cupti`       | Enable/disable + activity-buffer decoder (Kernel, Memcpy)      |
| `nvrtc`       | Runtime compilation + kernel cache                             |
| `nvtx`        | Ranges + markers                                               |

All vendor calls flow through typed `Result<T, Error>`; there are no
panics in the hot path, and no allocations on success.

## The ontology

The `ironaccelerator-ontology` crate ships a curated graph that an agent
can query at planning time:

- **HardwareNode** — every supported SKU with arch, FLOPS, bandwidth.
- **WorkloadClass** — every workload family with its roofline regime.
- **StrategyClass** — every implementation path with required capabilities.
- **Optimization** — cross-cutting techniques (fp8_recipe, kv_paging, …).
- **Edge** — weighted `Prefers` / `Supports` / `Requires` relations.

The whole graph round-trips through `to_json()` so it can be exported as a
tool-call schema for an LLM agent loop.

## Performance posture

Every host-side wrapper is benchmarked against the raw driver API. On a
reference run (RTX 3090 Ti, CUDA 13.2, release build):

| Path                                  | Wrapped       | Raw driver    |
|---------------------------------------|---------------|---------------|
| `Stream` create + destroy             | ~650 ns       | ~555 ns       |
| Stream synchronize (empty)            | ~98 ns        | ~98 ns        |
| `Event` create+record+sync+destroy    | ~18 µs        | ~18 µs        |
| Async alloc + free (1 KB … 256 MB)    | ~500–590 ns   | ~430–570 ns   |
| D→D memcpy round-trip (16 MB)         | 48.6 µs (321 GiB/s) | 48.8 µs (320 GiB/s) |
| D→D memcpy round-trip (256 MB)        | 616 µs (406 GiB/s)  | 616 µs (406 GiB/s)  |

Pure-CPU hot paths (host overhead, no GPU involvement):

| Path                                 | Time          |
|--------------------------------------|---------------|
| `capability_from_arch`               | 2–3 ns        |
| `plan_strategy` (full planner)       | 8–9 ns        |
| `Fp8Recipe::validate`                | ~0.3 ns       |
| `LaunchArgs::pack` (12 args, max)    | ~2 ns         |
| GemmKey hash                         | ~7 ns/key     |
| FNV-1a source hash                   | ~1.25 GiB/s   |

The wrapper overhead vs. calling the driver directly is **single-digit
nanoseconds per op** — it never shows up in a kernel-launch profile.

Run the benches yourself:

```bash
cargo bench -p ironaccelerator-cuda --bench host_hot_path
cargo bench -p ironaccelerator-cuda --bench gpu_vs_cudart   # needs a GPU
```

## Building

CUDA / ROCm / Metal / QNN toolkits are **not** required at build time —
every vendor library is loaded dynamically at first use. Missing libraries
return `Error::NotAvailable`; present ones Just Work.

```bash
# Full workspace
cargo build --release

# Just the CUDA backend
cargo build --release -p ironaccelerator-cuda
```

If the driver lives somewhere non-standard, set `IRON_CUDA_LIBDIR` to
prepend a search path before the platform default.

## Roadmap

- [ ] FlashAttention-3 front-end over the cuDNN backend-graph.
- [ ] CUTLASS template instantiation cache.
- [ ] HIP FFI + safe wrapper for `ironaccelerator-rocm`.
- [ ] Metal/MPS/MLX bindings via `objc2`.
- [ ] QNN SDK FFI + HTP graph builder.
- [ ] Cross-backend kernel cache keyed by `(workload, capability)` hash.
- [ ] MCP server exposing the ontology as tool calls.

## License

IronAccelerator is dual-licensed:

- **AGPL-3.0-or-later** — see [`LICENSE`](LICENSE).
- **Commercial license** — available from Nervosys for embedded,
  proprietary, or closed-source SaaS use cases. See
  [`LICENSING.md`](LICENSING.md) for the rationale and contact procedure.
