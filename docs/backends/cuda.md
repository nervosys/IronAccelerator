# CUDA backend

Crate: `ironaccelerator-cuda` (+ `ironaccelerator-cuda-sys`).

## Build time

No link-time dependency on the CUDA Toolkit. The crate builds on any
host. You do need the CUDA **runtime** at run time (see below).

## Runtime

- NVIDIA driver **≥ 535** (CUDA 12.x).
- CUDA Toolkit **12.5+** recommended for FP8 grouped-GEMM (and required
  for the MoE grouped-GEMM path once it lands in 1.1).
- Libraries loaded lazily: `nvcuda.dll` / `libcuda.so.1`,
  `cudart`, `cublas`, `cublasLt`, `cudnn`, `cusolver`, `cusparse`,
  `cufft`, `curand`, `cutensor`, `nccl`, `nvrtc`.
- First `cuda::register` call calls `cuInit(0)` and caches handles
  per-device-ordinal.

## Capabilities

- cuBLASLt matmul + FP8 recipe (E4M3 / E5M2) on Hopper / Blackwell.
- cuDNN v9 backend-descriptor graph → FlashAttention-3.
- NVRTC runtime kernel compilation.
- NCCL collectives.
- Flash-MoE per-expert matmul loop (grouped-GEMM variant tracked in
  `ROADMAP.md`).
