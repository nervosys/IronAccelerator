# Metal backend

Crate: `ironaccelerator-metal`.

## Build time

Apple Silicon or Intel Mac. Links against the system `Metal.framework`
and `MetalPerformanceShaders.framework` via `objc2`. No extra SDK.

## Runtime

- macOS **14+** (Sonoma) or iOS **17+**.
- MPSGraph available on all Apple Silicon; Intel Macs get the subset
  MPS exposes.

## Capabilities

- `MPSMatrixMultiplication` GEMM path.
- Metal command-queue dispatch skeleton for custom `.metallib`.
- CoreML / ANE + MLX JIT tracked post-1.1.
