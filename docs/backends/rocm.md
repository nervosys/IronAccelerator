# ROCm backend

Crate: `ironaccelerator-rocm` (+ `ironaccelerator-rocm-sys`).

## Build time

No link-time dependency. Wrappers over ROCm libraries load lazily.

## Runtime

- ROCm **6.2+**.
- Linux only (no Windows HIP runtime).
- Loads `libamdhip64.so`, `librocblas.so`, `libhipblas.so`,
  `libhipblaslt.so`, `libmiopen.so`, etc.
- `HIP_VISIBLE_DEVICES` honoured.

## Capabilities

- hipBLASLt matmul with FP8 on **gfx942** (MI300X).
- Composable Kernel / MIOpen safe wrappers parked for 1.2.
