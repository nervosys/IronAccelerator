# QNN (Qualcomm Hexagon NPU) backend

Crate: `ironaccelerator-qnn` (+ `ironaccelerator-qnn-sys`).

## Build time

Clean-room FFI over QNN SDK headers; no SDK needed at build.

## Runtime

- Qualcomm QNN SDK **2.22+** installed on device.
- Loads a backend provider `.so`:
  - `libQnnCpu.so` — reference CPU.
  - `libQnnGpu.so` — Adreno GPU.
  - `libQnnDsp.so` — Hexagon DSP.
  - `libQnnHtp.so` — Hexagon HTP (NPU, fastest).
- `QNN_SDK_ROOT` env var optional; backends can also be loaded by
  absolute path through the public API.

## Capabilities

- Backend → Device → Context → Graph object lifecycle.
- Compiled-graph serialise + rehydrate in-progress (tracked in
  `ROADMAP.md`).
- End-to-end `Graph::execute` needs a Hexagon HDK box for CI.
