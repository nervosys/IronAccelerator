# Intel Level Zero backend

Crate: `ironaccelerator-levelzero`. Intel oneAPI Level Zero — GPUs
(Arc, Flex, Ponte Vecchio, Battlemage) and NPUs (Meteor / Arrow / Lunar
Lake VPU).

## Build time

No SDK — `ze_loader` is loaded at run time.

## Runtime

- `ze_loader.dll` (Windows) or `libze_loader.so.1` (Linux).
- Part of Intel's compute-runtime / L0 driver package.
- `zeInit(0)` is called on first use; failures surface as
  `BackendUnavailable`.

## Capabilities

- Driver + device enumeration via `zeDriverGet` / `zeDeviceGet`.
- `ze_device_type_t` distinguishes GPU from VPU (NPU) in one backend.
- `compute::Context` — `zeContextCreate` + `zeCommandQueueCreate`
  (ordinal 0 = default compute group) + `zeCommandListCreate`.
- Buffer allocation (`zeMemAllocDevice` / `zeMemAllocShared`) and
  SPIR-V module load + `zeKernelCreate` + dispatch tracked in
  `ROADMAP.md`.
