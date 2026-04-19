# TPU (PJRT) backend

Crate: `ironaccelerator-tpu`. Google Cloud TPUs via the PJRT plugin
interface.

## Build time

No SDK — plugin is loaded at run time.

## Runtime

- Plugin: `libtpu.so` (TPU VM). Located by `PJRT_PLUGIN_PATH`
  or one of the default search paths.
- Host VM with TPU v4 / v5 / v5e / v5p / v6e topology.
- Topology surfaced from env:
  - `TPU_ACCELERATOR_TYPE` (e.g. `v5e-8`, `v5p-16`, `v6e-256`).
  - `TPU_NUM_DEVICES`.
  - `TPU_CHIPS_PER_HOST`.

## Capabilities

- Plugin loader (`GetPjrtApi` symbol probe).
- Env-driven topology enumeration → `DeviceDescriptor`.
- Real `PJRT_Client_Create` + StableHLO compile + execute tracked in
  `ROADMAP.md`.
