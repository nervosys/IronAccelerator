# AWS Neuron backend

Crate: `ironaccelerator-neuron`. AWS Trainium / Inferentia via the
Neuron Runtime (`libnrt`).

## Build time

No SDK — `libnrt.so` is loaded at run time.

## Runtime

- EC2 instance types: `inf1`, `inf2`, `trn1`, `trn1n`, `trn2`.
- Neuron SDK **2.x** installed on the instance.
- `libnrt.so` on the library search path.
- `nrt_init()` called on first use.

## Capabilities

- NeuronCore count + Neuron runtime version.
- Generation detection (Inf1 / Trn1 / Trn2) with FP8 flag on Trn1/Trn2
  and INT4 on Trn2.
- NEFF-binary load (`nrt_load`) + tensor I/O execute (`nrt_execute`)
  tracked in `ROADMAP.md`.
