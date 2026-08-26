# Hardware validation in CI — what runs where, and why

The project's rule is that a backend is not called "supported" until its path
has run on a real device. That runs into a blunt fact about GitHub Actions:
**GitHub-hosted runners provide CPUs and real Macs — and nothing else.** No AMD
GPU, no Intel GPU, no FPGA, no Gaudi. This doc says exactly which backends CI
can validate for free, which need hardware you attach, and which it cannot do
at all.

## Three tiers of hardware path

### 1. Free on GitHub-hosted runners

- **Metal (Apple GPU).** GitHub's `macos-latest` runners are real Macs with a
  Metal-capable GPU — the runner enumerates a Metal device (that is how the
  `capabilities()`/`enumerate()` BF16 mismatch was originally caught). The
  **`metal / live (macOS GPU)`** job in [`ci.yml`](../../.github/workflows/ci.yml)
  runs the `ComputeDevice` round-trip with `--nocapture`, so the log shows
  whether the dispatch genuinely executed ("roundtrip verified over N floats")
  or skipped, rather than that outcome hiding inside the umbrella `check`
  job's pass.

The Windows and Linux hosted runners have **no GPU**. The D3D12 / Vulkan /
OpenGL dispatch tests there fall back to whatever software rasteriser exists
(D3D12 WARP, if present) or skip cleanly; they were validated on real
multi-adapter hardware on the maintainer's box, not on hosted CI.

### 2. Needs a self-hosted runner (your hardware, GitHub's orchestration)

AMD GPU (ROCm), Intel GPU (Level Zero), and FPGA (XRT) reach CI **only** through
a [self-hosted runner](https://docs.github.com/en/actions/hosting-your-own-runners):
your machine, registered to the repo with a label, driven by GitHub Actions like
any hosted job. The [`hardware.yml`](../../.github/workflows/hardware.yml)
workflow is the wiring — manual-dispatch only, so it never leaves queued runs on
a repo that has no such runner.

**To close the ROCm gap for real:**

1. On an AMD-GPU Linux box with ROCm 6.2+ installed (`libamdhip64`,
   `libhiprtc`), register a runner:
   ```bash
   ./config.sh --url https://github.com/nervosys/IronAccelerator \
               --labels self-hosted,linux,rocm
   ./run.sh
   ```
2. Actions tab → **hardware** → **Run workflow** (lane: `rocm`).
3. The `rocm_smoke` test runs end-to-end — HIPRTC compile → module load →
   kernel launch → `MemPool` round-trip — with `IRON_RUN_GPU_TESTS=1`. Green
   here is the signal that promotes ROCm's `†` (compiled, not live-tested)
   rows in [`STATUS.md`](STATUS.md) to validated.

Intel-GPU and FPGA lanes follow the identical pattern; the workflow's trailing
comment shows how to add them.

### 3. No path GitHub can offer

- **Google TPU** and **AWS Neuron** live only inside their cloud (TPU VMs, trn/
  inf instances). A self-hosted runner *on such an instance* could validate them,
  but that is a cloud-provisioning task, not a GitHub-CI one.
- **Intel Gaudi (Habana)** and the AI-chip startups (Groq, Cerebras, …) likewise
  need their own hosts; none is wired here.

## How the smoke tests gate

Every live smoke test skips-and-passes when its device is absent, which is what
keeps hosted-runner CI green with no accelerators at all. The opt-in differs by
backend maturity:

- **CUDA** (`gpu_smoke`) gates on **device presence only** — CUDA is the
  production backend, so it validates automatically wherever a GPU exists (the
  maintainer's box, a CUDA self-hosted runner).
- **ROCm** (`rocm_smoke`) additionally requires **`IRON_RUN_GPU_TESTS=1`** — a
  deliberate opt-in for a backend that has not yet been proven on hardware, so
  it stays inert for a developer who merely has ROCm installed and only fires in
  the `hardware.yml` lane that sets it. Once ROCm has been validated on a real
  runner, this second gate can be dropped to match CUDA.
