# Security policy

## Supported versions

IronAccelerator follows semver. Security fixes are issued against the latest
released minor; older minors are out of scope unless explicitly noted in
[`CHANGELOG.md`](CHANGELOG.md).

| Version | Supported |
| ------- | --------- |
| 1.2.x   | ✅        |
| < 1.2   | ❌        |

## Reporting a vulnerability

**Do not open a public GitHub issue for security bugs.** Use one of the
private channels below.

- **Email** — `opensource@nervosys.ai` (preferred)
- **GitHub Security Advisory** — open a draft advisory on
  <https://github.com/nervosys/IronAccelerator/security/advisories>

Please include:

1. A description of the issue and its impact.
2. Steps to reproduce, ideally as a minimal Rust test or PTX/HSACO snippet.
3. The crate version and host platform (`cargo --version`, `nvidia-smi` /
   `rocminfo` output where relevant).
4. Whether you intend to publish a write-up; we will coordinate disclosure
   timing if you wish.

We aim to acknowledge reports within 3 business days and ship a fix within
30 days for high-severity findings.

## Scope

In scope:

- Memory-safety regressions in the `unsafe` FFI surface.
- Soundness bugs in the safe wrappers (`Send`/`Sync`, lifetime escapes,
  use-after-free against the driver allocator).
- Crash-on-malformed-input through any public API.
- Supply-chain issues affecting `Cargo.lock` (yanked crates, dependency
  confusion).
- Privilege-escalation or sandbox-escape made possible by misuse of a
  vendor driver call that the wrapper exposes.

Out of scope:

- Vulnerabilities in the underlying vendor SDK (CUDA driver, ROCm, Metal,
  Vulkan, etc.) — report those to the vendor.
- Denial-of-service from a kernel the caller passes in via NVRTC — the
  wrapper compiles and launches what it's given.
- Performance regressions (those are bugs, not security issues — open a
  normal GitHub issue).

## Cryptographic posture

IronAccelerator implements no cryptographic primitives. It exposes flags
that route to the underlying vendor's confidential-computing surface (for
example NVIDIA Hopper/Blackwell CC mode via `cuMemCreate` with
`CU_MEM_CREATE_USAGE_ENCRYPT`), but the cryptographic boundary is the
device firmware, not this crate. See [`docs/SECURITY_AUDIT.md`](docs/SECURITY_AUDIT.md)
for the full FIPS 140-3 and CMMC 2.0 disposition.

## Supply-chain integrity

- `Cargo.lock` is checked in for the workspace; releases are tagged and
  published from CI on a clean checkout.
- Every backend loads its vendor SDK through `libloading` at run time;
  there are no compile-time links to closed-source binaries.
- Dependency advisories are tracked via `cargo audit` against the RustSec
  advisory database; see `RELEASE.md` for the pre-flight checklist.
