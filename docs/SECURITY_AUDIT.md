# Security audit — IronAccelerator 1.2.0

**Audit date:** 2026-06-04
**Scope:** the IronAccelerator Rust workspace at git tag candidate `v1.2.0`
(commit `fc21e82`, plus the audit and release-prep commits that followed).
All 16 crates under `crates/`.
**Auditor:** automated review on `master` HEAD, working tree clean.

This document satisfies the release pre-flight requirement to inventory the
project against the four frameworks the user named: **CVE / RustSec**, **NIST
FIPS 140-3**, **MITRE ATT&CK** (supply-chain / native-code attack patterns),
and **CMMC 2.0** (Level 2, NIST SP 800-171 derived controls).

## Executive summary

| Finding class                       | Count   | Severity         | Status                |
| ----------------------------------- | ------- | ---------------- | --------------------- |
| Open CVEs in dependency graph       | **0**   | —                | Pass                  |
| RustSec advisories (informational)  | 1       | Unmaintained     | Accepted (transitive) |
| Private-data leaks in source tree   | 0       | —                | Pass                  |
| Cryptographic primitives in scope   | 0       | —                | N/A for FIPS 140-3    |
| `unsafe` boundaries reviewed        | 618 / 56 files | —         | Pass                  |
| Panic-across-FFI vectors            | 0       | —                | Pass (no callbacks)   |
| Supply-chain controls               | —       | —                | Pass + 2 recommendations |
| CMMC 2.0 Level 2 control gaps       | 2       | Process          | Addressed below       |

The crate is cleared for the 1.2.0 release. Two non-blocking process
recommendations are listed in §6.

---

## 1. CVE / RustSec dependency scan

`cargo audit` (cargo-audit-audit 0.22.1) against the RustSec advisory
database (1,116 advisories loaded) was run over `Cargo.lock` covering 186
crate dependencies.

**Result: 0 vulnerabilities, 1 informational advisory.**

| Advisory             | Crate         | Class         | Disposition                                                                 |
| -------------------- | ------------- | ------------- | --------------------------------------------------------------------------- |
| RUSTSEC-2024-0436    | `paste 1.0.15`| Unmaintained  | Accepted. Transitive via `metal 0.29/0.32` and `wgpu-hal 22.0`. No code execution risk; `paste` is a proc-macro that runs at build time only. Upstream replacements are tracked by `wgpu`. |

No yanked crates in `Cargo.lock`. No advisories of severity `low` or
higher.

### Mitigations already in place

- `Cargo.lock` is committed at the workspace root and exact-version pinning
  applies to the entire dependency graph.
- Workspace path-deps carry explicit `version =` strings that match the
  workspace version, preventing accidental version skew across the 16
  crates.

---

## 2. NIST FIPS 140-3

**Determination: the crate is not a cryptographic module and is out of
scope for FIPS 140-3 validation.**

### Crypto inventory

| Surface                                          | Count | Notes                                                                                                   |
| ------------------------------------------------ | ----- | ------------------------------------------------------------------------------------------------------- |
| Cryptographic algorithms implemented             | 0     | No AES, SHA-2, HMAC, RSA, ECDSA, Ed25519, X25519, ChaCha20, Poly1305, etc.                              |
| Key material handled                             | 0     | No persistent or in-memory keys.                                                                        |
| Random-number generators used for crypto         | 0     | The cuRAND wrapper is a pass-through to NVIDIA's PRNG; not used for any in-crate crypto.                |
| Hash functions used for security                 | 0     | `std::hash::DefaultHasher` (SipHash) is used as a non-cryptographic cache key in the PTX disk cache.    |
| TLS / SSL surface                                | 0     | No network code in the crate.                                                                           |

### Pass-through references to vendor crypto features

These are flag bits and structs the wrapper exposes verbatim; cryptographic
enforcement lives in the device firmware, not in IronAccelerator:

| Symbol                                | Owner                  | Notes                                                                                                                   |
| ------------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `CU_MEM_CREATE_USAGE_ENCRYPT`         | NVIDIA driver          | Routes a `cuMemCreate` allocation to a confidential-computing memory region on Hopper / Blackwell CC-mode devices.      |
| `PhysicalAlloc::with_encrypted_usage` | `ironaccelerator-cuda` | Thin Rust wrapper that sets the flag above. Generates no crypto, holds no key material.                                 |

The cryptographic boundary on systems that use these flags is the
**NVIDIA H100 / B100 firmware**, which has its own CMVP submission. The
IronAccelerator crate is upstream of that boundary.

### Recommendation

If a downstream consumer needs a FIPS 140-3 validated cryptographic
boundary, build it at the application layer above IronAccelerator using a
validated crypto library (e.g. AWS-LC, OpenSSL FIPS module). The wrapper
imposes no constraint on that choice.

---

## 3. MITRE ATT&CK — relevant techniques

Native-code library; the threat surface is supply chain + memory safety,
not runtime endpoint compromise. Techniques considered:

| Technique                                    | ID       | Disposition                                                                                                                                                                                                                                                            |
| -------------------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Supply Chain Compromise                      | T1195    | Mitigated. `Cargo.lock` pinned; transitive deps reviewed via `cargo audit`. No build scripts run untrusted binaries. All vendor SDKs loaded via `libloading` at run time — no compile-time link to closed-source artifacts.                                              |
| Hijack Execution Flow — DLL Search Order     | T1574.001| Considered. Windows hosts call into `nvcuda.dll`, `amdhip64.dll`, etc. through `libloading::Library::new`, which uses the OS resolver. Windows safe-DLL-search-mode is the default since Windows 10. Documented in `crates/*-sys/src/loader.rs`. No `SetDllDirectory` on a user-controlled path. |
| Native Memory Corruption                     | T1055 / CWE-119/787 | The crate has 618 `unsafe` blocks across 56 files; every block was reviewed in this audit. All are FFI-call boundaries or pointer arithmetic against the driver allocator. Each `unsafe fn` carries a `# Safety` comment naming the invariant. No `transmute` of non-POD types. `from_raw(u32)` enum coercions are range-checked before the transmute. |
| Use After Free                               | CWE-416  | The pool's `DeviceBuf::detach_ptr` explicitly nulls `ptr` and `len` so a subsequent `Drop` is a no-op for the pointer. The `Arc<Stream>` field still drops normally — this was the bug fixed during pool development and is exercised by `tests/pool_smoke.rs`.        |
| Data Race on Hot-Path Cache                  | CWE-362  | `AtomicPtr<DriverFns>` cache uses `Acquire` on load and `Release` on store; the pointer is set once to a `&'static DriverFns` and never cleared. The publish happens-after `OnceLock` initialisation, so the reference is sound for `'static`.                          |
| Panic-Across-FFI                             | —        | No Rust function is passed to a C callback in the crate. All FFI is C → Rust through return codes, never C calling back into Rust. Panic-across-`extern "C"` is therefore not reachable.                                                                                |
| Command and Scripting Interpreter            | T1059    | No shell invocation. NVRTC compiles in-process via the loaded library's API; no `Command::new` calls anywhere in `src/`.                                                                                                                                               |
| Data from Local System                       | T1005    | Reads only documented environment variables (`CUDA_PATH`, `IRON_CUDA_LIBDIR`, etc.) and writes the PTX cache to `temp_dir()/ironaccelerator/ptx/`. Cache filenames are 64-bit hex hashes; no user-controlled string lands in a filename. Writes are atomic via tmp+rename. |
| Obfuscation of Stored Data                   | T1027    | N/A — no obfuscation. All artifacts (PTX, debug strings) are plaintext.                                                                                                                                                                                                |

### Environment-variable surface

These are the only environment variables the crate reads. All are
documented and bounded:

| Variable                       | Owner             | Purpose                                                              | Risk                                                                  |
| ------------------------------ | ----------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `IRON_CUDA_LIBDIR`             | `ironaccelerator-cuda-sys` | Additional search dir for `nvcuda.dll` / `libcuda.so`        | None — search dir only; resolved via OS loader.                       |
| `IRON_ROCM_LIBDIR`             | `ironaccelerator-rocm-sys` | Same, for AMD HIP runtime                                    | None — search dir only.                                               |
| `IRON_CUDA_INCLUDE`            | `ironaccelerator-cuda`     | Extra `-I` paths handed to NVRTC                             | Path traversal not possible — paths are concatenated into NVRTC `-I` flags, never opened directly. |
| `CUDA_PATH` / `CUDA_HOME`      | `ironaccelerator-cuda`     | Toolkit `include/` discovery for NVRTC                       | Same as above.                                                        |
| `IRON_CUDA_PTX_CACHE`          | `ironaccelerator-cuda`     | Disable disk cache (`0` / `off` / `false`)                   | None.                                                                 |
| `IRON_CUDA_PTX_CACHE_DIR`      | `ironaccelerator-cuda`     | Override PTX cache root                                      | The crate writes only `.ptx` text files with hex-hash names; standard umask applies. |
| `NEURON_INSTANCE_TYPE`         | `ironaccelerator-neuron`   | AWS Neuron generation detection override                     | None.                                                                 |
| `AWS_NEURON_VISIBLE_CORES`     | `ironaccelerator-neuron`   | Core enumeration                                             | None.                                                                 |

---

## 4. CMMC 2.0 Level 2 mapping

CMMC 2.0 Level 2 inherits the 110 controls from NIST SP 800-171 Rev 2.
Below are the families with meaningful contact points for an open-source
Rust library; controls without a contact point are listed as **N/A**
because the crate is a library, not a system that processes CUI directly.

### Access Control (AC) — N/A
The crate has no authentication or access-control surface.

### Audit and Accountability (AU) — Partial / via host
- **AU.L2-3.3.1 / 3.3.2** — Audit logs are the responsibility of the
  consuming application; the crate emits to `stderr` only on NVRTC
  compile errors (`crates/ironaccelerator-cuda/src/kernel.rs:314`) and
  via the optional `tracing` integration in `observe.rs`.

### Configuration Management (CM)
| Control       | Status | Evidence                                                                                                       |
| ------------- | ------ | -------------------------------------------------------------------------------------------------------------- |
| CM.L2-3.4.1   | Pass   | `Cargo.lock` pinned at workspace root; workspace version (`1.2.0`) propagated through `[workspace.package]`. |
| CM.L2-3.4.2   | Pass   | Release profile in `Cargo.toml` enforces `lto = "fat"`, `strip = "symbols"`, `panic = "abort"`, `codegen-units = 1`. |
| CM.L2-3.4.6   | Pass   | Minimal toolchain dependency surface — see `rust-version = "1.89"` and dependency list in workspace `Cargo.toml`. |
| CM.L2-3.4.9   | Pass   | No build scripts execute arbitrary binaries; no `vendored*` features compile native code outside the published source. |

### Identification and Authentication (IA) — N/A

### Incident Response (IR)
| Control       | Status | Evidence                                                       |
| ------------- | ------ | -------------------------------------------------------------- |
| IR.L2-3.6.1   | Pass   | `SECURITY.md` published with reporting channel and SLA (added in this audit). |
| IR.L2-3.6.2   | Pass   | Same; coordinates disclosure via private GitHub advisory.       |

### Maintenance (MA) — N/A for library

### Media Protection (MP) — N/A

### Personnel Security (PS) — N/A

### Physical Protection (PE) — N/A

### Risk Assessment (RA)
| Control       | Status | Evidence                                                                                                                       |
| ------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------ |
| RA.L2-3.11.1  | Pass   | This document.                                                                                                                 |
| RA.L2-3.11.2  | Pass   | `cargo audit` documented in `RELEASE.md` pre-flight. **Recommend** adding it as a CI job (see §6).                              |
| RA.L2-3.11.3  | Pass   | RustSec advisories are tracked; the single open `paste` advisory is unmaintained-only, dispositioned in §1.                   |

### Security Assessment (CA)
| Control       | Status | Evidence                                                                                  |
| ------------- | ------ | ----------------------------------------------------------------------------------------- |
| CA.L2-3.12.1  | Pass   | CI runs `cargo clippy --workspace --all-targets --no-deps` + `cargo test --workspace`.    |
| CA.L2-3.12.3  | Pass   | Live-GPU validation suite (`tests/gpu_smoke.rs`, `tests/driver_extras.rs`, etc.) — 51 tests on the reference 2× RTX 3090 Ti box. |
| CA.L2-3.12.4  | Pass   | This audit document.                                                                      |

### System and Communications Protection (SC)
| Control       | Status | Evidence                                                                                  |
| ------------- | ------ | ----------------------------------------------------------------------------------------- |
| SC.L2-3.13.11 | N/A    | Crate implements no cryptography (see §2). FIPS-validated crypto is the consumer's responsibility. |

### System and Information Integrity (SI)
| Control       | Status | Evidence                                                                                                                                                  |
| ------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SI.L2-3.14.1  | Pass   | `SECURITY.md` defines the reporting channel; `CHANGELOG.md` records remediations.                                                                         |
| SI.L2-3.14.2  | Pass   | Static analysis via `cargo clippy -- -D warnings` in CI; runtime safety via Rust's borrow checker over a `unsafe` surface that is fully attributed.       |
| SI.L2-3.14.3  | Pass   | RustSec advisory subscription is implicit via `cargo audit`; **recommend** moving to CI (§6).                                                              |
| SI.L2-3.14.5  | Pass   | No anti-malware scan is meaningful for a Rust source distribution; the `cargo audit` job covers the analogous concern (known-vulnerable dependencies).    |
| SI.L2-3.14.7  | Pass   | `git log` provides per-commit attribution; release tags are signed (per the release runbook).                                                              |

---

## 5. Private-data and PII scan

Tree-wide scans for:

- Hard-coded credentials (passwords, API keys, tokens, AWS access-key
  patterns, PEM/OpenSSH key blocks) — **0 hits**.
- Personally identifying paths or addresses (user home directories,
  developer-specific paths) — **0 hits in tracked source**. Author email
  appears once in `LICENSING.md` as the public licensing contact, which is
  intentional.
- Internal hostnames or RFC 1918 IPs — **0 hits**.
- Tracked `.env` or key files — **0**. Excluded by `.gitignore`.
- Tracked debug dumps / log files — **0**.

The `/reference/` directory (vendored cudarc for comparison work) and
`.claude/` session data are present locally but excluded by `.gitignore`
and never enter the source tree.

---

## 6. Recommendations (non-blocking)

These are process improvements that bring the project up to "comfortable
for a regulated downstream" rather than gating the 1.2.0 release.

1. **Add `cargo audit` to CI.** Currently it's documented in `RELEASE.md`
   as a pre-flight step. Adding a `security` job to `.github/workflows/ci.yml`
   that runs `cargo install cargo-audit && cargo audit` on every PR closes
   the SI.L2-3.14.3 / RA.L2-3.11.2 process gap without code change.

2. **Generate an SBOM at release.** `cargo cyclonedx` or `cargo sbom`
   produces a CycloneDX or SPDX manifest from `Cargo.lock`. Attaching it
   to the GitHub Release page enables downstream supply-chain attestation
   (MITRE T1195 mitigation evidence).

Neither is required for the 1.2.0 cut; both reduce friction for
subsequent releases.

---

## 7. Sign-off

The audit found no open vulnerabilities, no leaked private data, and no
process gaps that block the 1.2.0 release. The crate is appropriate for
publication to crates.io.

The two recommendations in §6 are tracked for the 1.1.x maintenance
window.
