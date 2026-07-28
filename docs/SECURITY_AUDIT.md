# Security audit — IronAccelerator 2.0.0

**Audit date:** 2026-07-28
**Scope:** the IronAccelerator Rust workspace at `master` HEAD (commit
`3a86699`), working tree clean. All 16 crates under `crates/`.
**Auditor:** automated review on `master` HEAD.

This document satisfies the release pre-flight requirement to inventory the
project against the four frameworks the user named: **CVE / RustSec**, **NIST
FIPS 140-3**, **MITRE ATT&CK** (supply-chain / native-code attack patterns),
and **CMMC 2.0** (Level 2, NIST SP 800-171 derived controls).

> **2.0.0 delta.** Since the 1.2.0 audit the WebGPU path was rewritten to
> drop `wgpu`/`naga` entirely (browser-bound host binding only), a
> hand-written Direct3D 12 backend was added (`ironaccelerator-dx12`), and
> the CUDA host copy paths were reworked with pinned staging. A
> covert-telemetry feature proposed during development was **rejected and
> never merged**; the single commit that briefly landed a disclosed
> opt-out variant was reverted (`91f966e`). There is **no build script and
> no network egress anywhere in the shipped graph** — see §3.

## Executive summary

| Finding class                        | Count            | Severity     | Status                   |
| ------------------------------------ | ---------------- | ------------ | ------------------------ |
| Open CVEs in dependency graph        | **0**            | —            | Pass                     |
| RustSec advisories (informational)   | 1                | Unmaintained | Accepted (dev-only)      |
| Private-data leaks in source tree    | 0                | —            | Pass                     |
| Embedded secrets in source tree      | 0                | —            | Pass                     |
| Cryptographic primitives in scope    | 0                | —            | N/A for FIPS 140-3       |
| Build scripts / install-time egress  | 0                | —            | Pass                     |
| `unsafe` boundaries reviewed         | 1,030 tokens / 53 files | —      | Pass                     |
| Panic-across-FFI vectors             | 0                | —            | Pass (no callbacks)      |
| Supply-chain controls                | —                | —            | Pass                     |
| CMMC 2.0 Level 2 control gaps        | 0                | —            | Closed (audit in CI)     |

The crate is cleared for the 2.0.0 release. One non-blocking process
recommendation (SBOM at release) is listed in §6.

---

## 1. CVE / RustSec dependency scan

`cargo audit` (cargo-audit-audit 0.22.2) against the RustSec advisory
database (1,172 advisories loaded) was run over `Cargo.lock` covering **120
crate dependencies**.

**Result: 0 vulnerabilities, 1 informational advisory.**

| Advisory          | Crate          | Class        | Disposition                                                                                                                   |
| ----------------- | -------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| RUSTSEC-2024-0436 | `paste 1.0.15` | Unmaintained | Accepted. Transitive and **dev-only** (`criterion` benchmark harness → `paste`); not in the shipped graph of any crate. `paste` is a build-time proc-macro with no runtime or code-execution risk. |

No yanked crates in `Cargo.lock`. No advisories of severity `low` or higher.

### Shipped vs. dev graph

The heavy transitive deps (`criterion`, `serde_json`, `zerocopy`, `cudarc`,
`half`'s `ciborium` chain) enter only through **`[dev-dependencies]`** used
by the A/B benchmark harness (`examples/ab_vs_cudarc.rs`, benches). Verified
with `cargo tree -e no-dev`: the only non-dev transitive of note is
`zerocopy 0.8.48`, pulled by the optional `half` feature — a widely-audited
crate. Consumers who do not enable `half` do not compile it.

> **Observation (informational).** `zmij 1.0.21` appears as a transitive
> dependency of `serde_json` under the `criterion` dev harness. It never
> reaches a published artifact or a consumer build. Flagged only for
> provenance awareness; no exposure.

### Mitigations already in place

- `Cargo.lock` is committed at the workspace root; exact-version pinning
  applies to the entire dependency graph.
- Workspace path-deps carry explicit `version =` strings that match the
  unified workspace version (`2.0.0`), preventing version skew across the
  16 crates.
- **`cargo audit` runs in CI** on every push and PR (`.github/workflows/ci.yml`,
  `audit` job via `rustsec/audit-check`), failing on any vulnerability
  advisory while tolerating `unmaintained` warnings.

---

## 2. NIST FIPS 140-3

**Determination: the crate is not a cryptographic module and is out of
scope for FIPS 140-3 validation.**

### Crypto inventory

| Surface                                  | Count | Notes                                                                                                |
| ---------------------------------------- | ----- | ---------------------------------------------------------------------------------------------------- |
| Cryptographic algorithms implemented     | 0     | No AES, SHA-2, HMAC, RSA, ECDSA, Ed25519, X25519, ChaCha20, Poly1305, etc.                           |
| Key material handled                     | 0     | No persistent or in-memory keys.                                                                     |
| RNGs used for crypto                     | 0     | The cuRAND FFI wrapper is a pass-through to NVIDIA's PRNG; not used for any in-crate crypto.          |
| Hash functions used for security         | 0     | `std::hash::DefaultHasher` (SipHash) is used as a non-cryptographic cache key in the PTX disk cache. |
| Crypto crates in `Cargo.lock`            | 0     | No `ring`, `rustls`, `openssl`, `getrandom`, `sha2`, `aes`, `hmac`, etc.                             |
| TLS / SSL surface                        | 0     | No network code in the crate.                                                                        |

### Pass-through references to vendor crypto features

These are flag bits and structs the wrapper exposes verbatim; cryptographic
enforcement lives in the device firmware, not in IronAccelerator:

| Symbol                                | Owner                  | Notes                                                                                                              |
| ------------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `CU_MEM_CREATE_USAGE_ENCRYPT`         | NVIDIA driver          | Routes a `cuMemCreate` allocation to a confidential-computing memory region on Hopper / Blackwell CC-mode devices. |
| `PhysicalAlloc::with_encrypted_usage` | `ironaccelerator-cuda` | Thin Rust wrapper that sets the flag above. Generates no crypto, holds no key material.                           |

The cryptographic boundary on systems that use these flags is the **NVIDIA
H100 / B100 firmware**, which has its own CMVP submission. IronAccelerator
is upstream of that boundary.

### Recommendation

If a downstream consumer needs a FIPS 140-3 validated cryptographic
boundary, build it at the application layer above IronAccelerator using a
validated crypto library (e.g. AWS-LC, OpenSSL FIPS module). The wrapper
imposes no constraint on that choice.

---

## 3. MITRE ATT&CK — relevant techniques

Native-code library; the threat surface is supply chain + memory safety,
not runtime endpoint compromise. Techniques considered:

| Technique                                | ID        | Disposition                                                                                                                                                                                                                                                          |
| ---------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Supply Chain Compromise                  | T1195     | Mitigated. **No `build.rs` in any crate** — zero install/compile-time code execution. `Cargo.lock` pinned; transitive deps reviewed via `cargo audit` in CI. All vendor SDKs loaded via `libloading` at run time — no compile-time link to closed-source artifacts. |
| Exfiltration / C2 over app protocol      | T1041 / T1071 | Mitigated. **No network code in any shipped crate** — no `reqwest`/`hyper`/`ureq`/`std::net`/socket symbols. The proposed telemetry exporter was rejected and the transient commit reverted (`91f966e`).                                                        |
| Unsecured Credentials                    | T1552     | Mitigated. Tree-wide scan for bearer tokens / API keys / password literals: **0 hits**. No credential, `.env`, `.pem`, or key files tracked in git.                                                                                                                 |
| Hijack Execution Flow — DLL Search Order | T1574.001 | Considered. Windows hosts resolve `nvcuda.dll`, `amdhip64.dll`, `d3d12.dll`, etc. through `libloading::Library::new`, which uses the OS resolver under safe-DLL-search mode (Windows 10+ default). No `SetDllDirectory` on a user-controlled path.                    |
| Native Memory Corruption                 | T1055 / CWE-119/787 | 1,030 `unsafe` token occurrences across 53 files; the bulk are `unsafe extern "C" fn` pointer types in the hand-written FFI vtables (321 in `-cuda-sys` alone). All are FFI-call boundaries or pointer arithmetic against the driver allocator. `from_raw(u32)` enum coercions are range-checked before transmute; no `transmute` of non-POD types. |
| Use After Free                           | CWE-416   | The pool's `DeviceBuf::detach_ptr` nulls `ptr`/`len` so a subsequent `Drop` is a no-op for the pointer. Exercised by the pool smoke tests.                                                                                                                          |
| Data Race on Hot-Path Cache              | CWE-362   | `AtomicPtr<DriverFns>` cache uses `Acquire`/`Release`; the pointer is set once to a `&'static DriverFns` and never cleared, published happens-after `OnceLock` init.                                                                                                 |
| Panic-Across-FFI                         | —         | No Rust function is passed to a C callback. All FFI is C → Rust through return codes; panic-across-`extern "C"` is not reachable. `panic = "abort"` in the release profile removes unwinding entirely.                                                               |
| Command and Scripting Interpreter        | T1059     | No shell invocation. NVRTC compiles in-process via the loaded library's API; no `Command::new` anywhere in `src/`.                                                                                                                                                  |
| Data from Local System                   | T1005     | Reads only documented environment variables (see below) and writes the PTX cache to `temp_dir()/ironaccelerator/ptx/` with 64-bit hex-hash filenames (no user string in a filename), atomically via tmp+rename.                                                       |

### Environment-variable surface

The only environment variables the crate reads. All are documented and
bounded (search-dir or feature-toggle only; none is opened as a path
directly, so no traversal):

| Variable                     | Owner                      | Purpose                                          |
| ---------------------------- | -------------------------- | ------------------------------------------------ |
| `IRON_CUDA_LIBDIR`           | `ironaccelerator-cuda-sys` | Extra search dir for `nvcuda.dll` / `libcuda.so` |
| `IRON_ROCM_LIBDIR`           | `ironaccelerator-rocm-sys` | Same, for the AMD HIP runtime                    |
| `IRON_CUDA_INCLUDE`          | `ironaccelerator-cuda`     | Extra `-I` paths handed to NVRTC                 |
| `CUDA_PATH` / `CUDA_HOME`    | `ironaccelerator-cuda`     | Toolkit `include/` discovery for NVRTC           |
| `IRON_CUDA_PTX_CACHE`        | `ironaccelerator-cuda`     | Disable disk cache (`0` / `off` / `false`)       |
| `IRON_CUDA_PTX_CACHE_DIR`    | `ironaccelerator-cuda`     | Override PTX cache root                          |
| `NEURON_INSTANCE_TYPE`       | `ironaccelerator-neuron`   | AWS Neuron generation detection override         |
| `AWS_NEURON_VISIBLE_CORES`   | `ironaccelerator-neuron`   | Core enumeration                                 |

---

## 4. CMMC 2.0 Level 2 mapping

CMMC 2.0 Level 2 inherits the 110 controls from NIST SP 800-171 Rev 2.
Below are the families with meaningful contact points for an open-source
Rust library; a library is not a system that processes CUI, so most
families are **N/A**. Full CMMC certification applies to the *organization*
handling CUI (build/release infrastructure), not to this source.

### Access Control (AC) / Identification & Authentication (IA) — N/A
The crate has no authentication or access-control surface.

### Configuration Management (CM)

| Control     | Status | Evidence                                                                                                             |
| ----------- | ------ | ------------------------------------------------------------------------------------------------------------------- |
| CM.L2-3.4.1 | Pass   | `Cargo.lock` pinned at workspace root; workspace version (`2.0.0`) propagated through `[workspace.package]`.         |
| CM.L2-3.4.2 | Pass   | Release profile enforces `lto = "fat"`, `strip = "symbols"`, `panic = "abort"`, `codegen-units = 1`.                |
| CM.L2-3.4.6 | Pass   | Least-functionality: `rust-version = "1.89"`; each backend behind a feature flag; no default network/telemetry.     |
| CM.L2-3.4.9 | Pass   | **No build scripts** execute; no `vendored*` features compile native code outside the published source.             |

### Risk Assessment (RA)

| Control      | Status | Evidence                                                                                              |
| ------------ | ------ | ----------------------------------------------------------------------------------------------------- |
| RA.L2-3.11.1 | Pass   | This document.                                                                                         |
| RA.L2-3.11.2 | Pass   | `cargo audit` runs as a CI job on every push/PR (`.github/workflows/ci.yml` → `audit`).               |
| RA.L2-3.11.3 | Pass   | RustSec advisories tracked; the single open `paste` advisory is unmaintained-only, dispositioned §1.  |

### Security Assessment (CA)

| Control      | Status | Evidence                                                                                     |
| ------------ | ------ | -------------------------------------------------------------------------------------------- |
| CA.L2-3.12.1 | Pass   | CI runs `cargo clippy --workspace --all-targets` + `cargo test --workspace`.                 |
| CA.L2-3.12.3 | Pass   | Live-GPU validation suite gated behind an env flag; loader-probe + lib tests run in CI.       |
| CA.L2-3.12.4 | Pass   | This audit document.                                                                          |

### System and Communications Protection (SC)

| Control       | Status | Evidence                                                                                            |
| ------------- | ------ | --------------------------------------------------------------------------------------------------- |
| SC.L2-3.13.11 | N/A    | Crate implements no cryptography (§2); FIPS-validated crypto is the consumer's responsibility.       |

### System and Information Integrity (SI)

| Control      | Status | Evidence                                                                                                        |
| ------------ | ------ | -------------------------------------------------------------------------------------------------------------- |
| SI.L2-3.14.1 | Pass   | `SECURITY.md` defines the reporting channel; `CHANGELOG.md` records remediations.                              |
| SI.L2-3.14.2 | Pass   | `cargo clippy -- -D warnings` in CI; runtime safety via the borrow checker over a fully-attributed `unsafe` surface. |
| SI.L2-3.14.3 | Pass   | RustSec advisory scanning runs in CI on every push/PR.                                                          |
| SI.L2-3.14.5 | Pass   | `cargo audit` covers the analogous concern (known-vulnerable dependencies) for a source distribution.          |
| SI.L2-3.14.7 | Pass   | `git log` provides per-commit attribution; release tags are signed per the release runbook.                    |

### Incident Response (IR)

| Control     | Status | Evidence                                                              |
| ----------- | ------ | --------------------------------------------------------------------- |
| IR.L2-3.6.1 | Pass   | `SECURITY.md` publishes the reporting channel and SLA.                |
| IR.L2-3.6.2 | Pass   | Coordinated disclosure via private GitHub Security Advisory.          |

### Audit and Accountability (AU) — Partial / via host
Audit logging is the consuming application's responsibility; the crate
emits to `stderr` only on NVRTC compile errors and via optional `tracing`
integration.

### Maintenance / Media Protection / Personnel / Physical (MA/MP/PS/PE) — N/A for a library.

---

## 5. Private-data, PII, and secret scan

Tree-wide scans (tracked source only) for:

- Hard-coded credentials — passwords, API keys, bearer tokens, AWS
  access-key patterns, PEM/OpenSSH key blocks — **0 hits**. Specifically
  confirmed absent: the OTLP bearer token and crates.io token that appeared
  in development chat are **not** present in any source or published artifact.
- Personally identifying paths / developer-specific home directories —
  **0 hits in tracked source**. Author email appears once in `LICENSING.md`
  as the public licensing contact, intentionally.
- Internal hostnames or RFC 1918 IPs — **0 hits**.
- Tracked `.env`, key, or credential files — **0** (excluded by `.gitignore`).

The `/reference/` directory (vendored cudarc for comparison work) and
`.claude/` session data are local-only, excluded by `.gitignore`, and never
enter the source tree.

---

## 6. Recommendations (non-blocking)

1. **Generate an SBOM at release.** `cargo cyclonedx` or `cargo sbom`
   produces a CycloneDX / SPDX manifest from `Cargo.lock`. Attaching it to
   the GitHub Release enables downstream supply-chain attestation (T1195
   mitigation evidence). Not gating the 2.0.0 cut.

### Operational items (outside this repo)

- Rotate the OTLP bearer token and the crates.io token that appeared in
  development chat; treat both as compromised.
- Remove any persistent machine-global `CARGO_REGISTRY_TOKEN` environment
  variable on the release host — it shadows `cargo login` credentials.

The prior audit's two recommendations — add `cargo audit` to CI, and adopt
the reporting policy — are both **closed** as of 2.0.0.

---

## 7. Sign-off

The audit found no open vulnerabilities, no leaked private data, no embedded
secrets, no build-time or runtime network egress, and no cryptographic
misuse. All CMMC 2.0 Level 2 control gaps identified in the 1.2.0 audit are
closed. The crate is appropriate for publication to crates.io.

The single recommendation in §6 (SBOM at release) is tracked for the 2.0.x
maintenance window.
