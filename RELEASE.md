# Release runbook

How to cut and publish a new IronAccelerator release.

## Pre-flight (local)

From a clean checkout of `master`:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --release -- -D warnings
cargo test --workspace --release
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

All four must pass. The matching CI workflow in `.github/workflows/ci.yml`
runs the same set.

If a CUDA-capable machine is available, also run the live-GPU suite and
the cudarc benchmark:

```bash
cargo test -p ironaccelerator-cuda --release -- --include-ignored
cargo bench -p ironaccelerator-cuda --bench vs_cudarc
cargo run --release -p ironaccelerator-cuda --example saxpy_cudarc_style
```

The cudarc bench numbers go in the release notes; if there's a regression
(>10% on alloc/free or stream sync vs the prior release), root-cause it
before tagging.

## Version bump

1. Bump the workspace version in `/Cargo.toml` (`[workspace.package] version`).
   The path-deps in `[workspace.dependencies]` carry `version = "..."` strings
   that must match.
2. Move the `[Unreleased]` section in `CHANGELOG.md` to the new version
   heading with today's date.
3. Re-add an empty `[Unreleased]` section above it.
4. Commit: `git commit -m "release: vX.Y.Z"`.

## Tag + push

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin master --tags
```

## Publish to crates.io (in dependency order)

The crates publish bottom-up; each one waits ~30s for the registry to
index its newly-published deps before the next builds.

```bash
cargo publish -p ironaccelerator-core
cargo publish -p ironaccelerator-cuda-sys
cargo publish -p ironaccelerator-cuda
cargo publish -p ironaccelerator-rocm-sys
cargo publish -p ironaccelerator-rocm
cargo publish -p ironaccelerator-qnn-sys
cargo publish -p ironaccelerator-qnn
cargo publish -p ironaccelerator-metal
cargo publish -p ironaccelerator-vulkan
cargo publish -p ironaccelerator-opengl
cargo publish -p ironaccelerator-dx12
cargo publish -p ironaccelerator-webgpu
cargo publish -p ironaccelerator-tpu
cargo publish -p ironaccelerator-levelzero
cargo publish -p ironaccelerator-neuron
cargo publish -p ironaccelerator             # umbrella crate goes last
```

Use `cargo publish --dry-run -p <crate>` first if you want to inspect the
exact `.crate` contents. Skip with `--no-verify` only when the verify step
trips over deps that aren't yet on the registry (the in-process publishes
above resolve themselves; this is rare).

## Post-publish

1. Open the GitHub release for `vX.Y.Z` and paste the new CHANGELOG section.
2. Attach the cudarc bench numbers from the pre-flight run.
3. Update any external docs / dependent repos.

## Yanking

If a release has a critical regression:

```bash
cargo yank --version X.Y.Z ironaccelerator-cuda
# repeat per affected crate
```

Then cut a `vX.Y.Z+1` with the fix and follow this runbook from the top.
