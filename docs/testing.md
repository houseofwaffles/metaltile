# Testing

How kernels and codegen are verified, what runs where, how to write a test, and — importantly — the **gaps** in the test infrastructure that let bugs through silently.

## The test layers

Correctness is checked at four layers, each catching what the layer above cannot:

| Layer | Catches | Where it lives | Runs in CI? |
|---|---|---|---|
| **DSL / codegen unit tests** | Pass correctness, body-parser arms, IR variants, emit paths; `trybuild` compile-fail fixtures | `crates/metaltile-codegen`, `metaltile-core`, `metaltile-macros` | ✅ |
| **MSL snapshots** (`insta`) | Codegen output drift — a reviewable text diff in the PR | `crates/metaltile-codegen/tests/msl_snapshots.rs` | ✅ |
| **GPU correctness** | Numeric disagreement vs a naive CPU oracle, on a real Metal device | `crates/metaltile-std/tests/<kernel>_gpu_correctness.rs` | ✅ (macOS runner) |
| **MLX side-by-side** (bench) | Throughput + numeric parity vs the upstream MLX kernel | `tile bench` | local-only (needs an MLX checkout) |

No single layer is sufficient. The unit tests never touch a GPU; snapshots pin *whatever* the codegen emits (including wrong output); `xcrun metal` only checks syntax. **GPU correctness tests are the floor** — see the gaps section below.

## Running tests — what runs where

```bash
make test        # whole workspace: codegen, runtime, GPU correctness (GPU on a Mac)
make clippy      # lint, -D warnings
make fmt-check   # formatting
make typos       # spell-check
make coverage    # HTML coverage report (needs cargo-llvm-cov)
make bench       # MLX side-by-side benchmark suite (macOS + Metal only)
```

Per-kernel, via `cargo` directly (these are the documented exceptions to "always use `make`"):

```bash
# One kernel's GPU correctness test:
cargo test -p metaltile-std --test <kernel>_gpu_correctness

# One kernel's perf bench (the #[ignore]'d companion test):
cargo test --release -p metaltile-std --test <kernel>_gpu_correctness -- --ignored --nocapture
```

### CI vs local

| Job | Workflow | What it runs |
|---|---|---|
| `typos` / `clippy` / tests | `.github/workflows/check.yml` | spell-check, lint `-D warnings`, `cargo test --workspace` |
| coverage | `.github/workflows/coverage.yml` | `cargo llvm-cov --workspace --codecov` on macOS, uploads to Codecov; runs on pushes touching `crates/`, `Cargo.*`, `rust-toolchain.toml`, `.github/configs/codecov.yml` |
| PR title | `.github/workflows/pr.yml` | validates the conventional-commit format |
| labels | `.github/workflows/auto-label.yml` | release-notes labels from the PR-title prefix |

- The DSL / codegen / GPU-correctness layers all run in CI — including on a macOS runner with a real GPU.
- **`tile bench` (MLX side-by-side) is local-only** — it needs an MLX checkout the CI runners don't have. If a kernel has no MLX counterpart, MLX side-by-side does not apply; rely on the other three layers.

## Writing tests

### Every non-trivial kernel ships a GPU correctness test — same commit

The test runs the kernel on a real Metal device and compares against a naive CPU reference computed in `f32`. Shared helpers (`ramp`, dtype pack/unpack, `max_abs_diff`, `naive_*`) live in `crates/metaltile-std/tests/common/mod.rs`.

```rust
#![cfg(target_os = "macos")]
mod common;
use common::{ramp, pack_bytes, unpack_bytes, max_abs_diff};
use metaltile_runtime::Context;

#[test]
fn my_kernel_matches_naive_cpu_reference_f32() {
    // 1. Build small synthetic inputs (ramp / deterministic pattern).
    // 2. Compute a naive CPU reference in f32.
    // 3. Pack to bytes, populate the buffer map, dispatch via
    //    Context::dispatch_with_grid(&kernel, &buffers, &constexprs,
    //                                 grid_xyz, threadgroup_xyz).
    // 4. Unpack the output buffer; assert max_abs_diff < 1e-4.
}

#[test]
#[ignore = "perf bench, run via --ignored --nocapture"]
fn my_kernel_perf_bench_f32() {
    // 20 warmup + 100 measure iterations; report median GPU µs + GB/s.
}
```

The naive CPU reference **is the contract**. If kernel and reference disagree, decide which is wrong before merging — don't loosen the tolerance to pass.

### MSL snapshots for new emit paths

A new DSL primitive, fusion pattern, or dtype path also lands an `insta` fixture in `crates/metaltile-codegen/tests/msl_snapshots.rs` — a hand-built kernel run through `MslGenerator`, with the full MSL pinned via `assert_snapshot!`. Any future codegen change then surfaces as a reviewable text diff. Refresh intentional changes with `cargo insta review` (interactive) or `cargo insta test --accept`.

Fixtures exist to **exercise distinct emit paths**, not to be exhaustive — add one when a new path lands that the existing snapshots don't cover.

## Coverage

`make coverage` (or `./.github/scripts/coverage.sh`) produces an HTML report at `target/llvm-cov/html/index.html`; `./.github/scripts/coverage.sh summary` prints the per-file table CI emits. Per-crate floors live in `.github/configs/codecov.yml`:

| Crate | Floor |
|---|---|
| `metaltile-macros` | 92% |
| `metaltile-codegen` / `metaltile-core` | 90% |
| `metaltile-runtime` | 85% |
| `metaltile-cli` | 80% |
| `metaltile-std` | line-coverage exempt — gated by bench-correctness instead |
| `metaltile` (facade) | excluded |

`metaltile-std`'s `ffai/` and `mlx/` kernel-body files are excluded from the line-coverage denominator: the `#[kernel]` proc-macro consumes the body at compile time, the Rust body never executes, so line coverage on them is structurally meaningless. **Their correctness is gated by GPU correctness tests and bench equivalence instead — not by line coverage.**

## ⚠️ Gaps in the test infrastructure

These are the holes a bug can slip through. Know them; close them when you can.

### ⚠️ A wrong kernel can pass every check except a GPU correctness test

A kernel that emits an **empty body** — from an inner `macro_rules!` or from a codegen pass dropping a loop body (see [Developing → kernel-authoring hazards](developing.md#kernel-authoring-hazards)) — produces all-zeros output. That output:

- **passes `xcrun metal`** — an empty body is valid MSL;
- **passes `tile build --emit` smoke** — same reason;
- **passes MSL-snapshot drift checks** — the snapshot just pins the wrong-but-stable empty body;
- **passes a loose integration test** if its tolerance absorbs the noise.

It fails **only** when actual GPU output is compared to an expected value. That is the GPU correctness test, and nothing else. This is exactly how a family of quantized-gather kernels shipped silently broken until a correctness test was added. **Do not rely on the smoke build or snapshots to catch a broken kernel.**

### ⚠️ Not every kernel has a GPU correctness test yet

Coverage of `crates/metaltile-std/tests/` is incomplete — some kernels have a bench row but no correctness test, and some have neither. A kernel with no correctness test has *no automated proof it computes the right answer*. When you touch such a kernel, add the test; when you add a kernel, add it in the same commit.

### ⚠️ Perf numbers can be harness artifacts

A bench number is only meaningful if the harness measures the kernel and not its own overhead. A latency that doesn't scale with input size is the tell. See [Developing → kernel-authoring hazards](developing.md#kernel-authoring-hazards) ("too flat to be physical") for the resident-buffer and GPU-clock-warmup fixes.
