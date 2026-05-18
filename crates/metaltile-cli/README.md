# metaltile-cli

MetalTile CLI — benchmark, test, and inspect GPU kernels.
The `tile` binary is the primary developer tool for the MetalTile project:
run performance benchmarks against MLX, compile kernels to inspect generated
MSL, profile GPU occupancy, and manage regression baselines.

This is a binary crate only — it has no library API. All functionality is
exposed through subcommands of the `tile` binary.

## Position in the pipeline

```
metaltile (facade) ──┐
metaltile-core       │
metaltile-codegen    ├──► metaltile-cli (this crate) ──► terminal / JSON / files
metaltile-runtime    │         tile binary
metaltile-std ───────┘
```

The CLI sits at the top of the stack, consuming every other crate.
It's the only crate in the workspace that exercises the full
compile→dispatch→measure loop end-to-end.

## Quick start

```sh
# Install the tile binary
cargo install --path crates/metaltile-cli

# Run the full benchmark suite (requires macOS + Metal)
tile bench

# Compile all kernels and report errors
tile build

# Inspect one kernel's IR and generated MSL
tile inspect --kernel mt_rms_norm

# Profile occupancy and register pressure
tile profile

# Show GPU device info
tile device

# Save current bench results as a baseline
tile snap -o baseline.json

# Compare current bench results to a saved baseline
tile diff baseline.json
```

Subcommand-specific help:

```sh
tile bench --help
tile build --help
```

## Crate contents

| Module | Purpose |
|---|---|
| `cmd` | Subcommand dispatch: `bench`, `build`, `inspect`, `profile`, `device`, `snap`, `diff` |
| `cmd::bench` | Full benchmark suite: MetalTile vs MLX reference kernels |
| `cmd::build` | Compile all kernels to MSL and report errors |
| `cmd::inspect` | Print IR and/or MSL for a single kernel |
| `cmd::profile` | Estimate GPU occupancy and register pressure per kernel |
| `cmd::device` | Show GPU device info and supported Metal features |
| `cmd::snap` | Save benchmark results as a JSON regression baseline |
| `cmd::diff` | Compare current benchmark results to a saved baseline |
| `runner` | GPU dispatch: compile MSL, allocate buffers, run kernels, measure GPU time |
| `measure` | Timing and throughput measurement helpers |
| `run_spec` | Wire a `BenchSpec` through the full compile→dispatch→measure pipeline |
| `kernel_utils` | Shared utilities for kernel mode detection and spec iteration |
| `stats` | `BenchStats` struct and throughput calculation |
| `term` | Terminal styling: colored output, bold text |

## API reference

### Subcommands

| Command | Purpose |
|---|---|
| `tile bench` | Run the full benchmark suite. MetalTile kernels run against MLX Metal kernel reference. Use `--filter <op>` to narrow. Outputs per-op throughput ratio and correctness. |
| `tile build` | Compile all registered kernels to MSL and report any errors. Use `--emit` to write `.metal` files and compile a `kernels.metallib`. |
| `tile inspect --kernel <name>` | Print the IR (SSA-form) and/or generated MSL for one kernel. Use `--ir` for IR only, `--msl` for MSL only. |
| `tile profile [kernel]` | Estimate GPU occupancy and register pressure. Without a kernel name, profiles all kernels. With `--sweep`, shows per-threadgroup-size breakdown. |
| `tile device` | Show GPU device info: name, Metal feature set, supported language version, max threadgroup size. |
| `tile snap -o <file>` | Save current benchmark results as a JSON regression baseline file. |
| `tile diff <file>` | Compare current benchmark results to a saved baseline. Reports regressions (throughput drops below threshold). |

### Installation

```sh
cargo install --path crates/metaltile-cli
```

The binary is named `tile`. After installation it's available on your `$PATH`.

This crate is not published to crates.io (`publish = false`). It's a
project-internal developer tool, not a library.

## Dependencies

### Internal

| Crate | Role in this crate |
|---|---|
| `metaltile` | Facade re-exports (macros, `Context`) used by bench/inspect |
| `metaltile-core` | IR types for kernel iteration and inspect output |
| `metaltile-codegen` | MSL generation for build, inspect, and bench dispatch |
| `metaltile-runtime` | GPU dispatch, PSO compilation, buffer management |
| `metaltile-std` | `BenchSpec` registry via `inventory`, op catalog, benchmark shapes |
| `inventory` | Collects all `#[bench_kernel]`-registered `BenchSpec`s at link time |

### External

| Crate | Role |
|---|---|
| `serde` / `serde_json` | Serialize/deserialize snap/diff baseline files |
| `objc2` / `objc2-metal` / `objc2-foundation` | Metal GPU API bindings (macOS only, cfg-gated) |

## MSRV / platform

**macOS is required** for GPU commands (`bench`, `profile`, `device`).
All Metal API calls are cfg-gated behind `target_os = "macos"`.
On other platforms these commands return errors or zero-stub output.

`build` and `inspect` work on any platform — they only need the
compiler crates, not the GPU runtime.

Rust: nightly (workspace-wide, edition 2024).

## Extending

- **New subcommand:** Create `src/cmd/<name>.rs` with `pub fn run(args: &[String])` and `pub fn help()` functions. Add `pub mod <name>;` to `src/cmd/mod.rs`. Add a match arm in `src/main.rs` for the subcommand name and its `--help` case. Add an entry to the usage text in `print_usage_and_exit`.

- **New global flag:** Add a helper function alongside `flag_val` / `flag_present` in `src/main.rs` if the flag follows a different convention. Otherwise parse it in the relevant subcommand's `run()` function.

- **New benchmark output format:** `src/cmd/bench.rs` — extend the output rendering or add a `--format` flag.

- **New runner capability:** `src/runner.rs` — the `GpuRunner` handles Metal dispatch, buffer allocation, and GPU timing. Add methods for new measurement primitives here.

- **Tests to update:** Integration tests in `src/cmd/`. Run `tile bench` on macOS to verify no regressions.

## Related documentation

- [Root README](../../README.md) — project overview and architecture
- [CONTRIBUTING](../../CONTRIBUTING.md) — dev setup, PR process, CI
- [`metaltile-std` README](../metaltile-std/README.md) — the `BenchSpec` registry and op catalog this CLI exercises
- [`metaltile-runtime` README](../metaltile-runtime/README.md) — the GPU dispatch layer used by `runner`

## License

Licensed under the [Apache License, Version 2.0](../../LICENSE).
