# CLI

`tile` is the command-line driver for benchmarking, building, and inspecting kernels. Install it, or run it through `cargo` from a checkout.

```bash
cargo install --path crates/metaltile-cli      # installs the `tile` binary
# or, from a checkout, without installing:
cargo run -p metaltile-cli -- <command> …
```

`make bench` wraps `tile bench`; for the other subcommands run `tile` (or the `cargo run` form) directly.

## `tile bench` — benchmark vs MLX

Runs every kernel against its MLX reference and reports throughput + a correctness check.

```
tile bench [-f <substr>] [-v|-vv] [-o <file.json>] [--allow-dirty]
           [--diff] [--baseline-ref <git-ref>]
```

| Flag | Effect |
|---|---|
| `-f, --filter <substr>` | only run kernels whose name contains `<substr>` |
| `-v` / `-vv` | `-v` adds occupancy + register profile; `-vv` adds GPU timing (min µs + bandwidth) |
| `-o, --json <file>` | also write results as JSON |
| `--allow-dirty` | run on a dirty working tree (default: refuses, so numbers tie to a clean SHA) |
| `--diff` | opt into the post-bench diff against the target-branch baseline |
| `--baseline-ref <ref>` | git ref whose `baselines/<chip>.json` to diff against (default: first of `origin/dev`, `upstream/dev`, `dev`) |

## `tile build` — compile kernels to MSL

Compiles every kernel and reports errors; with `--emit`, writes artifacts.

```
tile build [-f <substr>] [--dtypes f32,f16,bf16] [-v]
           [--emit msl,metallib,swift,ir,all] [-o <dir>] [--sdk <sdk>] [-t]
```

| Flag | Effect |
|---|---|
| `-f, --filter <substr>` | only build matching kernels |
| `--dtypes <list>` | comma-separated dtypes to build (`f32,f16,bf16`) |
| `-v` | print the generated MSL for each kernel |
| `--emit <list>` | emit artifacts — `msl`, `metallib`, `swift`, `ir`, or `all` |
| `-o, --out <dir>` | output directory (required when `--emit` is set) |
| `--sdk <sdk>` | `xcrun` SDK for the Metal toolchain (default: `macosx`) |
| `-t, --time-passes` | run the pass pipeline 25× per kernel, print per-pass median wall time instead of emitting |

Codegen smoke check — emit everything and confirm `xcrun metal` accepts it: `tile build --emit all -o /tmp/mt-smoke`.

## `tile emit` — emit a Swift-consumable kernel package

Generates a `kernels.metallib` + per-kernel `.metal` sources + `MetalTileKernels.swift` dispatch wrappers under `<out>/`:

```
tile emit --out <swift-package-dir> [--sdk macosx] [--no-compile]
```

| Flag | Effect |
|---|---|
| `--out <dir>` | output root (required); artifacts land in `<dir>/Resources/` and `<dir>/Generated/` |
| `--sdk <sdk>` | `xcrun` SDK for the Metal toolchain (default: `macosx`) |
| `--no-compile` | skip the `xcrun metal` / `metallib` step (still writes `.metal` + manifest + Swift) |

Output layout matches a SwiftPM `Sources/<Target>/` convention:

```
<out>/Resources/kernels/<name>.metal
<out>/Resources/kernels.metallib
<out>/Resources/manifest.json
<out>/Generated/MetalTileKernels.swift
```

## `tile inspect` — IR and MSL for one kernel

```
tile inspect [<kernel>] [--filter <substr>] [--all] [--ir] [--stats]
             [--pass <name>] [--dtype <f32|f16|bf16|i32|u32>] [-o <dir>]
```

| Flag | Effect |
|---|---|
| *(no flag)* | print the final generated MSL |
| `--ir` | print the raw IR before any passes |
| `--pass <name>` | print the IR after a specific pass (`--pass all` for every stage) |
| `--stats` | print the per-pass op-count reduction table |
| `--dtype <d>` | dtype override for monomorphisation |
| `--filter <substr>` / `--all` | inspect many kernels at once |
| `-o, --dir <dir>` | write output files instead of printing to stdout |

Omit the kernel name to list every registered kernel. See [Developing → debugging a kernel](developing.md#debugging-a-kernel).

## `tile device` — GPU info

Prints the Metal device name, Metal version, Apple GPU family, and the supported feature flags (native `bfloat`, simdgroup matrix, etc.). Add `--json` for machine-readable output.

## `tile snap` — save a perf regression baseline

```
tile snap [-o <file>] [--from <file.json>] [--note <text>] [-f <substr>]
```

| Flag | Effect |
|---|---|
| `-o, --out <file>` | write the snapshot here (default: `.tile-snapshots/<sha>.json`) |
| `--from <file.json>` | promote an existing bench JSON instead of re-running the bench |
| `--note <text>` | attach a note to the snapshot |
| `-f, --filter <substr>` | only include kernels whose name contains `<substr>` |

## `tile diff` — compare against a baseline

```
tile diff <baseline> [<current>] [-f <substr>] [--threshold <pct>]
          [--sort name|delta|pct] [--only-regressions] [--only-improvements]
```

`<baseline>` is a saved snapshot JSON; `<current>` is an optional bench JSON — omit it and `diff` runs the bench itself.

| Flag | Effect |
|---|---|
| `-f, --filter <substr>` | only show kernels whose name contains `<substr>` |
| `--threshold <pct>` | highlight regressions larger than this percentage (default: `5`) |
| `--sort <key>` | sort rows by `name`, `delta`, or `pct` (default: `name`) |
| `--only-regressions` | show only regressed kernels |
| `--only-improvements` | show only improved kernels |
