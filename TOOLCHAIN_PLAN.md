# MetalTile Toolchain Plan

> Working document — edit freely.

---

## Status Quo

### What exists

| Binary | Invocation | What it does |
|--------|-----------|--------------|
| `bench_suite` | `cargo run -p metaltile-bench --bin bench_suite` | perf + correctness table |
| `dump_msl` | `cargo run -p metaltile-bench --bin dump_msl` | print generated MSL |

Both binaries live inside `metaltile-bench`. No unified entry point. No correctness-only mode. No regression tracking. No scaffolding.

### Pain points

**`dump_msl` has a hardcoded kernel list** (`dump_msl.rs:43–113`). Every time a new op is added, someone has to remember to add it here. It will drift.

**No `tile test`**. The only way to check correctness is to run the full bench suite, which also runs perf measurement (SLC flush, multiple timing iterations). A correctness-only pass would be 10–20× faster and runnable in CI without caring about timing noise.

**No regression tracking**. There is `--json` output but nothing that compares against a known-good baseline. Perf regressions are caught by eye.

**`cargo run -p metaltile-bench --bin ...` is not a UX**. Every tool invocation requires knowing the internal crate structure. A unified `tile` binary fixes this.

**Op files live in `metaltile-bench`** which is `publish = false`. If someone wants to depend on MetalTile kernels as a library they have no crate to import.

### What is actually clean today

The op files (`ops/unary.rs`, `ops/softmax.rs`, etc.) are already well-separated from the bench harness. Each file is:
- `#[bench_kernel(...)]` metadata
- `#[kernel]` DSL implementation
- No direct `GpuRunner` calls

The harness lives entirely in `spec.rs` + `shared.rs`. This separation should be preserved and formalized.

---

## Proposed Crate Layout

```
crates/
  metaltile-core/     IR types                          (unchanged)
  metaltile-macros/   #[kernel], #[bench_kernel]        (unchanged)
  metaltile-codegen/  MSL lowering + pass pipeline      (unchanged)
  metaltile-interp/   CPU reference interpreter         (unchanged)
  metaltile-runtime/  Metal GPU dispatch                (unchanged)
  metaltile/          re-export facade                  (unchanged)
  metaltile-std/      NEW: kernel stdlib + bench metadata
  metaltile-cli/      NEW: CLI binary (command: tile)
```

`metaltile-bench` is eliminated. Its contents split between `metaltile-std` and `metaltile-cli`.

### Why `metaltile-bench` can be eliminated

`metaltile-bench` currently has two distinct responsibilities:

1. **Bench metadata** — `BenchSpec`, `BenchDispatch`, `ShapeSpec`, `Dim`, `BufInit`, `DispatchGrid`, `MlxArg`, the `inventory::collect!` call. This is pure data with no GPU dependencies. It must be in a crate that both `metaltile-std` (which `submit!`s specs via `#[bench_kernel]`) and the CLI (which iterates them) can see.

2. **Bench runner** — `GpuRunner`, `bench_gbps`, `check_equiv`, `OpResult`, `SuitePrinter`, `BenchSpec::run()`, `stats.rs`, `term.rs`. This is measurement infrastructure with Metal/objc dependencies.

The cycle constraint is: `metaltile-std` must depend on whatever defines `BenchSpec` (to `submit!` into inventory), and `metaltile-cli` must depend on `metaltile-std`. So `BenchSpec` cannot live in `metaltile-cli`.

The solution: **move bench metadata into `metaltile-std`, move the runner into `metaltile-cli`**. `BenchSpec::run(&GpuRunner)` becomes a free function or extension trait in `metaltile-cli`, not a method on the type. `metaltile-bench` disappears.

### `metaltile-std` — kernel stdlib

All `#[kernel]` + `#[bench_kernel]` DSL op files (unary, binary, softmax, …), plus the bench metadata types that `#[bench_kernel]` generates submissions for:

- `ops/` — all kernel DSL files (`unary.rs`, `softmax.rs`, …)
- `steel/` — matmul family
- `bench/` — `BenchSpec`, `BenchDispatch`, `ShapeSpec`, `Dim`, `BufInit`, `DispatchGrid`, `MlxArg`, `ScalarBufSpec`, `TensorBufSpec`, bytes formula helpers, `inventory::collect!(BenchSpec)`

Does **not** contain:
- `GpuRunner`, `GpuBuffer`, `CompiledKernel`, any Metal/objc imports
- `bench_gbps`, `check_equiv`, `SuitePrinter`, `OpResult`, `BenchSpec::run()`
- Any timing or measurement code

Dependencies: `metaltile`, `metaltile-core`, `metaltile-macros`, `metaltile-codegen`, `metaltile-interp`, `inventory`

`publish = true` — this is the public kernel library.

### `metaltile-cli` — CLI binary (command: `tile`)

Single binary, all subcommand logic. Contains everything that was in `metaltile-bench` except the metadata types:

- `runner.rs` — `GpuRunner`, `GpuBuffer`, `CompiledKernel`, `flush_slc`
- `measure.rs` — `bench_gbps`, `BenchStats`, timing infrastructure
- `check.rs` — `check_equiv`, `EquivResult`, correctness helpers
- `report.rs` — `OpResult`, `SuitePrinter`, `term.rs` terminal formatting
- `run_spec.rs` — `fn run_spec(spec: &BenchSpec, runner: &GpuRunner, dt: DType) -> Vec<OpResult>` (the former `BenchSpec::run` and all its dispatch arms)
- `cmd/` — one file per subcommand (`bench.rs`, `test.rs`, `build.rs`, `inspect.rs`, `device.rs`, `snap.rs`, `diff.rs`)

`publish = false`

---

## The `tile` CLI

### Top-level interface

```
tile <subcommand> [options]

Subcommands:
  build     Compile all kernels to MSL and report errors (no GPU)
  test      Run correctness checks: interpreter ↔ GPU
  bench     Benchmark suite: MetalTile vs MLX reference
  inspect   Print IR and/or MSL for one kernel
  device    Show GPU device info and supported feature flags
  snap      Save current bench results as a regression baseline
  diff      Compare a bench run to a saved baseline
  trace     Generate a Metal GPU capture (.gputrace) for Instruments
  profile   Run with GPU hardware counters (occupancy, ALU, bandwidth)
  fuzz      Run with adversarial inputs to find correctness edge cases

Global flags (all subcommands):
  --color [always|never|auto]   force or suppress ANSI color (default: auto)
  -q, --quiet                   suppress banners and headers, only emit results/errors
  -v, -vv                       verbosity; -v shows extra detail, -vv shows everything
```

---

### `tile build`

**Purpose**: verify every registered kernel compiles to valid MSL. No GPU required. Fast enough for pre-push CI.

Iterates `inventory::iter::<BenchSpec>` (same auto-discovery as `bench`). For each spec, calls the `kernel_ir()` accessor and runs `MslGenerator::generate()`. Reports success or error per kernel × dtype variant.

```
$ tile build
mt_exp          f32/f16/bf16   ok
mt_log          f32/f16/bf16   ok
mt_my_new_op    f32            ERROR: undefined variable v_42
                               at msl.rs:emit_block — Op::Load src not found

14 ok, 1 error
```

Flags:
- `-f, --filter <pattern>` — only build matching kernels
- `--dtypes <f32,f16,bf16>` — restrict which dtype variants to build (default: all registered for each kernel)
- `-o, --out <path>` — write every generated `.metal` file to a directory (useful for auditing or external tooling)
- `-v` — print generated MSL for every kernel, even on success
- `--metal-compile` — also invoke `metal`/`metallib` to catch GPU-level type errors (requires macOS + Xcode, slow, Phase 4)

---

### `tile test`

**Purpose**: correctness only. Runs `check_equiv` (interpreter ↔ GPU) with small shapes. No perf measurement, no SLC flush, no timing iterations. Roughly 20× faster than full bench for a single op.

```
$ tile test
mt_exp     f32   ok   max_err=0.000001
mt_exp     f16   ok   max_err=0.000813
mt_exp     bf16  ok   max_err=0.007812
mt_rms_norm f32  ok   max_err=0.000002
...
154 passed, 0 failed

$ tile test --filter rms_norm
mt_rms_norm f32   ok   max_err=0.000002
mt_rms_norm f16   ok   max_err=0.000489
mt_rms_norm bf16  ok   max_err=0.007812
```

Flags:
- `-f, --filter <pattern>` — only test matching kernels
- `--dtypes <f32,f16,bf16>` — only test specific dtype variants
- `--ref [interp|mlx]` — reference to compare against (default: `interp`; use `mlx` for ops where the interpreter stub is incomplete)
- `--check-n <n>` — override the element count used for correctness shapes (default: class-specific, e.g. 2048 for Unary)
- `--json` — emit per-kernel pass/fail + max_err as JSON to stdout
- `-o, --out <path>` — write JSON results to a file instead of stdout (implies `--json`)
- `--fail-fast` — stop on first failure
- `-v` — on failure, print actual vs expected values side-by-side
- `-vv` — also print passing kernel outputs

Relationship to `cargo test`: `cargo test` runs Rust unit tests (pure CPU, no Metal). `tile test` runs GPU correctness checks. They are complementary, not overlapping. CI should run both.

---

### `tile bench`

**Purpose**: replaces `bench_suite`. Identical output format.

```
$ tile bench
$ tile bench -f softmax
$ tile bench -f softmax --json -o results/run.json
$ tile bench --no-ref             # MT throughput only, no MLX reference
$ tile bench --no-slc-flush       # faster but noisier
$ tile bench --dtypes f32         # only f32 variants
$ tile bench --sort pct           # sort table by MT% ascending (find worst first)
$ tile bench --snap .tile-snapshots/latest.json   # bench + save snapshot in one step
```

Flags:
- `-f, --filter <pattern>` — only bench matching kernels
- `--dtypes <f32,f16,bf16>` — restrict dtype variants
- `--json` — emit full results as JSON to stdout
- `-o, --out <path>` — write JSON results to a file (implies `--json`)
- `--snap <path>` — save a snapshot after bench completes (shorthand for piping to `tile snap --from`)
- `--no-ref` — skip MLX reference kernel runs (graceful degradation when MLX metallib absent)
- `--no-correct` — skip correctness checks, measure throughput only (faster)
- `--no-slc-flush` — skip SLC cache eviction before each measurement (faster but introduces cache-residency variance)
- `--sort [name|pct|gbps]` — sort output table (default: `name`; `pct` useful for finding worst performers)
- `--warmup <n>` — warmup iteration count (default: 3)
- `--iters <n>` — timing iteration count (default: 10)
- `--min-ms <n>` — run each kernel for at least N ms total, overrides `--iters` (useful for noisy machines)

---

### `tile inspect`

**Purpose**: introspect a kernel — print IR at any pass stage, or the final MSL. Replaces `dump_msl` and adds IR-level visibility.

**Auto-discovery**: uses inventory, no hardcoded list. Any `#[bench_kernel]`-annotated kernel is available by name.

```
$ tile inspect                              # list all registered kernel names
$ tile inspect rms_norm                     # print final MSL (default)
$ tile inspect rms_norm --ir                # print raw IR (before any passes)
$ tile inspect rms_norm --pass fusion       # print IR after FusionPass
$ tile inspect rms_norm --pass all          # print IR after each pass, then final MSL
$ tile inspect rms_norm --mode reduction    # override KernelMode
$ tile inspect rms_norm --dtype bf16        # use bf16 specialization
$ tile inspect rms_norm -o /tmp/out         # write .metal file to a directory
$ tile inspect rms_norm --stats             # print op/value counts per pass stage
$ tile inspect --all -o /tmp/out            # dump every registered kernel to directory
```

Pass names: `type_check`, `const_fold`, `tile_lowering`, `fusion`, `schedule`, `vectorize`, `all`

For `--pass all`, output format:
```
// ── BEFORE PASSES ───────────────────────────
[IR dump]

// ── AFTER type_check ────────────────────────
[IR dump]

// ── AFTER const_fold ────────────────────────
[IR dump]
...

// ── FINAL MSL ───────────────────────────────
[MSL source]
```

Flags:
- `--ir` — print raw IR before any passes
- `--pass <name>` — print IR after the named pass (or `all` for every stage)
- `--mode [elementwise|reduction|tile2d|grid3d]` — override the kernel's `KernelMode`
- `--dtype [f32|f16|bf16]` — select dtype specialization (default: f32)
- `-o, --out <path>` — write output to `.metal` files in a directory instead of stdout
- `--all` — dump every registered kernel (combine with `-o` to write all to disk)
- `--stats` — after each pass stage, print: op count, value count, fused groups, vectorized ops

**IR pretty-printer**: a structured display of `Kernel` and `Block` (included in Phase 2). Lives in `metaltile-core` as a `Display` impl or a dedicated `IrPrinter`. Format: one op per line, indented by nesting level, showing `ValueId`, `Op` variant, and resolved types. Example:

```
kernel mt_rms_norm  mode=Reduction  params=[inp:Tensor<T>, out:Tensor<T>, n:u32]
  block:
    v0  = ProgramId(axis=0)
    v1  = BinOp(Mul, v0, n)
    v2  = Reduce(Sum, v1..v1+n)
    v3  = BinOp(Div, v2, n)
    ...
```

`--stats` output example:
```
  after fusion:    ops=14 → 9  (5 fused)  values=18 → 13
  after vectorize: ops=9  → 7  (2 vectorized, width=4)
```

---

### `tile device`

**Purpose**: show the active Metal device and its feature flags. Useful for diagnosing why a kernel behaves differently on different machines.

```
$ tile device
Device:          Apple M4 Max
GPU family:      Apple9 (M3+)
Recommended MT settings:
  native_bfloat  yes   (Metal 3.1+ bfloat type)
  simdgroup_hw   yes   (simdgroup matrix multiply)
  async_copy     yes   (async threadgroup copy)
Threadgroup mem: 32 KB
Max TPG:         1024
SLC size:        ~64 MB  (estimated from flush timing)

$ tile device --json   # machine-readable output for CI env capture
```

Flags:
- `--json` — emit as JSON to stdout (useful for recording the test environment alongside bench results)
- `--slc-measure` — explicitly run the SLC size probe (writes 128 MB scratch, ~200 ms; skipped by default)

---

### `tile snap`

**Purpose**: save current bench results as a named baseline for regression tracking.

```
$ tile snap                                                      # run bench then save to default path
$ tile snap -o .tile-snapshots/$(git rev-parse --short HEAD).json
$ tile snap --from results/run.json                              # promote existing bench JSON to snapshot
$ tile snap --from results/run.json --note "after fusion fix"
$ tile snap -f softmax -o .tile-snapshots/softmax-baseline.json
```

Flags:
- `-o, --out <path>` — where to write the snapshot (default: `.tile-snapshots/<device>-<date>.json`)
- `--from <path>` — use an existing bench JSON instead of running bench (avoids a redundant bench run in CI)
- `-f, --filter <pattern>` — only include matching kernels
- `--note <text>` — free-text annotation stored in the snapshot (e.g. `"after fusion rewrite"`)

Snapshot file format (extends existing `--json` format):
```json
{
  "device": "Apple M4 Max",
  "gpu_family": "Apple9",
  "git_sha": "abc1234",
  "timestamp": "2026-05-16T12:00:00Z",
  "note": "after fusion fix",
  "results": [
    { "op": "mt_exp", "dtype": "f32", "shape": "N=67108864",
      "ref_gbps": 1847.2, "mt_gbps": 1851.0, "pct": 100.2,
      "correct": true }
  ]
}
```

---

### `tile diff`

**Purpose**: compare a bench run to a saved baseline. Surfaces regressions. Intended for CI.

```
$ tile diff .tile-snapshots/m4max.json                    # run bench then diff
$ tile diff .tile-snapshots/m4max.json run.json           # diff two existing JSON files (no bench run)
$ tile diff .tile-snapshots/m4max.json -f softmax
$ tile diff .tile-snapshots/m4max.json --threshold 3 --sort regression
$ tile diff .tile-snapshots/m4max.json --only-regressions # only print regressions
```

`baseline` is a required positional argument. `current` is an optional second positional — if omitted, bench is run fresh.

Output:
```
mt_rms_norm  f32   was 104%  →  97%   ▼  -7%  [REGRESSION]
mt_softmax   f16   was  98%  → 101%   ▲  +3%  [ok]
mt_exp       f32   was 100%  → 100%   —   0%  [ok]

1 regression (threshold: 5%), 2 improved, 151 unchanged
Exit code: 1
```

Flags:
- `-f, --filter <pattern>` — only compare matching kernels
- `--threshold <pct>` — regression threshold in percent (default: 5); exit 1 if any regression exceeds this
- `--sort [name|delta|regression]` — sort output (default: `name`; `regression` puts worst regressions first)
- `--only-regressions` — hide ok/improved rows
- `--only-improvements` — hide ok/regressed rows
- `--no-correct` — ignore correctness changes, only compare throughput

For CI: `tile bench --json -o /tmp/current.json && tile diff .tile-snapshots/main.json /tmp/current.json`

---

## Terminal Design Language

### Philosophy

Every color and weight carries meaning. No decoration. Secondary information is dimmed, not hidden. Semantic grouping via whitespace rather than boxes. Dense but scannable.

---

### Color palette

| Token | ANSI | Semantic use |
|-------|------|-------------|
| `BrightWhite` | 97 | Primary values — kernel names, numbers, file paths, MSL source |
| `Cyan` | 36 | Structure — op column, section headers, saved-to arrows |
| `Green` | 32 | Success — pass, ≥ 90% MT%, improvements in diff |
| `Yellow` | 33 | Warning — NYI, 60–89% MT%, `!` notices |
| `Red` | 31 | Failure — error, < 60% MT%, regressions, correctness fail |
| `Blue` | 34 | Type tags — `f32`, `f16`, `bf16` inline in output |
| `BrightBlack` | 90 | Noise reduction — separators, dimmed repeated values, timestamps, secondary labels |

Bold amplifies any color. Dim further mutes BrightBlack for separators.

---

### Symbol set

| Symbol | Meaning | ASCII fallback |
|--------|---------|----------------|
| `✓` | pass / ok | `ok` |
| `✗` | fail / error | `FAIL` |
| `~` | NYI / not implemented | `NYI` |
| `!` | warning | `!` |
| `→` | output to / saved as | `->` |
| `▼` | regression / decrease | `-` |
| `▲` | improvement / increase | `+` |
| `—` | no data / no change | `--` |
| `│` | column separator | `\|` |
| `─` | horizontal rule | `-` |

Unicode symbols are on by default and respect the same `--color` flag. Symbols are suppressed when color is disabled (`--color never` or non-TTY) so plaintext output is clean.

---

### Status verbs

Cargo-style: right-padded bold verb followed by the subject. Verb column is 12 chars wide, indented 2 spaces. Used for one-per-event lines outside of tables.

```
  Building   mt_exp f32                ← tile build, per kernel
  Testing    mt_rms_norm f32           ← tile test, per kernel
  Benching   softmax f32               ← tile bench, per kernel
  Inspecting rms_norm                  ← tile inspect
  Saved    → .tile-snapshots/m4max.json  (154 results)
  Warning  ! no MLX reference for mt_custom
  Error    ✗ mt_my_op f32: undefined variable v_42
```

Verb colors: `Building`/`Testing`/`Benching`/`Inspecting` — Cyan bold. `Saved` — Green bold. `Warning` — Yellow bold. `Error` — Red bold.

---

### Table format (`tile bench`)

Two-space left indent throughout. Column separator `│` BrightBlack dim. Horizontal rules BrightBlack dim. Blank line between top-level op groups. Repeated shape bases dimmed when only the dtype changes.

```
  Op                           │ Shape                      │   Reference │  MetalTile  │   MT% │ Correct
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────
  exp                          │         N=64M f32          │  1847.1 GB/s│ 1851.0 GB/s │  100% │ ✓
                               │                      f16   │   924.1 GB/s│  928.3 GB/s │  100% │ ✓
                               │                      bf16  │   924.1 GB/s│  928.3 GB/s │  100% │ ✓

  rms_norm                     │    B=1024 N=4096 f32       │   412.3 GB/s│  398.1 GB/s │   97% │ ✓ 1.2e-5
                               │                      f16   │   412.3 GB/s│  398.1 GB/s │   97% │ ✓
                               │                      bf16  │   412.3 GB/s│  330.1 GB/s │   80% │ ✓

  sort                         │         N=64M f32          │   124.1 GB/s│          —  │    ~ │ —
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────
```

Column styling rules:
- **Op**: Cyan bold, left-aligned, 28 chars. Blank on repeated rows within same subop.
- **Shape**: BrightWhite. When dtype repeats, base is BrightBlack dim, dtype is BrightWhite.
- **Reference**: BrightWhite right-aligned 14 chars. `—` in BrightBlack if absent.
- **MetalTile**: BrightWhite bold if correct; Red bold if correctness failed; Yellow bold if NYI (`~`).
- **MT%**: Green bold ≥ 90%; Yellow bold 60–89%; Red bold < 60%. `~` Yellow for NYI.
- **Correct**: Green `✓` on pass; `✓ 1.2e-5` when max_err ≥ 1e-5; Red `✗ 1.2e-2` on fail; BrightBlack `—` when unavailable; Yellow `! ?` when implemented but unchecked.

---

### `tile test` format

```
  exp      f32   ✓  1.0e-6
           f16   ✓  8.1e-4
           bf16  ✓  7.8e-3

  rms_norm f32   ✓  1.2e-5
           f16   ✗  max_err=0.12  (tol=0.001)
           bf16  ✓  8.4e-3

  ─────────────────────────────────
  153 passed  ·  1 failed
```

Op name: Cyan. Dtype tag: BrightBlack. `✓`: Green. `✗`: Red, followed by BrightWhite `max_err=` value and BrightBlack `(tol=...)`. Summary sep `·` is BrightBlack dim.

`-v` adds side-by-side diff on failure:

```
           f16   ✗  max_err=0.12  (tol=0.001)
                    expected  [ 0.1250  0.2500  0.3750  0.5000 ... ]
                    got       [ 0.1251  0.2502  0.5039  0.5001 ... ]
                                                ^^^^^^
```

---

### `tile build` format

```
  exp      f32/f16/bf16  ✓
  rms_norm f32/f16/bf16  ✓
  my_op    f32           ✗
           undefined variable v_42
           at metaltile-codegen/src/msl.rs:emit_block

  ─────────────────────
  14 ok  ·  1 error
```

Error detail lines indented to align under the kernel name. Error message BrightWhite, location BrightBlack.

`-v` prints the generated MSL for every kernel even on success, separated by the kernel name as a comment header.

---

### `tile diff` format

```
  rms_norm  f32    104% → 97%    ▼  -7%   REGRESSION
  softmax   f16     98% → 101%   ▲  +3%
  exp       f32    100% → 100%   —   0%

  ─────────────────────────────────────────
  1 regression (threshold 5%)  ·  2 improved  ·  151 unchanged
```

Row colors: regression rows — entire row Red. improvement rows — `▲` and delta Green. unchanged — BrightBlack for the delta column. `REGRESSION` label: Red bold.

---

### `tile device` format

```
  Device        Apple M4 Max
  GPU family    Apple9  (M3+)
  ──────────────────────────
  native_bfloat ✓   Metal 3.1+ bfloat type
  simdgroup_hw  ✓   M3+ simdgroup matrix multiply
  async_copy    ✓   async threadgroup copy
  ──────────────────────────
  Threadgroup   32 KB
  Max TPG       1024
  SLC           ~64 MB
```

Label column: BrightBlack. Value: BrightWhite. `✓`: Green, `✗`: Red. Parenthetical notes: BrightBlack dim.

---

### `tile snap` / `tile inspect` format

```
  Saved    → .tile-snapshots/m4max-2026-05-16.json  (154 results, "after fusion fix")
```

```
  Kernels registered: 47

  mt_exp             Elementwise   f32/f16/bf16
  mt_log             Elementwise   f32/f16/bf16
  mt_rms_norm        Reduction     f32/f16/bf16
  ...
```

For `tile inspect <kernel>`, MSL is printed with no color (it gets piped to Metal tools). IR dump uses Cyan for op names, BrightBlack for `v0 =` prefixes, BrightWhite for values.

---

### Summary lines

All commands end with a summary. Format: items separated by `·` (BrightBlack dim). Each item is a BrightBlack label followed by a semantically-colored value.

```
  Implemented 154/154  ·  Avg MT% 120%  ·  Correct 146/154  ·  1 failed
```

Value colors in summary: counts matching total — Green bold. passing counts — Green bold. failing counts — Red bold. averages — pct-color rules (same as MT% column).

---

### Implementation notes

Changes needed to `term.rs`:

1. Add `Blue` color variant (`34`)
2. Add `Italic` style bit (for future use, not needed in Phase 1)
3. Add `unicode_enabled()` function that returns false when `--color never` or non-TTY — used by symbol helpers
4. Add a `sym` module with `SYM_OK`, `SYM_FAIL`, `SYM_NYI`, `SYM_ARROW`, `SYM_DOWN`, `SYM_UP`, `SYM_DASH`, `SYM_COL_SEP`, `SYM_RULE` constants that select unicode or ASCII based on `unicode_enabled()`
5. Add a `verb(label, style)` helper that right-pads to 12 chars and emits the `  Label   content` pattern

The `--color [always|never|auto]` global flag wires into the existing `OnceLock` detection logic, overriding it before any first call.

---

## Extended Commands

### `tile trace`

**Purpose**: generate a Metal GPU capture (`.gputrace`) that can be opened in Xcode Instruments / GPU Frame Debugger. The single most useful tool for diagnosing unexpectedly slow kernels — shows per-shader timing, occupancy, ALU vs memory bound, and buffer contents at each dispatch.

```
$ tile trace rms_norm
$ tile trace rms_norm -o captures/rms_norm.gputrace
$ tile trace rms_norm --dtype bf16
$ tile trace rms_norm --open      # open in Xcode after capture
```

Output:
```
  Tracing    rms_norm f32
  Saved    → captures/rms_norm_f32_2026-05-16.gputrace
             open with: open captures/rms_norm_f32_2026-05-16.gputrace
```

**Implementation**: wraps `GpuRunner` dispatch with `MTLCaptureManager`. Sequence: warmup run → `startCapture(descriptor:)` with `destination = .gpuTraceDocument` → timed run → `stopCapture()` → print path. The capture descriptor's `outputURL` is the only non-trivial parameter. If `--open`, shell out to `open <path>`.

Flags:
- `-o, --out <path>` — output path (default: `.tile-traces/<kernel>_<dtype>_<date>.gputrace`)
- `--dtype [f32|f16|bf16]` — dtype specialization
- `--open` — launch Xcode after capture completes
- `--runs <n>` — number of dispatches to capture (default: 1)

---

### `tile profile`

**Purpose**: run a kernel with `MTLCounterSet` sampling and report hardware utilization metrics alongside throughput. Answers "why is this fast/slow" rather than just "how fast is it". Turns a throughput number into a diagnosis.

```
$ tile profile rms_norm
$ tile profile rms_norm --filter rms,softmax
$ tile profile rms_norm --json
```

Output:
```
  rms_norm f32

  Throughput      398.1 GB/s  (97% of reference)
  ──────────────────────────────────
  Occupancy       87%         ← good
  ALU active      43%         ← memory-bound, not compute-bound
  BW achieved     398 GB/s    / 820 GB/s theoretical  (48%)
  L1 hit rate     91%
  Simdgroup util  96%
```

The ALU vs BW gap identifies the root cause: a kernel at 43% ALU and 48% BW is memory-bound — more arithmetic reuse (tiling, fusion) will help. A kernel at 90% ALU and 20% BW is compute-bound.

**Implementation**: `MTLCommonCounterSetStatistic` provides occupancy, ALU active fraction, and memory bandwidth on Apple Silicon. Requires creating a `MTLCounterSampleBuffer`, attaching it to the command encoder before and after dispatch, then resolving. Counter availability varies by GPU family — check `device.supportsCounterSampling(.atStageBoundary)` before using.

Flags:
- `-f, --filter <pattern>` — only profile matching kernels
- `--dtype [f32|f16|bf16]`
- `--json` — emit as JSON
- `--counters <list>` — restrict which counters to sample (default: all available)

---

### `tile fuzz`

**Purpose**: find correctness bugs by running a kernel with systematically adversarial inputs — NaN, inf, denormals, near-overflow, all-zeros, all-same. Builds on `check_equiv` but replaces the standard test inputs with edge-case categories.

```
$ tile fuzz rms_norm
$ tile fuzz rms_norm --categories nan,inf,denormal
$ tile fuzz rms_norm --seed 42 --rounds 1000   # random mode
```

Output:
```
  rms_norm f32

  zeros       ✓
  ones        ✓
  nan         ✗  interp=nan, gpu=0.0  (GPU silent-NaN-to-zero divergence)
  inf         ✓
  denormal    ✓
  near-max    ✓
  alternating ✓
  random      ✓  (seed=0, 100 rounds)
```

Built-in input categories:

| Category | Values |
|----------|--------|
| `zeros` | all 0.0 |
| `ones` | all 1.0 |
| `nan` | single NaN at index N/2 |
| `inf` | single +inf and -inf pair |
| `denormal` | values near f32/f16 minimum (1e-38) |
| `near-max` | values near f32 max (1e38) |
| `alternating` | cycling large positive / large negative |
| `random` | seeded PRNG, `--rounds` iterations |

Flags:
- `-f, --filter <pattern>`
- `--categories <list>` — run only specified categories (default: all)
- `--seed <n>` — RNG seed for random mode (default: 0)
- `--rounds <n>` — random rounds (default: 100)
- `--dtype [f32|f16|bf16]`

---

## Op Classification Reference

The `class` in `#[bench_kernel(class=...)]` drives `BenchSpec` — it determines which generic runner to use, what shapes to bench/check, and how the dispatch grid is computed.

| Class | `BenchDispatch` | bench N | check N | TPG |
|-------|----------------|---------|---------|-----|
| `Unary` | Generic | 64M | 2048 | 256 |
| `Binary` | Generic | 64M | 2048 | 1024 (N/2 threads) |
| `AllReduce` | Generic | 64M | 16384 | 256 |
| `RowReduce` | Generic | 1024×4096 | 8×512 | 256 |
| `Arange` | Generic | 64M | 4096 | 1024 |
| `BinaryTwo` | Generic | 64M | 2048 | 1024 |
| `Select` | Generic | 64M | 2048 | 256 |
| `RowNorm` | Generic | 1024×4096 | 8×512 | 1024 |
| `MatVec` | Generic | 4096×4096 | 64×256 | 64 |
| `MatVecMasked` | Generic | 4096×4096 | 64×256 | 64 |
| `Sort` | Custom | - | - | - |
| `Scan` | Custom | - | - | - |
| `ArgReduce` | Custom | - | - | - |
| `Random` | Custom | - | - | - |
| `FpQuantized` | Custom | - | - | - |
| `QuantizedMatVec` | Custom | - | - | - |
| `Rope` | Custom | - | - | - |
| `Attention` | Custom | - | - | - |
| `StridedCopy` | Custom | - | - | - |

---

## Migration Phases

### Phase 1 — `tile` binary, no structural changes

Create `crates/metaltile-cli/` with a `tile` binary. All logic delegates to existing `metaltile-bench` internals.

Deliverables:
- `tile bench` — wraps current `bench_suite` logic
- `tile inspect` — replaces `dump_msl`, auto-discovers via inventory
- `tile device` — new, queries Metal device
- `tile build` — new, iterates inventory + runs codegen only
- Delete `bench_suite.rs` and `dump_msl.rs` as standalone binaries (logic absorbed into `tile`)

No change to where ops live. No change to `spec.rs` or `shared.rs`.

### Phase 2 — `tile test`, `tile snap`, `tile diff`, IR pretty-printer

- `tile test` — correctness-only runner, extracts the `check_equiv` path from `BenchSpec`
- `tile snap` / `tile diff` — snapshot format, JSON save/load, regression comparison
- IR pretty-printer in `metaltile-core` as `Display` on `Kernel` — needed for `tile inspect --ir` and `--pass`

### Phase 3 — `metaltile-std` extraction, `metaltile-bench` elimination

Two moves, one PR:

1. **Create `crates/metaltile-std/`**. Move all op files from `metaltile-bench/src/ops/` and all bench metadata types (`BenchSpec`, `BenchDispatch`, `ShapeSpec`, `Dim`, `BufInit`, `DispatchGrid`, `MlxArg`, `ScalarBufSpec`, `TensorBufSpec`, bytes helpers, `inventory::collect!(BenchSpec)`) from `spec.rs`.

2. **Absorb remaining `metaltile-bench` into `metaltile-cli/`**. Move `runner.rs`, measurement helpers (`bench_gbps`, `check_equiv`, `OpResult`, `SuitePrinter`, `stats.rs`, `term.rs`), and `BenchSpec::run()` (renamed `run_spec`) into `metaltile-cli/src/`. Delete `crates/metaltile-bench/`.

No functional change — just crate boundaries shifting. After this phase, `metaltile-bench` no longer exists.

### Phase 4 — `tile trace`, `tile profile`, `tile fuzz`, polish

- `tile trace` — `MTLCaptureManager` integration, `.gputrace` output, `--open` Xcode launch
- `tile profile` — `MTLCounterSet` sampling (occupancy, ALU active, BW achieved, L1 hit rate)
- `tile fuzz` — edge-case input categories built on `check_equiv`
- `tile build --metal-compile` — invoke `metal`/`metallib` for GPU-level type checking
- Man pages / shell completions
- REPL (low priority, high effort)

---

## Decisions

| # | Question | Decision |
|---|----------|----------|
| 1 | CLI binary name | `tile` (crate: `metaltile-cli`) |
| 2 | Kernel stdlib crate name | `metaltile-std` |
| 3 | Keep `metaltile-bench` as a crate? | No — eliminated in Phase 3; metadata → `metaltile-std`, runner → `metaltile-cli` |
| 4 | `tile build --metal-compile`: opt-in metallib compile | Yes, opt-in, Phase 4 |
| 5 | `tile test` reference | Interpreter by default; `--ref mlx` for MLX fallback |
| 6 | `tile diff` default regression threshold | 5% |
| 7 | IR pretty-printer | Structured `Display` on `Kernel`, Phase 2 |
| 8 | Snapshot directory | `.tile-snapshots/`, committed to repo |
| 9 | `tile bench --no-ref` | Yes, graceful degradation when MLX metallib absent |
