# Cross-developer bench baselines

Per-machine snapshots of `tile bench` results, captured against a known
upstream SHA, that the project keeps in version control as a regression
reference. Distinct from `.tile-snapshots/` (per-developer, gitignored
output directory used by `tile snap`).

Workflow when adding a new baseline:

1. Sync to the SHA you want to capture (typically `dev` HEAD).
2. `cargo build --release -p metaltile-cli`
3. `tile bench --json /tmp/bench-raw.json`
4. `tile snap --from /tmp/bench-raw.json -o baselines/<chip-slug>.json
   --note "<context, e.g. compiler flags, env vars>"`
5. Skim the JSON for regressions vs the prior baseline for the same
   chip (or for a related chip family) and append the headline findings
   to this file.
6. Open a PR.

Each baseline JSON is a `Snapshot` envelope (`tile snap` format):
`{ device, gpu_family?, git_sha?, timestamp, note?, results[...] }`.
Per-row schema is documented in `cmd::bench::save_json`. Captured with
default codegen flags unless the `note` says otherwise.

Naming: `baselines/<chip-slug>.json`. One canonical file per chip;
overwrite on update (the file metadata carries the SHA + timestamp,
git history preserves the older snapshots). The slug is derived from
`tile device`'s reported name — lowercase, non-alphanumeric runs
collapsed to a single dash — so `Apple M5 Max` ↔ `apple-m5-max.json`.

## Dirty-tree guard and auto-diff

`tile bench` refuses to run when `git diff HEAD` is non-empty: bench
numbers from a stale `target/` against a modified source tree don't
tie back to any commit. Override with `--allow-dirty` when you're
intentionally measuring uncommitted work.

On success, bench also runs a diff against the target-branch baseline
without being asked. It picks the file `baselines/<your-chip>.json`
out of the first ref of `origin/dev`, `upstream/dev`, `dev` that
resolves, at the merge-base with `HEAD` — so the comparison stays
honest even on a PR that already updated its own baseline. Override
the ref with `--baseline-ref <ref>`, or opt in with `--diff`.

In CI (`.github/workflows/kernels.yml`), the same diff is rendered
against the in-tree baseline matching the runner chip and posted as a
PR comment. The runner chip rarely matches any committed slug, so
that step is informational only — it never gates the job.

## Current baselines

| File | Chip | Captured @ | Headline |
|---|---|---|---|
| [`apple-m1-max.json`](apple-m1-max.json) | Apple M1 Max | `dd4c2ef` (2026-05-20) | 327/327 implemented + numerically correct; **avg MT% 146%**. Refreshed against current `dev` (post #110/#128/#129/#135); the prior baseline predated those landings so `tile bench`'s auto-diff was comparing against stale numbers. |
| [`apple-m5-max.json`](apple-m5-max.json) | Apple M5 Max | `0cb0a85` (2026-05-18) | 241/241 implemented + numerically correct; **avg MT% 136%** but masked by an `sdpa` GQA bf16 regression to **31%** of MLX (see below). |

## Apple M5 Max — headline findings (2026-05-18, dev @ `0cb0a85`)

Codegen flags: `native_bfloat=true`, `use_simd_matrix=false`,
`async_copy=false` (`MslConfig::default()`).

### Regressions on the LLM hot path

The "avg MT% 136%" headline is driven by big elementwise wins; the
kernels that actually matter for LLM decode regress on M5 Max:

| Kernel | Shape | dtype | MT / MLX |
|---|---|---|---|
| `sdpa` | H=32 N=4096 D=128 **gqa=4** | **bf16** | **31%** |
| `sdpa` | H=32 N=4096 D=128 gqa=4 | f16 | 62% |
| `sdpa` | H=32 N=4096 D=128 gqa=4 | f32 | 55% |
| `sdpa` | H=8 N=2048 D=128 | f32 | 33% |
| `softmax` | B=1024 N=4096 | bf16 | 29% |
| `affine` quant | bits=3 gs=32 | f16 | 32% |
| `affine` quant | bits=3 gs=32 | bf16 | 41% |
| `affine` quant | bits=4 gs=64 (one variant) | f16 | 24% |
| `rms_norm` | B=1024 N=4096 | f16 | 80% |
| `rope` | B1H32L512D128 (one variant) | f16 | 75% |

The dominant pattern is **SDPA + GQA + bf16**. MLX's
`sdpa_vector_2pass` picks a per-shape `blocks` value tuned for the
target chip; MetalTile's current `mt_sdpa` uses a fixed single-pass
8-simdgroups-per-head layout. The mismatch widens on M5 because the
optimal block size differs from M3/M4.

### Wins worth keeping an eye on

- `rms_norm` bf16: **338%** — native bf16 path lands cleanly.
- `affine` quant bits=4 bf16: 191%.
- Most `unary` / `binary` elementwise paths land 1.5–4× over MLX on
  bf16, and roughly at parity on f32/f16.

### Methodology caveats

- Single bench run, no warmup-aware averaging — treat individual
  outlier ratios (e.g. `unary sqrt f16` at 980%) as noise unless they
  reproduce on a re-run.
- Avg MT% is an unweighted mean over 241 kernel/shape/dtype rows.
  It does not reflect FFAI inference throughput; for that, drill into
  the per-kernel rows tied to the LLM hot path.
- `equiv` is checked for every row; 241/241 passed.

### Reproducing on M5 Max

```sh
git checkout 0cb0a85           # the SHA this baseline was captured against
cargo build --release -p metaltile-cli
./target/release/tile device   # confirm "Apple M5 Max" / "Apple10 (M5)"
# Auto-diffs against the merge-base baseline on success; emits JSON
# at /tmp/bench.json for snap/diff round-trips.
./target/release/tile bench --json /tmp/bench.json
# Or run the diff explicitly against the committed baseline:
./target/release/tile diff baselines/apple-m5-max.json /tmp/bench.json
```
