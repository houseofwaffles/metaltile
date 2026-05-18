# metaltile-codegen

Metal Shading Language (MSL) code generator for MetalTile kernels.
Takes algorithm IR from `metaltile-core`, applies a 14-pass optimization
pipeline, and emits valid MSL source ready for the Metal compiler.

This crate is the middle of the MetalTile compiler stack: it receives
`Kernel` IR nodes, lowers tile-level ops into thread-mapped, vectorized
MSL, and exposes `MslGenerator` for both programmatic use and the
`tile inspect` / `tile build` CLI flows.

## Position in the pipeline

```
metaltile-core (IR) ──► metaltile-codegen (this crate) ──► metaltile-runtime
                                │
                         14 opt passes
                         MSL emission
```

The crate is a pure compiler — it has no Metal runtime dependency.
Generated MSL is handed to `metaltile-runtime` for PSO compilation
and dispatch, or serialized to disk by `metaltile-cli`.

## Quick start

Generate MSL from a kernel's IR:

```rust,ignore
use metaltile_codegen::msl::{MslGenerator, MslConfig};

let kernel = my_kernel::kernel_ir_for(DType::F16);
let msl = MslGenerator::default().generate(&kernel)?;
println!("{msl}");
```

Or with custom configuration:

```rust,ignore
let config = MslConfig {
    debug_comments: true,
    native_bfloat: false,
    ..MslConfig::default()
};
let msl = MslGenerator::new(config).generate(&kernel)?;
```

## Crate contents

| Module | Purpose |
|---|---|
| `msl` | MSL generator and configuration |
| `msl::emit_block` | Block-level MSL emission (the main lowering engine) |
| `msl::fused` | Fused-operation codegen (fused multiply-add, etc.) |
| `msl::matmul` | Matrix-multiplication MSL patterns |
| `msl::reduce` | Reduction codegen (simdgroup, threadgroup) |
| `msl::helpers` | Shared MSL helper functions |
| `msl::preamble` | Header includes, typedefs, feature gates |
| `msl::features` | Metal language feature version detection |
| `msl::config` | `MslConfig` struct |
| `passes` | Optimization pass infrastructure and all pass implementations |
| `passes::mod` | `Pass` trait, `PassRegistry`, `PipelineBuilder` |
| `emit` | Multi-file .metal + manifest + .metallib emission |
| `error` | `Error` enum and `Result` alias |

## API reference

### Optimization pipeline

Passes run in this order. The canonical order is defined in
`PassRegistry::order()` and `PassRegistry::get()`.

```
TypeCheck → ConstFold → AlgebraicSimplify → CopyProp → CSE → LICM
  → IfConversion → ValueSink → TileLowering → Fusion → Unroll
  → Schedule → Vectorize → DeadStoreElim
```

| Order | Pass | File | Effect |
|---|---|---|---|
| 1 | `type_check` | `passes/type_check.rs` | Validates dtype consistency, shape compatibility, block scoping |
| 2 | `const_fold` | `passes/const_fold.rs` | Evaluates constant expressions at compile time |
| 3 | `algebraic_simplify` | `passes/algebraic_simplify.rs` | Rewrites identities (x+0→x, x*1→x, etc.) |
| 4 | `copy_prop` | `passes/copy_prop.rs` | Replaces value copies with their sources |
| 5 | `cse` | `passes/cse.rs` | Common subexpression elimination |
| 6 | `licm` | `passes/licm.rs` | Loop-invariant code motion |
| 7 | `if_conversion` | `passes/if_conversion.rs` | Converts conditional blocks to predicated ops |
| 8 | `value_sink` | `passes/value_sink.rs` | Sinks computations closer to their uses |
| 9 | `tile_lowering` | `passes/tile_lowering.rs` | Lowers tile-level ops to explicit thread loops and shared memory |
| 10 | `fusion` | `passes/fusion.rs` | Merges compatible adjacent ops into fused operations |
| 11 | `unroll` | `passes/unroll.rs` | Loop unrolling with configurable factor |
| 12 | `schedule` | `passes/schedule.rs` | Assigns ops to simdgroup lanes, inserts barriers |
| 13 | `vectorize` | `passes/vectorize.rs` | Packs scalar ops into vector ops (e.g. `float4`) |
| 14 | `dead_store_elim` | `passes/dead_store_elim.rs` | Removes stores to outputs that are never read |

You can customize the pipeline at runtime:

```rust,ignore
use metaltile_codegen::passes::PipelineBuilder;

let passes = PipelineBuilder::standard()
    .without("licm")
    .with_unroll_factor(8)
    .build();
```

### MSL generation

`MslGenerator` is the main entry point. Configure with `MslConfig`:

| Config field | Default | Purpose |
|---|---|---|
| `simd_size` | `32` | SIMD group width |
| `use_simd_matrix` | `false` | Emit `simdgroup_multiply_accumulate` (requires M1+) |
| `debug_comments` | `false` | Emit `//` comments with IR value IDs |
| `native_bfloat` | `true` | Use native `bfloat` type (Metal 3.1+, M3+) vs. `bfloat16_t` struct |
| `async_copy` | `false` | Emit `async_copy` prefetch (Metal 3, M2+) |
| `tile_schedule` | `TileSchedule::default()` | Thread-to-tile mapping for lowering |

Errors return `codegen::Error` with variants for unsupported ops, MSL
generation failures, and forwarded core errors.

## Dependencies

### Internal

| Crate | Role in this crate |
|---|---|
| `metaltile-core` | Reads `Kernel`, `Op`, `DType`, `Shape` from IR |

### External

| Crate | Role |
|---|---|
| `thiserror` | Derive `Error` for the error enum |
| `smallvec` | Small-vector optimization in pass internals |
| `half` | `f16` / `bf16` constant handling |
| `serde` / `serde_json` | Serialize manifest JSON during `emit` |

## MSRV / platform

No platform gating — `metaltile-codegen` is a pure-Rust compiler that
runs on any host OS. It generates MSL text but never calls Metal APIs.

Rust: nightly (workspace-wide, for edition 2024).

## Extending

- **New pass:** Create `src/passes/<name>.rs`. Implement the `Pass` trait
  (`fn name()`, `fn run(&self, kernel: &mut Kernel)`). Add the pass name to
  `PassRegistry::order()` and a `Box::new(...)` entry to `PassRegistry::get()`
  in `src/passes/mod.rs`. Add `pub mod <name>;` at the top of `mod.rs`.

- **Custom pipeline:** Use `PipelineBuilder` — chain `.without()`,
  `.with_unroll_factor()`, or build your own `Vec<Box<dyn Pass>>`.
  No need to edit `PassRegistry` for one-off pipelines.

- **New MSL intrinsic:** Add the emission logic to `src/msl/emit_block.rs`
  (the main lowering dispatcher). If it's a new category (matmul pattern,
  reduction pattern), add a dedicated module under `src/msl/`.

- **New MSL feature gate:** `src/msl/features.rs` — add a version check
  for the Metal language feature.

- **New `MslConfig` field:** `src/msl/config.rs` — add the field,
  add a default, consume it in the relevant emitter code.

- **Tests to update:** Unit tests in each pass file. MSL snapshot tests
  in `src/msl/` (if any). Run `make test` workspace-wide.

## Related documentation

- [Root README](../../README.md) — project overview and architecture
- [CONTRIBUTING](../../CONTRIBUTING.md) — dev setup, PR process, CI
- [`metaltile-core` README](../metaltile-core/README.md) — the IR types this crate lowers
- [`metaltile-runtime` README](../metaltile-runtime/README.md) — the runtime that consumes generated MSL
- [Crate docs on docs.rs](https://docs.rs/metaltile-codegen)

## License

Licensed under the [Apache License, Version 2.0](../../LICENSE).
