# metaltile

Rust DSL for writing Apple Metal GPU kernels — write once, run fast on Apple Silicon.
This is the user-facing facade crate: add `metaltile` to your `Cargo.toml`, import
`metaltile::prelude::*`, annotate functions with `#[kernel]`, and dispatch them on
the GPU with a few lines of Rust.

The crate re-exports the compiler, runtime, and macro crates under one namespace so
you never need to depend on `metaltile-core`, `metaltile-codegen`, or the others
directly unless you are writing tooling or compiler extensions.

## Position in the pipeline

```
        ┌──────────────────────────────┐
        │  metaltile (this crate)       │
        │  user-facing facade           │
        │                              │
        │  use metaltile::prelude::*;   │
        │  #[kernel]                    │
        │  kernel::launch(&ctx)         │
        └──────────┬───────────────────┘
                   │ re-exports
    ┌──────────────┼──────────────┬──────────────┐
    ▼              ▼              ▼              ▼
metaltile-core  metaltile-macros  metaltile-codegen  metaltile-runtime
   (IR types)    (#[kernel])      (MSL lowering)     (GPU dispatch)
```

`metaltile` is the only crate end users depend on. It re-exports the DSL macros,
placeholder types, IR/codegen modules, and runtime entry points under flat paths
like `metaltile::kernel`, `metaltile::core`, `metaltile::codegen`, and
`metaltile::Context`.

## Quick start

```rust
use metaltile::prelude::*;

#[kernel]
fn vector_add(a: Tensor<f32>, b: Tensor<f32>, c: Tensor<f32>) {
    let idx = program_id::<0>();
    store(c[idx], load(a[idx]) + load(b[idx]));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = Context::new()?;
    let n = 256usize;
    let a: Vec<u8> = (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let b: Vec<u8> = (0..n).flat_map(|_| (1.0f32).to_le_bytes()).collect();
    let c = vec![0u8; n * 4];

    let result = vector_add::launch(&ctx)
        .input("a", a)
        .input("b", b)
        .input("c", c)
        .dispatch()?;

    let out: Vec<f32> = result.outputs["c"]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    println!("out[0] = {}", out[0]); // 1.0
    Ok(())
}
```

To inspect the generated MSL directly without dispatching:

```rust
use metaltile::codegen::msl::MslGenerator;

let msl = MslGenerator::default().generate(&vector_add::kernel_ir())?;
println!("{msl}");
```

## Crate contents

| Module | Purpose |
|---|---|
| `prelude` | Everything needed in a `#[kernel]` module: `Tensor`, DSL stubs, macros, core types |
| `codegen` | Re-export of `metaltile_codegen` — MSL generation and optimization passes |
| `core` | Re-export of `metaltile_core` — IR types, DType, Shape, ConstExpr |

## API reference

### Prelude

`use metaltile::prelude::*;` brings these into scope:

**Macros (from `metaltile-macros`):**

| Macro | Kind | What it does |
|---|---|---|
| `#[kernel]` | attribute | Transforms a Rust function into IR + host-side `LaunchBuilder` |
| `#[constexpr]` | attribute | Marks a kernel parameter as a compile-time constant |
| `shape!(…)` | function-like | Constructs a `Shape` from dimension expressions |
| `tile!(…)` | function-like | Constructs a 2D tile shape |

**Types (from `metaltile-core`):**

| Type | Purpose |
|---|---|
| `Tensor<T, S>` | Placeholder type for kernel signatures — zero-sized, carries element type and optional shape |
| `DType` | Numeric type: F32, F16, BF16, I32, U32, etc. |
| `Shape` | Compile-time dimension tracking |
| `Dim` | A single dimension: `Known(usize)` or `ConstExpr(name)` |
| `ConstExpr` | Named compile-time constant |
| `KernelMode` | Dispatch shape hint: Elementwise, Reduction, Matmul |
| `Context` | Metal GPU device and command queue |

**DSL function stubs (panic if called outside `#[kernel]`):**

| Function | Purpose |
|---|---|
| `program_id::<AXIS>()` | Current thread/program ID along a grid axis |
| `load(tensor[idx])` | Load a value from a tensor index expression |
| `store(tensor[idx], value)` | Store a value into a tensor index expression |
| `dot(a, b)` | Dot product placeholder for tiled kernels |

**Unary math (recognized by the body parser):**
`exp`, `log`, `sqrt`, `rsqrt`, `abs`, `silu`, `gelu`, `relu`, `tanh`, `sigmoid`, `sin`, `cos`, `ceil`, `floor`, `recip`

### Re-exports

Directly accessible from `metaltile::`:

| Path | What it re-exports |
|---|---|
| `metaltile::kernel` | `#[kernel]` proc-macro attribute |
| `metaltile::bench_kernel` | `#[bench_kernel]` proc-macro attribute |
| `metaltile::constexpr` | `#[constexpr]` proc-macro attribute |
| `metaltile::shape` | `shape!` proc-macro |
| `metaltile::tile` | `tile!` proc-macro |
| `metaltile::codegen` | `metaltile_codegen` crate (MSL generator, optimization passes) |
| `metaltile::CodegenError` | `metaltile_codegen::error::Error` |
| `metaltile::core` | `metaltile_core` crate (IR, DType, Shape) |
| `metaltile::Context` | `metaltile_runtime::Context` — GPU device + command queue |
| `metaltile::DispatchResult` | `metaltile_runtime::DispatchResult` — output buffers after a kernel run |
| `metaltile::MetalTileError` | `metaltile_runtime::MetalTileError` — top-level runtime error |
| `metaltile::Tensor` | `prelude::Tensor` — placeholder tensor type |
| `metaltile::VERSION` | Crate version string constant |
| `metaltile::version()` | Returns `VERSION` |

## Dependencies

### Internal

| Crate | Role in this crate |
|---|---|
| `metaltile-core` | Re-exported as `metaltile::core`; provides IR types and DType for the prelude |
| `metaltile-macros` | Re-exported as individual proc macros (`kernel`, `bench_kernel`, `constexpr`, `shape`, `tile`) |
| `metaltile-codegen` | Re-exported as `metaltile::codegen`; provides MSL generation for inspection |
| `metaltile-runtime` | Re-exported as `Context`, `DispatchResult`, `MetalTileError`; provides GPU dispatch |

### External

None — all external dependencies come transitively through the internal crates.

## MSRV / platform

The facade crate itself has no platform gating. The runtime (`Context::new()`)
requires macOS + Metal; codegen and IR introspection work on any host.

Rust: nightly (workspace-wide, edition 2024).

## Extending

- **New re-export:** `src/lib.rs` — add `pub use metaltile_<crate>::<Item>;` with a doc comment.
- **New prelude item:** `src/prelude.rs` — add the type stub, function stub, or re-export, with a doc comment.
- **New DSL intrinsic:** `src/prelude.rs` — add a `pub fn` stub that panics, then add recognition in `metaltile-macros/src/body_parser.rs`.
- **Tests to update:** Doc-tests in `src/lib.rs`.

## Related documentation

- [Root README](../../README.md) — project overview and architecture
- [CONTRIBUTING](../../CONTRIBUTING.md) — dev setup, PR process, CI
- [`metaltile-core` README](../metaltile-core/README.md) — the IR types prelude re-exports
- [`metaltile-macros` README](../metaltile-macros/README.md) — how `#[kernel]` transforms your function
- [`metaltile-codegen` README](../metaltile-codegen/README.md) — the MSL generator behind `metaltile::codegen`
- [`metaltile-runtime` README](../metaltile-runtime/README.md) — the `Context` and dispatch lifecycle
- [Crate docs on docs.rs](https://docs.rs/metaltile)

## License

Licensed under the [Apache License, Version 2.0](../../LICENSE).
