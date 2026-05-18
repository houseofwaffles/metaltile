# metaltile-macros

Proc-macro crate providing the `#[kernel]` DSL for MetalTile GPU kernels.
Parses Rust function signatures and bodies at compile time, translates
DSL intrinsics into `metaltile-core` IR, and generates host-side launch
code.

This crate is the front door of the MetalTile compiler: user-written
`#[kernel]` functions enter here, and IR + dispatch surfaces exit. It
also provides `shape!`/`tile!` constructors for shape annotations and
`#[bench_kernel]` for declarative benchmark registration.

## Position in the pipeline

```
User Rust code (fn with #[kernel])
            │
      metaltile-macros (this crate)
            │
      metaltile-core IR (Kernel, Block, Op, …)
            │
      metaltile-codegen (opt passes → MSL)
            │
      metaltile-runtime (GPU dispatch)
```

The proc macro runs at the call site's compile time. It consumes the
user's token stream and produces a module containing `kernel_ir()`,
`kernel_ir_for(DType)`, `LaunchBuilder`, and a `launch()` entry point.

## Quick start

Define and expand a kernel:

```rust,ignore
use metaltile_macros::kernel;

#[kernel]
pub fn vector_add(a: Tensor<f32>, b: Tensor<f32>, c: Tensor<f32>) {
    let idx = program_id::<0>();
    store(c[idx], load(a[idx]) + load(b[idx]));
}

// The macro generates:
//   pub mod vector_add {
//       pub fn kernel_ir() -> Kernel { … }
//       pub fn kernel_ir_for(_t: DType) -> Kernel { … }
//       pub struct LaunchBuilder<'a> { … }
//       pub fn launch(ctx: &Context) -> LaunchBuilder<'_> { … }
//   }
```

For generic kernels:

```rust,ignore
#[kernel]
pub fn scale<T>(a: Tensor<T>, factor: f32, out: Tensor<T>) {
    let idx = program_id::<0>();
    store(out[idx], load(a[idx]) * factor);
}
// Now call: scale::kernel_ir_for(DType::F16)
```

## Crate contents

| Module | Purpose |
|---|---|
| `lib.rs` | All proc-macro entry points: `#[kernel]`, `#[autotune]`, `#[bench_kernel]`, `#[constexpr]`, `#[scalar]`, `#[strided]`, `shape!`, `tile!` |
| `body_parser.rs` | `DslBodyParser` — walks `syn::Expr` trees and translates DSL calls into IR-building token streams |

## API reference

### Macros

| Macro | Kind | What it does |
|---|---|---|
| `#[kernel]` | attribute | Parses a Rust function into IR + generates a module with `kernel_ir`, `kernel_ir_for`, `LaunchBuilder`, and `launch()` |
| `#[autotune]` | attribute | Placed before `#[kernel]` to enable autotuning: `#[autotune(configs = [...], key = [M, N, K])]`. **Not yet implemented** — `AutotuneArgs` struct exists but parsing is a TODO in `expand_kernel`. |
| `#[bench_kernel]` | attribute | Registers a kernel for automatic benchmarking via `inventory::submit!`. Must be placed *before* `#[kernel]`. |
| `#[constexpr]` | attribute | Pass-through: marks a function parameter as a compile-time constant detected by `#[kernel]` |
| `#[scalar]` | attribute | Pass-through: marks a `Tensor` parameter for `constant T&` lowering in MSL |
| `#[strided]` | attribute | Pass-through: marks a `Tensor` parameter for strided lowering (shape + stride arrays emitted) |
| `shape!` | function-like | Constructs a `Shape` from dimension expressions: `shape!(M, K)`, `shape!(32, 64)`, `shape!()` |
| `tile!` | function-like | Constructs a 2D tile shape: `tile!(TILE_M, TILE_N)`, `tile!(32, 64)` |

### What `#[kernel]` expands to

For a kernel `pub fn my_kernel(a: Tensor<f32>, out: Tensor<f32>) { … }`, the expansion produces:

```
pub mod my_kernel {
    // Build IR for specific DType(s). For non-generic kernels this takes ().
    pub fn kernel_ir_for(_t: DType) -> Kernel { … }

    // Default to f32.
    pub fn kernel_ir() -> Kernel { kernel_ir_for(DType::F32) }

    // Host-side builder.
    pub struct LaunchBuilder<'a> { … }
    impl<'a> LaunchBuilder<'a> {
        pub fn input(self, name: &str, data: Vec<u8>) -> Self { … }
        pub fn dispatch(self) -> Result<DispatchResult, MetalTileError> { … }
    }

    // Entry point.
    pub fn launch(ctx: &Context) -> LaunchBuilder<'_> { … }
}
```

For generic kernels (`fn foo<T>(a: Tensor<T>, …)`), `kernel_ir_for` takes
one `DType` argument per type parameter (`kernel_ir_for(_t: DType)`).
The `#[bench_kernel]` macro detects generics and calls `kernel_ir_for`
directly instead of wrapping in a closure.

Output tensors are detected by one of:
- `mut` binding on a `Tensor` parameter (e.g. `mut result: Tensor<f32>`)
- Legacy heuristic: parameter named `out`, `c`, or `output`

### Kernel-level attributes

Attributes placed on the function itself (before or alongside `#[kernel]`):

| Attribute | Effect |
|---|---|
| `#[autotune(configs = [...], key = [M, N, K])]` | **Not yet implemented.** Enables autotuning for this kernel. `configs` is a comma-separated list of config names (defined in the autotuner). `key` lists the shape dimensions used for cache bucketing. |

### Kernel parameter attributes

| Attribute | Effect |
|---|---|
| `#[constexpr]` | Extracts the parameter as a `ConstExprDecl` in the kernel IR. Used for shape dimensions and compile-time constants. Automatically deduplicated — the same name appearing in multiple tensor shapes only generates one constexpr. |
| `#[scalar]` | Emits the parameter as `constant T& name` in MSL rather than `device T*`. Used for scalar values like `eps` or `scale`. |
| `#[strided]` | Emits the parameter as `device T*` plus `constant uint* name_shape` and `constant uint* name_strides` in MSL. Used for non-contiguous tensor views. |

### `#[bench_kernel]` arguments

| Argument | Required | Purpose |
|---|---|---|
| `op` | yes | Bench table group, e.g. `"unary"`, `"binary"` |
| `subop` | yes | Sub-operation label, e.g. `"exp"`, `"add"` |
| `class` | yes | Dispatch class: `Unary`, `Binary`, `AllReduce`, `RowReduce`, `Arange`, `BinaryTwo`, `Select`, `RowNorm`, `Sort`, `Scan`, `ArgReduce`, `Random`, `FpQuantized`, `MatVec`, `MatVecMasked`, `QuantizedMatVec`, `Rope`, `Attention`, `StridedCopy` |
| `tol` | yes | Maximum absolute correctness error, e.g. `1e-4` |
| `input` | no | Input buffer init for unary: `Signed`, `Positive`, `Half`, `Unit` (default: `Half`) |
| `input_a` / `input_b` | no | Input buffer init for binary (default: `Half`) |
| `mlx` | no | MLX kernel name pattern; `{tn}` is replaced with the MLX type name |
| `metal_file` | no | MLX reference .metal source path (loaded via `include_str!`) |
| `dtypes` | no | `&'static [DType]` slice (default: `FLOAT_DTYPES`) |
| `shapes` | no | Custom `ShapeSpec` array for complex dispatch shapes |
| `start` / `step` | no | Arange start/step values (float literals) |
| `reads` | no | Read count for bandwidth calculation (`RowNorm` class) |
| `out_elements` | no | Output element count (`RowNorm` class; 1 = per-row scalar, >1 = full B×N) |
| `tpg` | no | Threads per threadgroup override |
| `pre_weight` / `pre_bias` / `post_eps` | no | RowNorm-specific: weight, bias, epsilon values |
| `n` / `check_n` / `b` | no | Shape dimensions for complex dispatch (`RowNorm`, `Rope`) |
| `h` / `l` / `d` / `n_per_group` | no | Rope-specific dimensions |
| `group_size` | no | Quantization group size (`QuantizedMatVec` class) |
| `m` / `pad` | no | StridedCopy dimensions |

## Dependencies

### Internal

| Crate | Role in this crate |
|---|---|
| `metaltile-core` | Emits IR type constructors (`Kernel`, `Block`, `Op`, `DType`, `Shape`, `ConstExpr`) in the generated token stream |

### External

| Crate | Role |
|---|---|
| `syn` | Parses user-written Rust functions and DSL bodies |
| `quote` | Token-stream construction for generated code |
| `proc-macro2` | Proc-macro token stream API |
| `darling` | Derive-based attribute parsing (`AutotuneArgs`) |

## MSRV / platform

No platform gating — pure compile-time code, no GPU calls.
Rust: nightly (workspace-wide, edition 2024).
Requires `[lib] proc-macro = true` in `Cargo.toml`.

## Extending

- **New DSL intrinsic:** `src/body_parser.rs` — add a recognized function name to the
  expression walker. Update the `Recognized call:` list in the module doc comment.

- **New kernel parameter attribute:** `src/lib.rs` — add a new `#[proc_macro_attribute]`
  pass-through function, update `has_attr` checks, and wire it into
  `parse_kernel_params_generic`.

- **New kernel-level attribute (like `#[autotune]`):** `src/lib.rs` — add the
  `#[proc_macro_attribute]` pass-through, parse its args in `expand_kernel`,
  and emit the corresponding token stream into the generated module.

- **New `bench_kernel` class:** `src/lib.rs` — add variant to `ClassKind` enum in
  `bench_impl`, add a match arm in `generate_submit` with its `ShapeSpec` and
  `BenchDispatch` variant.

- **New `bench_kernel` argument:** `src/lib.rs` — add field to `BenchArgs`, add parse
  arm in `BenchArgs::parse()`, consume in `generate_submit`.

- **New shape/tile constructor syntax:** `src/lib.rs` — add a new `#[proc_macro]`
  function following the `shape!` / `tile!` pattern.

- **Tests to update:** Unit tests in `src/lib.rs` (at bottom of file). The tests
  cover param output detection, constexpr deduplication, and legacy output naming.

## Related documentation

- [Root README](../../README.md) — project overview and architecture
- [CONTRIBUTING](../../CONTRIBUTING.md) — dev setup, PR process, CI
- [`metaltile-core` README](../metaltile-core/README.md) — the IR types emitted by these macros
- [`metaltile-codegen` README](../metaltile-codegen/README.md) — the passes that consume the generated IR
- [`metaltile-std` README](../metaltile-std/README.md) — the `BenchSpec` type that `#[bench_kernel]` submits to
- [Crate docs on docs.rs](https://docs.rs/metaltile-macros)

## License

Licensed under the [Apache License, Version 2.0](../../LICENSE).
