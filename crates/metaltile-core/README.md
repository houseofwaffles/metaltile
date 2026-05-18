# metaltile-core

Core IR types, shape algebra, and DType system for the MetalTile GPU
kernel compiler. This is the foundation crate — every other crate in
the workspace depends on it.

Defines the SSA-form intermediate representation that `#[kernel]`
functions parse into, that `metaltile-codegen` optimizes and lowers,
and that `metaltile-runtime` reads for dispatch metadata.

## Position in the pipeline

```
                    ┌─────────────────────┐
                    │  metaltile-core      │
                    │  (this crate)        │
                    │  IR · DType · Shape  │
                    └──────┬──────┬───────┘
                           │      │
              ┌────────────┘      └────────────┐
              ▼                                ▼
    metaltile-macros                  metaltile-codegen
    (parses Rust → IR)               (lowers IR → MSL)
```

Every crate depends on `metaltile-core`. The crate itself has no
dependency on the proc-macro, codegen, or runtime layers.

## Quick start

Build IR programmatically (no proc-macro needed):

```rust,ignore
use metaltile_core::ir::{Kernel, Block, Op, Param, ParamKind, BinOpKind};
use metaltile_core::dtype::DType;
use metaltile_core::shape::Shape;

let mut kernel = Kernel::new("add_two");
kernel.params.push(Param {
    name: "x".into(),
    dtype: DType::F32,
    shape: Shape::scalar(),
    is_output: false,
    kind: ParamKind::Tensor,
});

let mut block = Block::new(BlockId::new(0));
let a_id = ValueId::new(0);
let two_id = ValueId::new(1);
let sum_id = ValueId::new(2);
block.push_op(Op::Load { src: "x".into(), indices: vec![], mask: None, other: None }, a_id);
block.push_op(Op::Const { value: 2 }, two_id);
block.push_op(Op::BinOp { op: BinOpKind::Add, lhs: a_id, rhs: two_id }, sum_id);
block.push_op(Op::Store { dst: "out".into(), indices: vec![], value: sum_id, mask: None }, ValueId::new(3));
kernel.body = block;
```

## Crate contents

| Module | Purpose |
|---|---|
| `ir` | SSA-form kernel IR: `Kernel`, `Block`, `Op`, `ValueId`, `Param`, `ParamKind` |
| `dtype` | `DType` enum: F32, F16, BF16, I32, U32, I8, U8, I4, U64, I64, Bool |
| `shape` | `Shape`, `Dim` (Known / ConstExpr), `tile()` constructor |
| `constexpr` | `ConstExpr` — symbolic constants resolved at kernel compile time |
| `error` | `Error` enum and `Result<T>` alias |
| `utils` | Internal helpers (bit manipulation, alignment) |

## API reference

### Core types

| Type | Purpose | Defined in |
|---|---|---|
| `Kernel` | Top-level IR container: params, constexprs, blocks | `src/ir.rs` |
| `Block` | Sequence of `Op`s; owned by a `Kernel` | `src/ir.rs` |
| `Op` | A single IR operation (load, store, binary, reduce, loop, etc.) | `src/ir.rs` |
| `ValueId` | SSA value handle | `src/ir.rs` |
| `BlockId` | Block handle | `src/ir.rs` |
| `VarId` | Loop / block-level variable handle | `src/ir.rs` |
| `Param` | Kernel tensor/scalar parameter descriptor | `src/ir.rs` |
| `ParamKind` | How a param is bound: `Tensor`, `Strided`, or `Scalar` | `src/ir.rs` |
| `KernelMode` | Dispatch shape hint: `Elementwise`, `Reduction`, `Grid3D`, `Tile2D` | `src/ir.rs` |
| `DType` | Numeric type: `F32`, `F16`, `BF16`, `I32`, `U32`, `I8`, `U8`, `I4`, `U64`, `I64`, `Bool` | `src/dtype.rs` |
| `Shape` | Compile-time dimension tracking (array of `Dim`) | `src/shape.rs` |
| `Dim` | A single dimension: `Known(usize)`, `ConstExpr(name)`, or `Any` | `src/shape.rs` |
| `ConstExpr` | Named compile-time constant used in shapes and kernel configs | `src/constexpr.rs` |

### Op variants

The `Op` enum supports these operation categories:

| Category | Op variants |
|---|---|
| Memory | `Load`, `Store`, `VectorLoad`, `VectorStore` |
| Arithmetic | `BinOp` (Add, Sub, Mul, Div, Max, Min, Pow, And, Or, Xor, CmpLt, CmpGt, CmpLe, CmpGe, CmpEq, CmpNe, Shl, Shr), `UnaryOp` (Neg, Recip, Exp, Log, Sqrt, Rsqrt, Abs, Ceil, Floor, Sin, Cos, Erf, Exp2, Log2, Sign, Round, Trunc) |
| Activations | `Activation` (Silu, Gelu, Relu, Tanh, Sigmoid) — separate from UnaryOp |
| Reductions | `Reduce` (Sum, Max, Min, Mean), `Dot`, `StrideReduce` |
| Control flow | `Loop`, `If` |
| Shape ops | `Transpose`, `ExpandDims`, `Reshape`, `Cat`, `Slice`, `Broadcast` |
| Tile ops | `Zeros`, `Splat`, `Arange`, `Cast`, `Select` |
| High-level ML | `FlashAttention`, `SlidingWindowAttention`, `RmsNorm`, `GatedMlp` |
| Misc | `ProgramId`, `Const`, `FusedElementwise`, `InlineMsl` |

### Error types

`Error` — forwarded from all crates that produce or transform IR.
`Result<T>` — `std::result::Result<T, Error>`.

## Dependencies

### Internal

None — `metaltile-core` is the leaf crate. All other crates depend on it.

### External

| Crate | Role |
|---|---|
| `thiserror` | Derive `Error` |
| `smallvec` | Compact storage in IR structures |
| `half` | `f16` / `bf16` constant values |
| `serde` | Serialization for IR dump and manifest |
| `bytemuck` | Safe transmutation of byte buffers to typed slices |
| `bitvec` | Bit-level operations for packed types (I4, Bool) |

## MSRV / platform

No platform gating — pure data structures, no GPU calls.
Rust: nightly (workspace-wide, edition 2024).

## Extending

- **New DType variant:** `src/dtype.rs` — add to the `DType` enum. Update
  `size_bytes()`, `msl_name()`, `is_float()`, and `is_int()`. Run workspace
  tests — most passes and the MSL emitter match on `DType`.

- **New IR op variant:** `src/ir.rs` — add to the `Op` enum. Add a
  `Display` arm. Update `metaltile-codegen` passes that exhaustively match
  `Op` (start with `type_check` and `msl::emit_block`).

- **New shape constructors:** `src/shape.rs` — add free functions or methods
  on `Shape`. If you need a macro, add it to `metaltile-macros`.

- **New error variant:** `src/error.rs` — add to `Error` enum.

- **Tests to update:** `src/ir.rs` tests, pass tests in `metaltile-codegen`,
  `metaltile-macros` tests.

## Related documentation

- [Root README](../../README.md) — project overview and architecture
- [CONTRIBUTING](../../CONTRIBUTING.md) — dev setup, PR process, CI
- [`metaltile-codegen` README](../metaltile-codegen/README.md) — the optimization passes that operate on this IR
- [`metaltile-macros` README](../metaltile-macros/README.md) — how `#[kernel]` produces this IR
- [Crate docs on docs.rs](https://docs.rs/metaltile-core)

## License

Licensed under the [Apache License, Version 2.0](../../LICENSE).
