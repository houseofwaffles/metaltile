# MetalTile

A Rust-embedded DSL for writing Apple Metal GPU kernels. Write tile-level algorithms in Rust, get optimized Metal Shading Language out.

```rust
#[kernel]
pub fn mt_rms_norm<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let ssq = strided_reduce_dot(x, x, rs, 0, re);
    let tg_ssq = reduce_sum(ssq);
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    let n_full = n / (lsize * 4u32);
    for _r in range(0u32, n_full, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let col = base - rs;
        let n0 = load(x[base]).cast::<f32>() * rms * load(w[col]).cast::<f32>();
        let n1 = load(x[base + 1u32]).cast::<f32>() * rms * load(w[col + 1u32]).cast::<f32>();
        let n2 = load(x[base + 2u32]).cast::<f32>() * rms * load(w[col + 2u32]).cast::<f32>();
        let n3 = load(x[base + 3u32]).cast::<f32>() * rms * load(w[col + 3u32]).cast::<f32>();
        store(out[base], n0.cast::<T>());
        store(out[base + 1u32], n1.cast::<T>());
        store(out[base + 2u32], n2.cast::<T>());
        store(out[base + 3u32], n3.cast::<T>());
    }
    for _i in range(rs + n_full * lsize * 4u32 + tid, re, lsize) {
        let ni = load(x[_i]).cast::<f32>() * rms * load(w[_i - rs]).cast::<f32>();
        store(out[_i], ni.cast::<T>());
    }
}
```

This generates ~104% of MLX's hand-tuned `rms` kernel throughput on M4 Max across f32/f16/bfloat16.

## Why MetalTile

- **Write once in Rust, run fast on Apple Silicon.** No raw MSL, no thread-position arithmetic.
- **Tile-level, not thread-level.** `strided_reduce`, `reduce_sum`, `dot` — express what to compute, the compiler handles thread mapping, vectorization, and SIMD-group reductions.
- **Verified against MLX.** Every kernel is benchmarked and numerically compared against the corresponding MLX Metal kernel. 139/139 ops correct, avg 110% of MLX throughput on M4 Max.
- **All three float dtypes.** `f32`, `f16`, and `bfloat16` work identically — native `bfloat` emitted on Metal 3.1+.
- **Rust-only correctness checking.** Every kernel is verified against CPU-computed reference values without a Mac.

## Status

Early development — APIs are not yet stable. Core DSL works; autotuner and type-level shape algebra are planned for v0.2.

| Crate | Description |
|---|---|
| [`metaltile-core`](crates/metaltile-core/README.md) | IR types, DType, Shape |
| [`metaltile-macros`](crates/metaltile-macros/README.md) | `#[kernel]` proc macro |
| [`metaltile-codegen`](crates/metaltile-codegen/README.md) | MSL lowering + 14 opt passes |
| [`metaltile-runtime`](crates/metaltile-runtime/README.md) | Metal dispatch, PSO cache |
| [`metaltile`](crates/metaltile/README.md) | Facade re-exporting all crates |
| [`metaltile-std`](crates/metaltile-std/README.md) | Kernel stdlib, op files, bench types |
| [`metaltile-cli`](crates/metaltile-cli/README.md) | `tile` CLI binary |

## Quick Start

```toml
[dependencies]
metaltile = "0.1"
```

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

To inspect the generated MSL directly:

```rust
use metaltile::codegen::msl::MslGenerator;

let msl = MslGenerator::default().generate(&vector_add::kernel_ir())?;
println!("{msl}");
```

## Supported Operations

22 operation categories benchmarked against MLX:

**Elementwise**: unary (exp, log, sqrt, sin, cos, erf, sigmoid, silu, gelu, relu, …), binary (add, mul, sub, div, max, min, pow, logaddexp), binary_two (fused add+mul), ternary (select), copy, arange

**Reductions**: all-reduce sum/max/min, row-reduce sum/max/min, logsumexp, softmax, rms-norm, layer-norm

**Matrix**: GEMV, masked GEMV, SDPA vector decode

**Misc**: RoPE, scan (parallel prefix sum), arg-reduce (argmax), sort (bitonic), random (xorshift32), quantized GeMV (int4), fp4 quantize/dequantize, strided copy (non-contiguous tensors)

## CLI

Install the `tile` binary:

```sh
cargo install --path crates/metaltile-cli
```

```
tile bench                      # full benchmark suite vs MLX
tile bench --filter softmax     # narrow to one op
tile build                      # compile all kernels, report errors
tile inspect --kernel mt_rms_norm  # print IR and generated MSL
tile profile                     # occupancy & register analysis for all kernels
tile profile <kernel> --sweep    # per-threadgroup-size breakdown
tile device                     # show GPU info and supported features
tile snap -o baseline.json      # save bench results as a regression baseline
tile diff baseline.json         # compare current results to baseline
```

## Benchmarks

Run against MLX Metal kernels on M4 Max:

```sh
tile bench
```

Selected results (M4 Max, higher = better vs MLX):

| Op | MT% of MLX |
|---|---|
| softmax f32 | ~105% |
| rms_norm f16 | ~104% |
| all_reduce f32 | ~100% |
| gemv f16 | ~100% |
| argmax f32 | **206%** |
| scan f32 | ~104% |

## Architecture

```
#[kernel] fn  →  metaltile-macros (proc macro)
                          │
                    MetalTile IR  (metaltile-core)
                          │
               metaltile-codegen (14 opt passes → MSL)
                 │
         metaltile-runtime
         (Metal GPU dispatch)
```

Optimization passes: TypeCheck → ConstFold → AlgebraicSimplify → CopyProp → CSE → LICM → IfConversion → ValueSink → TileLowering → Fusion → Unroll → Schedule → Vectorize → DeadStoreElim.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
