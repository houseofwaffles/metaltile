# MetalTile

A Rust-embedded DSL for writing Apple Metal GPU kernels. Write tile-level algorithms in Rust, get optimized Metal Shading Language out — verified against, and frequently faster than, hand-tuned MLX.

<!-- TODO(image): replace this HTML table with a side-by-side graphic of the DSL ↔ MSL. -->
<table>
<tr>
<th>Rust DSL — what you write</th>
<th>Metal Shading Language — what you get</th>
</tr>
<tr>
<td>

```rust
#[kernel]
pub fn mt_exp<T>(
    a: Tensor<T>,
    out: Tensor<T>,
) {
    let idx = program_id(0);
    store(out[idx], exp(load(a[idx])));
}
```

</td>
<td>

```cpp
kernel void mt_exp(
    const device float *a [[buffer(0)]],
    device float *out [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    uint v_idx = tid;
    auto v1 = a[v_idx];
    auto v2 = exp(v1);
    out[v_idx] = v2;
}
```

</td>
</tr>
</table>

One generic `#[kernel]` fn becomes a monomorphised `f32` / `f16` / `bfloat16` Metal kernel — the compiler handles thread indexing, dtype lowering, and Metal idioms. Bigger kernels lean on tile-level primitives (`reduce_sum`, `strided_reduce`, `dot`); the codegen emits the simdgroup and threadgroup machinery for you.

## Why MetalTile

| Functionality | Description | Status |
|---|---|---|
| **Write kernels in Rust** | A real `#[kernel]` proc-macro — no raw MSL, no hand-written thread-position arithmetic. | ✅ |
| **Tile-level primitives** | `reduce_sum`, `strided_reduce`, `dot` — say *what* to compute; codegen emits the simdgroup + threadgroup reduction. | ✅ |
| **One source, three dtypes** | Generic `<T>` kernels lower to `f32`, `f16`, and `bfloat16` — native `bfloat` on Metal 3.1+. | ✅ |
| **Optimizing compiler** | A 14-pass pipeline — const-folding, CSE, LICM, fusion, vectorization, and more — sits between the IR and the emitted MSL. | ✅ |
| **Verified against MLX** | Every benched kernel runs side-by-side against the hand-tuned MLX Metal kernel and must match it numerically. | ✅ |
| **Frequently faster than MLX** | A meaningful slice of ops — argmax, small-N RMSNorm, quantized matmul — land 3×+ over MLX on M4 Max. | ✅ |
| **`tile` CLI** | `bench` / `build` / `inspect` / `device` / `snap` / `diff` — one binary for the whole dev loop. | ✅ |
| **Cross-hardware baselines** | Committed `tile bench` snapshots per chip; CI diffs every PR against them. | ✅ |
| **Autotuner** | Per-shape kernel tuning so no performance is left on the table. | 🚧 Planned |
| **Type-level shape algebra** | Tensor shapes checked at compile time. | 🚧 Planned |

## Status

> ⚠️ **Heads Up: Construction Ahead!** Early development — APIs are not yet stable. The core DSL, codegen, and runtime work today; the autotuner and type-level shape algebra are planned. See [`docs/getting-started.md`](docs/getting-started.md) for the crate layout.

## Quick Start

Add the crate:

```toml
[dependencies]
metaltile = "0.1"
```

Write a kernel, dispatch it, read the result:

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
    let b: Vec<u8> = (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect();
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

A real kernel is barely longer. This is `mt_rms_norm_small` — RMSNorm for small head dims, the whole thing:

```rust
#[kernel]
pub fn mt_rms_norm_small<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    let base = rs + tid * 2u32;
    let col = tid * 2u32;
    let x0 = load(x[base]).cast::<f32>();
    let x1 = load(x[base + 1u32]).cast::<f32>();
    let partial_ssq = x0 * x0 + x1 * x1;
    let tg_ssq = reduce_sum(partial_ssq);          // ← tile-level: the codegen emits the simdgroup reduction
    let eps = load(eps_buf[0]);
    let rms = rsqrt(tg_ssq / n + eps);
    store(out[base], (x0 * rms * load(w[col]).cast::<f32>()).cast::<T>());
    store(out[base + 1u32], (x1 * rms * load(w[col + 1u32]).cast::<f32>()).cast::<T>());
}
```

That single `reduce_sum` lowers to a full two-level simdgroup + threadgroup reduction. The result runs at **354% of MLX's hand-tuned `rms` kernel** on an Apple M4 Max (`B=1024 N=64`, f32 — see [`baselines/`](baselines/)).

Full walkthrough and crate layout: [`docs/getting-started.md`](docs/getting-started.md).

## CLI

`tile` drives benchmarking, building, and inspecting kernels:

| Command | What it does |
|---|---|
| `tile bench` | Benchmark every kernel against its MLX reference; reports throughput + a correctness check |
| `tile build` | Compile all kernels to MSL and report errors; `--emit` writes `.metal` / `.metallib` / Swift / IR |
| `tile inspect <kernel>` | Print a kernel's IR and generated MSL (`--ir`, `--pass`, `--stats` for codegen debugging) |
| `tile device` | Show the GPU device, Metal version, and supported feature flags |
| `tile snap` | Save bench results as a regression baseline |
| `tile diff` | Compare bench results against a saved baseline |

**Running it today:** there is no published binary yet — clone the repo and run `cargo run -p metaltile-cli -- <command>`, or `cargo install --path crates/metaltile-cli` for a local `tile`. An installable release will ship once the APIs stabilise. Full flag reference: [`docs/cli.md`](docs/cli.md).

## Supported Operations

| Operation | Status |
|---|---|
| Unary elementwise — `exp`, `log`, `sqrt`, trig/hyperbolic, `erf`, `gelu`, `silu`, `sigmoid`, `relu`, … (40+) | ✅ |
| Binary elementwise — `add`, `sub`, `mul`, `div`, `max`, `min`, `pow`, `logaddexp`, `atan2`, `remainder` | ✅ |
| Fused binary (add+mul), ternary `select`, `copy`, strided copy, `arange` | ✅ |
| Reductions — all-reduce & row-reduce (sum / max / min / prod) | ✅ |
| `softmax`, `logsumexp` | ✅ |
| `rms_norm` (+ small-N variant), `layer_norm` | ✅ |
| `rope` — rotary position embedding | ✅ |
| `argmax`, `scan` (parallel prefix sum), `sort` (bitonic) | ✅ |
| `random` — xorshift / key-hash | ✅ |
| GEMV — dense and masked | ✅ |
| Quantized GEMV / GEMM (`qmv`, `qmm`, int4) | ✅ |
| Affine quantize / dequantize — int3 / 4 / 5 / 6 / 8 | ✅ |
| FP4 quantize / dequantize | ✅ |
| SDPA — vector decode (GQA), two-pass decode | ✅ |
| SDPA — Flash-Attention-2 prefill, incl. simdgroup-MMA fragments | ✅ |
| Tiled GEMM — general matmul (`steel_gemm`) | 🚧 Planned |
| Convolution — 1D / 2D / general | 🚧 Planned |
| FFT | 🚧 Planned |
| Scatter / gather-indexing family | 🚧 Planned |
| FP8 quantization | 🚧 Planned |

Survey of the codebase as of the current `dev`; see [`docs/developing.md`](docs/developing.md) for how kernels are organised.

## Benchmarks

`tile bench` dispatches every MetalTile kernel and its MLX Metal reference on identical buffers, then reports throughput and a numerical-equivalence check. Run the whole suite, or narrow with `--filter`:

```sh
tile bench                   # full suite
tile bench --filter softmax  # one op
```

<!-- TODO(image): screenshot of a `tile bench` run table goes here. -->

See [`docs/cli.md`](docs/cli.md) for `-v` / `-vv` profiling, JSON output, and the `snap` / `diff` regression workflow.

📊 **Full cross-hardware results live in [`baselines/`](baselines/)** — committed `tile bench` snapshots, one canonical file per chip, refreshed as new hardware is benched. CI diffs every PR against the matching baseline.

## Architecture

```
#[kernel] fn  →  metaltile-macros (proc macro)
                          │
                    MetalTile IR  (metaltile-core)
                          │
               metaltile-codegen (optimization passes → MSL)
                          │
                  metaltile-runtime (Metal GPU dispatch)
```

Optimization passes run in order: TypeCheck → ConstFold → AlgebraicSimplify → CopyProp → CSE → LICM → IfConversion → ValueSink → TileLowering → Fusion → Unroll → Schedule → Vectorize → DeadStoreElim.

## Documentation

Full docs live in [`docs/`](docs/README.md):

- [Getting started](docs/getting-started.md) — toolchain, crate layout, build, first kernel.
- [Developing](docs/developing.md) — repo layout, dev loop, and the **kernel-authoring hazards** (a wrong dispatch can freeze the machine).
- [Testing](docs/testing.md) — test layers, CI, and test-infra gaps.
- [CLI](docs/cli.md) — the `tile` binary.
- [Publishing](docs/publishing.md) — the release flow.

## Contributing

Contributions — including AI-assisted ones — are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the issue / PR process and [`docs/developing.md`](docs/developing.md) for the kernel-authoring hazards **before** writing a kernel.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
