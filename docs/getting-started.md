# Getting started

Get a MetalTile checkout building, tested, and emitting a kernel.

## Prerequisites

- **Rust nightly.** The workspace is `edition = 2024` and uses unstable `rustfmt` features; the toolchain is pinned in `rust-toolchain.toml`, so `rustup` installs the right nightly automatically on first build.
- **macOS + Metal** — only needed to *run* kernels on the GPU (`tile bench`, GPU correctness tests). The DSL, codegen passes, and MSL emission build and test on any platform; non-Mac CI exercises everything except GPU dispatch.
- **Xcode command-line tools** (`xcrun metal`) on macOS — the codegen smoke step compiles emitted MSL with the Metal toolchain.

## Clone and set up

```bash
git clone git@github.com:0xClandestine/metaltile.git
cd metaltile
./.github/scripts/setup-dev.sh
```

`setup-dev.sh` verifies the nightly toolchain, the `rustfmt` and `clippy` components, and the optional `typos-cli` / `cargo-llvm-cov` tools.

## First build and test

```bash
make build      # debug build of the whole workspace
make test       # workspace tests — codegen, runtime, GPU correctness (GPU on a Mac)
```

`make` is the canonical entry point — it centralises flags and always passes `--workspace`. See [Developing](developing.md) for the full dev loop and [the CLI reference](cli.md) for the `tile` binary.

## Crate layout

The workspace is seven crates:

| Crate | Description |
|---|---|
| [`metaltile-core`](../crates/metaltile-core/README.md) | IR types, `DType`, `Shape` |
| [`metaltile-macros`](../crates/metaltile-macros/README.md) | the `#[kernel]` proc-macro + body parser |
| [`metaltile-codegen`](../crates/metaltile-codegen/README.md) | MSL lowering + optimization passes |
| [`metaltile-runtime`](../crates/metaltile-runtime/README.md) | Metal dispatch, PSO cache |
| [`metaltile`](../crates/metaltile/README.md) | facade re-exporting all crates |
| [`metaltile-std`](../crates/metaltile-std/README.md) | kernel stdlib, op files, bench types |
| [`metaltile-cli`](../crates/metaltile-cli/README.md) | the `tile` CLI binary |

The compile pipeline: `#[kernel] fn` → `metaltile-macros` parses the body into **MetalTile IR** → `metaltile-codegen` runs the optimization passes and emits **MSL** → `metaltile-runtime` dispatches it on the GPU.

## Your first kernel

A kernel is a Rust function annotated with `#[kernel]`. The proc-macro parses the body into MetalTile IR; the codegen lowers it to Metal Shading Language.

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

Inspect the generated MSL without running anything:

```rust
use metaltile::codegen::msl::MslGenerator;
let msl = MslGenerator::default().generate(&vector_add::kernel_ir())?;
println!("{msl}");
```

or from the CLI: `tile inspect vector_add`.

## Next steps

- [Developing](developing.md) — repo layout, dev loop, and the kernel-authoring hazards. **Read the ⚠️ sections before writing a non-trivial kernel** — one of them is "a wrong dispatch can freeze your machine."
- [Testing](testing.md) — every non-trivial kernel ships a paired GPU correctness test in the same commit; this page explains why and how.
