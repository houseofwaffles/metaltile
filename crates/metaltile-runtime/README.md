# metaltile-runtime

Apple Metal runtime dispatch for MetalTile GPU kernels.
Manages Metal devices, compiles generated MSL into pipeline state objects,
dispatches compute kernels, and returns output buffers to the host.

This crate is the bottom of the MetalTile stack — it is the only crate
that links against Apple's Metal framework, and all kernel execution
ultimately flows through its `Context` type.

## Position in the pipeline

```
metaltile-codegen (MSL) ──► metaltile-runtime (this crate) ──► host results
                                    │
                              Metal framework
                              (MTLDevice, MTLCommandQueue,
                               MTLComputePipelineState)
```

`metaltile-runtime` receives MSL source from `metaltile-codegen`, compiles
it into a Metal compute pipeline, dispatches with user-provided buffers, and
returns `DispatchResult` with timing and output data. The crate also owns
the autotuner and its persistent disk cache.

## Quick start

```rust
#[cfg(target_os = "macos")]
fn example() -> Result<(), Box<dyn std::error::Error>> {
    use metaltile_runtime::Context;
    use metaltile_core::ir::Kernel;

    let ctx = Context::new()?;

    // Build IR programmatically or get it from a #[kernel] expansion
    let kernel = my_kernel::kernel_ir_for();

    // Compile MSL → Metal PSO → dispatch
    let result = ctx.dispatch(&kernel)?;
    println!("kernel ran in {:.1} µs", result.elapsed_us);
    Ok(())
}
```

Most users don't call `metaltile-runtime` directly — they use the facade's
`kernel::launch(&ctx).input(...).dispatch()` builder, which delegates to
`Context::dispatch_with_buffers`.

## Crate contents

| Module | Purpose |
|---|---|
| `context` | `Context` type: device management, PSO compilation, `dispatch` / `dispatch_with_buffers` / `dispatch_with_options` |
| `autotune` | Persistent autotuner: `TuneConfig`, `ShapeBucket`, `TuneCache`, on-disk cache at `~/.cache/metaltile/` |
| `buffer` | Typed buffer descriptors: `GpuBuffer` (GPU-side metadata) and `HostData` (host-side data ready for upload) |
| `error` | `MetalTileError` enum covering all runtime failure modes |

## API reference

### Lifecycle

```
Context::new() → MslGenerator::generate(kernel) → Metal library compile
  → build PSO → encode + dispatch command buffer → wait → read DispatchResult
```

1. **Create a `Context`.** Acquires the system default Metal device and
   command queue (macOS), or returns a no-op context on other platforms.
2. **Generate MSL.** The context calls `MslGenerator` internally — you pass
   IR, not MSL text.
3. **Compile and dispatch.** MSL is compiled to a `MTLComputePipelineState`,
   cached by kernel hash, then dispatched with your buffers.
4. **Read results.** `DispatchResult` contains output buffers keyed by
   parameter name, plus elapsed time and GFLOPS.

### Key types

| Type | Purpose |
|---|---|
| `Context` | GPU device handle, command queue, PSO cache. Created once per process. |
| `DispatchResult` | Timings (`elapsed_us`, `gflops`) and output buffer contents (`outputs: BTreeMap<String, Vec<u8>>`). |
| `MetalTileError` | All error variants: `Metal`, `NoDevice`, `Compilation`, `Buffer`, `Dispatch`, `Autotune`, `Core`, `Codegen`, `UnsupportedPlatform`. |
| `GridSpec` | Dispatch grid sizing: `Elementwise`, `Reduction`, `Grid3D`. |
| `GpuBuffer` | Buffer metadata: dtype, shape, element count, byte size. |
| `HostData` | Host-side data with dtype and shape, ready for GPU upload. |

### Autotuner

The autotuner searches for the best kernel schedule configuration for each
(chip, shape bucket) pair and persists results to disk.

**Cache location:** `~/.cache/metaltile/tuning_cache.json` (single file per machine)

**Search strategy** (planned; currently returns defaults):
1. Coarse grid over config space → pick top 3 candidates.
2. Fine grid around each candidate → pick best.
3. Store winner to the per-chip, per-kernel cache file.

**Config fields** (`TuneConfig`):

| Field | Purpose |
|---|---|
| `tile_dims` | Tile dimensions (M, N, K for matmul-style ops) |
| `threads` | Threads per threadgroup (x, y, z) |
| `unroll_factor` | Inner loop unroll depth |
| `use_simd_matrix` | Whether to use SIMD matrix multiply |
| `use_async_copy` | Whether to use async copy for streaming |

## Dependencies

### Internal

| Crate | Role in this crate |
|---|---|
| `metaltile-core` | Reads kernel IR for param shapes, dtypes, and dispatch metadata |
| `metaltile-codegen` | Calls `MslGenerator` to lower IR → MSL before Metal compilation |

### External

| Crate | Role |
|---|---|
| `objc2` | Objective-C runtime bindings (macOS only) |
| `objc2-metal` | Metal framework bindings: `MTLDevice`, `MTLCommandQueue`, `MTLLibrary`, `MTLComputePipelineState`, `MTLBuffer`, etc. |
| `objc2-foundation` | Foundation types (`NSString`) for Metal API calls |
| `parking_lot` | `Mutex` for thread-safe PSO cache |
| `serde` / `serde_json` | Serialize/deserialize autotune cache to disk |
| `thiserror` | Derive `Error` for `MetalTileError` |

## MSRV / platform

**macOS only.** All Metal API calls are `#[cfg(target_os = "macos")]`-gated.
On non-macOS platforms, `Context` returns a no-op stub — `has_gpu()` returns
`false` and `dispatch` returns an empty `DispatchResult` without error.

Rust: nightly (workspace-wide, for edition 2024).

## Extending

- **New Metal feature query:** `src/context.rs` — add a device capability
  check (e.g., `supportsRayTracing()`) to the `Context` struct.
- **New autotuner config field:** `src/autotune.rs` — add to `TuneConfig`,
  update the search logic, and bump the cache schema version if needed.
- **New dispatch mode:** `src/context.rs` — add a `dispatch_with_*` method
  (e.g., indirect dispatch, tile dispatch).
- **New buffer type:** `src/buffer.rs` — add a descriptor struct for the
  new allocation pattern.
- **New error variant:** `src/error.rs` — add to `MetalTileError` enum.
- **Tests to update:** Integration tests require macOS + Metal. Run
  `make test` on a Mac to exercise the full dispatch path.

## Related documentation

- [Root README](../../README.md) — project overview and architecture
- [CONTRIBUTING](../../CONTRIBUTING.md) — dev setup, PR process, CI
- [`metaltile-codegen` README](../metaltile-codegen/README.md) — the MSL generator this crate calls before compiling
- [`metaltile-core` README](../metaltile-core/README.md) — the IR types this crate reads for dispatch metadata
- [Crate docs on docs.rs](https://docs.rs/metaltile-runtime)

## License

Licensed under the [Apache License, Version 2.0](../../LICENSE).
