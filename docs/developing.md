# Developing

Repo layout, the dev loop, and — most importantly — the **kernel-authoring hazards** that cause silent or catastrophic failure. If you only read one section, read [Kernel-authoring hazards](#kernel-authoring-hazards).

## Glossary

Short terms used throughout the docs and the codebase:

| Term | Meaning |
|---|---|
| **DSL** | Domain-specific language — the Rust-embedded language you write inside a `#[kernel]` function. |
| **IR** | Intermediate representation — MetalTile's typed kernel graph (`Kernel` / `Op`), produced by the proc-macro and consumed by the codegen. |
| **MSL** | Metal Shading Language — Apple's C++-based GPU shader language, and the codegen's final output. |
| **Kernel** | A GPU compute function; in this repo, a Rust `fn` annotated with `#[kernel]`. |
| **Threadgroup** | A block of GPU threads that share threadgroup memory and can synchronise with a barrier. |
| **Simdgroup** | 32 threads ("lanes") within a threadgroup that execute in lockstep — Apple's SIMD width. Cross-lane ops are the `simd_*` intrinsics. |
| **TPG** | Threads per threadgroup — the threadgroup dimension of a dispatch. Must be a multiple of 32 for reduction kernels. |
| **PSO** | Pipeline state object — a compiled Metal compute pipeline. The runtime caches these so a kernel compiles once. |
| **Pass** | One transformation stage in the codegen pipeline (e.g. `ConstFold`, `Vectorize`). |

## Repo layout

The workspace is seven crates, layered from the shared data model up to the CLI binary:

| Crate | What it is |
|---|---|
| [`metaltile-core`](../crates/metaltile-core/README.md) | The shared data model — the `Kernel` / `Op` IR types, `DType`, `Shape`, and error types. Pure data structures with no logic, so every layer above can speak the same vocabulary. Used by **every** other crate. |
| [`metaltile-macros`](../crates/metaltile-macros/README.md) | The compiler front end — the `#[kernel]` proc-macro and its body parser, which turn a Rust function into MetalTile IR at compile time. Owns the DSL grammar and its compile-error diagnostics. Used by **`metaltile`** (re-exported as the public `#[kernel]` attribute). |
| [`metaltile-codegen`](../crates/metaltile-codegen/README.md) | The optimizing compiler — lowers MetalTile IR through the [14-pass pipeline](#debugging-a-kernel) and emits MSL. The largest crate; owns every pass and the MSL emitter. Used by **`metaltile-runtime`**, **`metaltile-std`**, and **`metaltile-cli`**. |
| [`metaltile-runtime`](../crates/metaltile-runtime/README.md) | The GPU execution layer — compiles emitted MSL into Metal PSOs (with a PSO cache) and dispatches kernels through `Context`. Owns all Metal-framework / `objc2` interop. Used by **`metaltile`** and **`metaltile-std`**. |
| [`metaltile`](../crates/metaltile/README.md) | The facade — re-exports `core`, `macros`, `codegen`, and `runtime` behind one `prelude` so downstream code and external users depend on a single crate. No logic of its own. Used by **`metaltile-std`** and **`metaltile-cli`**. |
| [`metaltile-std`](../crates/metaltile-std/README.md) | The kernel standard library — the actual `#[kernel]` definitions (`mlx/`, `ffai/`), their `BenchSpec`s, the bench harness, and the GPU correctness tests. Where new kernels land. Used by **`metaltile-cli`**. |
| [`metaltile-cli`](../crates/metaltile-cli/README.md) | The `tile` binary — `bench` / `build` / `inspect` / `device` / `snap` / `diff`, the developer-facing entry point. The top of the dependency graph; nothing depends on it. |

The compile pipeline: `#[kernel] fn` → `metaltile-macros` parses the body into **MetalTile IR** → `metaltile-codegen` runs the optimization passes and emits **MSL** → `metaltile-runtime` dispatches it on the GPU.

## Dev loop

```bash
make build       # debug build
make test        # workspace tests — codegen, runtime, GPU correctness (GPU on a Mac)
make clippy      # lint with -D warnings
make fmt         # format
make fmt-check   # check formatting without writing
make typos       # spell-check
make coverage    # HTML coverage report (needs cargo-llvm-cov)
make bench       # full benchmark suite vs MLX (macOS + Metal)
make clean       # remove target/
```

Prefer `make` over raw `cargo` — it centralises flags and always passes `--workspace`. See [the CLI reference](cli.md) for the `tile` binary.

## Branching model

| Branch | Purpose |
|---|---|
| `main` | Stable releases only. Commits here are tagged (`v0.1.0`, …). |
| `dev`  | Integration branch for the next release. Feature PRs merge here. |
| `feat/*` `fix/*` `perf/*` `docs/*` | Short-lived topic branches cut from `dev`. |

Cut a topic branch from `dev`, PR back into `dev`, squash- or rebase-merge after review + green CI. See [Publishing](publishing.md) for the `dev` → `main` release flow.

## Conventional commits

PR titles follow [Conventional Commits](https://www.conventionalcommits.org/) so `.github/workflows/auto-label.yml` can categorise them for release notes and `.github/workflows/pr.yml` can validate the format:

```
feat: add softmax vector path for small N
fix(codegen): correct version gate for half2 stores
perf(runtime): cache PSO lookups by function signature
docs: update CLI install instructions
test(core): add scan correctness test
chore: bump nightly toolchain
```

Add `!` for breaking changes (`feat!: …`) and describe them in the PR body.

## Debugging a kernel

| Want | Command |
|---|---|
| IR before any passes | `tile inspect <kernel> --ir` |
| Final MSL | `tile inspect <kernel>` |
| IR after one pass | `tile inspect <kernel> --pass <name>` (or `--pass all`) |
| Per-pass op-count deltas | `tile inspect <kernel> --stats` |
| Which pass is slow | `tile build --time-passes --filter <kernel>` |
| Emit every kernel's MSL | `tile build --emit all -o <dir>` |

When a kernel regresses, `--stats` before/after the change shows which pass changed the op count; `--pass all` dumps the IR at every stage.

## Kernel-authoring hazards

These are not style preferences. Each one has bitten this project; each one fails *silently* (wrong output, no error) or *catastrophically* (frozen machine). Read all of them before writing a kernel.

### ⚠️ A wrong dispatch can freeze the machine

Metal compute dispatches are **non-preemptive** — once a threadgroup starts, the GPU runs it to completion. An infinite loop inside a kernel never yields: the WindowServer compositor starves of GPU time, the screen locks at the last frame, and a **hard power-cycle is the only recovery**.

The concrete trap: reduction-mode kernels compute the simdgroup count as `n_simd = lsize / 32` (integer division). A loop strided by `n_simd` — `for _t in range(sg, n_kv, n_simd)` — becomes an **infinite GPU loop** when `n_simd == 0`, i.e. when the kernel is dispatched with **fewer than 32 threads per threadgroup**. A 4-thread dispatch of a 1024-thread kernel once froze a dev machine for a full day. The kernel was correct; the *dispatch geometry* was not.

Rules for any kernel that uses `simd_*` / `threadgroup_*`:

- Threads-per-threadgroup **must be a multiple of 32** and **≥ 32** (one full simdgroup).
- The dispatch geometry is part of the kernel's contract — derive it from the kernel's invariants, never from an unrelated "number of elements" count.
- GPU correctness tests and `BenchSpec`s set the threadgroup size from the kernel side, so they are safe. The danger is any *consumer* that turns a caller-supplied dimension into a dispatch shape — guard those.

### ⚠️ Pick the right dispatch mode

`Context::dispatch_with_grid(kernel, buffers, constexprs, grid_xyz, tg_xyz)` calls `dispatchThreadgroups`. **`grid_xyz` is counted in threadgroups, not threads** — total threads = `grid.{x·y·z} · tg.{x·y·z}`.

- **Grid3D** — one thread per output element, no cross-thread cooperation. `program_id::<i>()` lowers to the **thread** index.
  ```rust
  #[kernel] fn mul<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
      let i = program_id::<0>();
      store(out[i], load(a[i]) * load(b[i]));
  }
  // dispatch: grid=[1,1,1] tg=[N,1,1]   (or grid=[ceil(N/TPG),1,1] tg=[TPG,1,1])
  ```
- **Reduction** — uses `simd_*` / `threadgroup_*`. `program_id::<i>()` lowers to the **threadgroup** index; threads within a group cooperate.
  ```rust
  #[kernel] fn rms_norm<T>(x: Tensor<T>, /* … */) {
      let row = program_id::<0>();   // = threadgroup index, one TG per row
      // … reduce_sum across the threadgroup …
  }
  // dispatch: grid=[rows,1,1] tg=[TPG,1,1]
  ```

**Wrong:** `grid=[N,1,1] tg=[N,1,1]` for a Grid3D kernel — that is `N²` threads in flight, most with garbage indices. The product `grid · tg` must equal exactly the thread count the kernel expects.

### ⚠️ Inner `macro_rules!` silently empties the kernel body

To share a body across non-generic variants (bit widths, fixed group sizes), wrap the **entire `#[kernel] fn` declaration** in an outer `macro_rules!` — **never** put a `macro_rules!` call *inside* a `#[kernel]` body. The proc-macro does not expand inner declarative macros: it sees the call as opaque tokens, drops it, and emits a kernel with **no body**. `xcrun metal` compiles it fine and it ships **all-zeros output**. The proc-macro now rejects the inner-body shape with a compile error — heed it rather than working around it.

**Right** — wrap the whole declaration; the compiler expands the outer macro before the `#[kernel]` proc-macro runs, so the body parser sees concrete tokens with `$bits` already substituted:

```rust
macro_rules! dequant_gather_kernel {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(/* params */) {
            let bit_off = d * $bits;   // $bits already substituted
            // …
        }
        inventory::submit! { /* BenchSpec */ };
    };
}
dequant_gather_kernel!(dequant_gather_int4, 4u32, "int4");
```

**Wrong** — inner macro inside the body → empty MSL (now a compile error):

```rust
macro_rules! body { ($bits:literal) => { /* … */ }; }
#[kernel] pub fn dequant_gather_int4<T>(/* … */) { body!(4); }
```

Canonical reference: `crates/metaltile-std/src/ffai/dequant_gather.rs`. For hand-unrolled tree reductions, replace `*_step!` macros with a DSL `for` loop over the halving strides — identical MSL, survives the proc-macro.

### ⚠️ Empty-body MSL also slips through pass ordering

The macro trap above is one way the codegen emits a kernel with a valid function/loop *header* but no *body*. The other is **pass ordering**: a pass eliminates a loop body but leaves the loop header, or a `Const` a later pass needs is still rolled inside a `BinOp` so the trip count is invisible. The result is `for (…) { }` — an empty loop — and again the kernel ships all-zeros output that `xcrun metal` accepts.

Invariants for any codegen pass you write or touch:

- A pass that rewrites blocks must walk **both `kernel.body` and every entry in `kernel.blocks`** — `kernel.body` is the entry block, *not* part of the map.
- A pass that removes a loop body must also remove the loop header.
- A pass that consumes a `Const` must run after the pass that produces it.

Detection — emit every kernel, then scan for empty bodies:

```bash
tile build --emit all -o /tmp/mt-smoke
awk '
  /for \(.*\) \{$/               { f=1; fn=FILENAME; l=FNR; next }
  f && /^[[:space:]]*\}$/        { print fn":"l": empty for-loop body"; f=0; next }
  f                              { f=0 }
  /^kernel void [A-Za-z_0-9]+\(/ { k=1; fn=FILENAME; l=FNR; next }
  k && /^\{$/                    { next }
  k && /^\}$/                    { print fn":"l": empty kernel body"; k=0; next }
  k                              { k=0 }
' /tmp/mt-smoke/Resources/kernels/*.metal
```

Empty output = clean. Any hit = ship-stopper. **Neither `xcrun metal` nor MSL snapshots catch an empty body** — only a GPU correctness test does (see [Testing](testing.md)).

### Document the dispatch contract: `## DISPATCH INVARIANTS`

A reduction kernel's threadgroup geometry is part of its API, but the kernel cannot enforce it at runtime. Make the contract explicit in the kernel's `.rs` doc comment so anyone dispatching it has something to verify against:

```rust
//! ## DISPATCH INVARIANTS
//!
//! - **TPG: 1024 threads** (32 simdgroups × 32 lanes).
//! - **Grid: 1 threadgroup per row** (1D grid, program_id<0> = row).
//! - **head_dim == 128.** Each lane owns 4 consecutive elements; loads are
//!   unconditional — other head dims read out of bounds / pin the GPU.
```

This is a young convention — most kernels don't carry a block yet. Add one whenever you write or touch a reduction kernel; four lines, and it is the only place the geometry contract is written down.

### ⚠️ Perf numbers that look "too flat to be physical"

A latency that does not scale with input size is almost always the *harness* contaminating the measurement, not an exceptional kernel. Before publishing a bench number, confirm the curve has the shape physics predicts. Two harness fixes belong in every perf measurement:

- **Use resident buffers for inputs constant across iterations** — otherwise the bench re-uploads them every iteration and you measure host→GPU traffic, not the kernel.
- **Dummy-dispatch once to warm the GPU clock** — cold DVFS gives the first measured shape a ~2× bandwidth deficit.

A "flat ~215 µs regardless of context length" sliding-window-attention claim turned out to be upload overhead drowning the kernel; switching to resident buffers dropped the floor to ~98 µs and revealed the real curve.

## Kernel-writing philosophy

- **Improve the compiler, don't hand-write MSL.** If the DSL can't express a pattern, extend the codegen (body parser → IR → MSL emit). Don't bypass it.
- **One generic `<T>` kernel** beats five precision-specific copies — `f32` / `f16` / `bf16` all flow through the same `#[kernel] fn`.
- **Every non-trivial kernel ships a GPU correctness test in the same commit.** See [Testing](testing.md) — it is the only layer that catches the empty-body and numeric-correctness bugs above.
- **Dispatch safety is the caller's job — for now.** Nothing in the kernel itself stops a degenerate dispatch from hanging the GPU (see [the freeze hazard](#kernel-authoring-hazards)). A future direction is a runtime-checked dispatch path — kernel-level escape hatches that trap a runaway threadgroup before it pins the device — but until that lands, the consumer constructing the dispatch geometry is the only line of defense.
