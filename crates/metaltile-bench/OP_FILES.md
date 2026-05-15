# Writing Op Files

## Process

1. **Read the `.metal` file first.** Before writing a single line of Rust, open
   `src/metal/<op>.metal` and extract:
   - The exact kernel function name(s) you will call as the reference.
   - The parameter list and buffer slot order (MLX uses varied signatures; do not guess).
   - Any Metal function constants (`[[function_constant(N)]]`) — you need
     `runner.compile_with_bool_constants` instead of `runner.compile`.
   - The dispatch dimensions: grid shape `[GX, GY, GZ]` and threadgroup size `[TPG, 1, 1]`
     the kernel was designed for.
   - The algorithm: what reduction strategy, how many passes, what work per thread.

2. **Implement the same algorithm in the `#[kernel]` DSL.** The MT kernel must
   replicate the MLX algorithm — not a functionally equivalent but structurally
   different one.  Correctness against a CPU oracle is necessary but not sufficient;
   the implementation must also match the MLX dispatch pattern (KernelMode, grid
   sizing, work-per-thread).

3. **Benchmark every dtype variant** that the MLX kernel supports (use
   `FLOAT_DTYPES` = `[f32, f16, bf16]` unless the metal file instantiates fewer).

---

## Getting performance to par with the reference

When MT% is below ~95%, work through this loop:

### 1. Run the benchmark

```sh
cargo run -p metaltile-bench --release -- --filter <op>
```

The output shows `ref GB/s`, `mt GB/s`, and `MT%` per dtype.  A number below
100% means the generated MSL is leaving performance on the table.

### 2. Dump the generated MSL

```sh
cargo run -p metaltile-bench --bin dump_msl --release -- --filter <op>
```

This writes the generated MSL for every dtype to stdout (or `--dir <path>` to
files).  You now have two things to compare side-by-side:

- **Generated MSL** — what the DSL produced
- **Reference metal** — `src/metal/<op>.metal`

### 3. Compare generated MSL vs the reference

Read both files and look for structural differences.  Common gaps:

| What the reference does | What to check in the generated MSL |
|---|---|
| Vectorized loads (`float4`, `half4`) | Does `VectorizePass` emit `float4` loads? |
| N_READS > 1 per thread (loop unroll) | Does the stride-reduce loop unroll? |
| `simd_sum` / `simd_max` reduction | Is `Op::SimdReduce` emitted and correct? |
| Threadgroup tile + barrier | Are `threadgroup_barrier` calls present? |
| Work computed in `float`, stored as `T` | Are casts explicit in the output? |
| Inline constant folding (e.g. `inv = 1/n`) | Does `ConstFold` eliminate the division? |

If the generated MSL is structurally equivalent but slower, the issue is likely
a missing codegen optimization (vectorization, loop unroll, or instruction
selection).  If the generated MSL is structurally different, the DSL kernel
needs to be revised to match the algorithm more closely.

### 4. Fix the gap

- **DSL kernel wrong**: edit `<op>.rs` — adjust the algorithm, dispatch pattern,
  or work-per-thread to match the reference.
- **Codegen missing a feature**: add it to the relevant pass in
  `metaltile-codegen/src/passes/` (vectorize, const_fold, fusion, schedule) or
  the MSL emitter in `metaltile-codegen/src/msl.rs`.
- **IR op missing**: add it to `metaltile-core/src/ir.rs`, wire up the body
  parser in `metaltile-macros`, add an MSL emitter stub, and an interpreter stub.

After each fix, re-run the benchmark and re-dump MSL to verify the change had
the intended effect.  Iterate until MT% ≥ 95% across all dtypes.

---

## Mandatory pre-flight checklist

Before writing `<op>.rs`, answer every question from the metal file:

- [ ] What is the exact MLX kernel function name? (grep `kernel void` in the file)
- [ ] Does it use `[[function_constant(N)]]`? Which slot numbers and boolean values
      are needed for the variant you are benchmarking?
- [ ] What are the buffer slots in order? Map them to your `runner.bench` call.
- [ ] What grid shape does MLX use? Match it in both ref and MT dispatch.
- [ ] What is the threadgroup size for the reference? (may differ from MT)
- [ ] What is the memory traffic per element? (reads + writes × elem_bytes)
- [ ] Is it row-parallel (`B × N` shape), flat (`N` shape), or 3D?

---

## File structure

```
src/ops/<op>.rs
```

```rust
//! <Op> benchmark — #[kernel] DSL vs MLX metal/<op>.metal
//!
//! MLX kernel: <exact_function_name> (from <op>.metal, line N)
//!   Params: (buf0, buf1, ...) — list them in slot order
//!   Grid: [GX, GY, GZ] × [TPG, 1, 1]
//!   Algorithm: <describe what the MLX kernel does: passes, work-per-thread,
//!              reduction strategy — copy from or paraphrase the metal source>
//!
//! MetalTile: mt_<op> — same algorithm via #[kernel] DSL.
//!   KernelMode::<Elementwise|Reduction|Grid3D>

use metaltile::kernel;
// Only import KernelMode when the kernel is not Elementwise:
use metaltile::core::ir::KernelMode;
use metaltile_codegen::msl::MslGenerator;

use crate::{
    ops::{
        DType, FLOAT_DTYPES, OpBench, OpResult,
        buffer_typed, zeros_typed, run_typed_once,
        check_equiv, quantize_roundtrip,
        dtype_tol,        // elementwise ops
        // dtype_tol_reduce, // reductions — use this instead for reduce/norm/softmax
        dtype_label, elem_bytes, mlx_tname, to_gbps,
    },
    runner::GpuRunner,
};

// Metal file must exist at this path before this file compiles.
static SRC: &str = include_str!("../metal/<op>.metal");

const BENCH: OpBench = OpBench::new("<op_name>", "GB/s");
// For 1-D flat ops:
const SHAPES: &[usize] = &[64 * 1024 * 1024];
// For row-parallel ops (B rows × N cols):
// const SHAPES: &[(usize, usize)] = &[(1_024, 4_096)];
const N_CHECK: usize = 2_048;   // correctness only; keep small (256–4096)
const TPG: usize = 256;         // match the MT kernel threadgroup size

// ── Kernel ────────────────────────────────────────────────────────────────────

#[kernel]
pub fn mt_<op><T>(a: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let idx = program_id(0);
    store(out[idx], /* ... */);
}

// ── MSL generation helper ─────────────────────────────────────────────────────

fn <op>_msl_for(dt: DType) -> String {
    // Set KernelMode only when the kernel is NOT Elementwise:
    // let mut k = mt_<op>::kernel_ir_for(dt);
    // k.mode = KernelMode::Reduction;
    // MslGenerator::default().generate(&k).unwrap_or_else(...)
    MslGenerator::default()
        .generate(&mt_<op>::kernel_ir_for(dt))
        .unwrap_or_else(|e| { eprintln!("[<op> {dt:?}]: {e}"); String::new() })
}

// ── Bench ─────────────────────────────────────────────────────────────────────

pub fn bench_<op>(runner: &GpuRunner) -> Vec<OpResult> {
    FLOAT_DTYPES.iter().flat_map(|&dt| bench_for(runner, dt)).collect()
}

fn bench_for(runner: &GpuRunner, dt: DType) -> Vec<OpResult> {
    let tn     = mlx_tname(dt);
    let dlabel = dtype_label(dt);
    let eb     = elem_bytes(dt);
    let tol    = dtype_tol(dt); // or dtype_tol_reduce(dt)

    let msl = <op>_msl_for(dt);
    let mk  = runner.compile(&msl, "mt_<op>").ok();

    // Some MLX kernels require bool function constants.  When they do:
    //   let rk = runner.compile_with_bool_constants(SRC, &format!("...{tn}"),
    //                &[(SLOT, true), ...]).ok();
    // Otherwise:
    let rk  = runner.compile(SRC, &format!("<mlx_kernel_name_{}>", tn)).ok();

    // ── Correctness (compare MT vs CPU reference or MT vs MLX reference) ──────
    //
    // RULE: every op that produces an `BENCH.implemented` result MUST have a
    // correctness check.  OpBench panics if equiv is not provided.
    //
    // Strategy A — MT vs CPU (preferred for simple elementwise/reduction ops):
    let equiv: Option<crate::ops::EquivResult> = mk.as_ref().map(|mk| {
        let a_f32: Vec<f32> = (0..N_CHECK).map(|i| i as f32 * 0.001).collect();
        let a_q   = quantize_roundtrip(&a_f32, dt);
        let cpu_ref: Vec<f32> = a_q.iter().map(|&x| /* scalar op on x */ x).collect();
        let a_buf   = buffer_typed(runner, &a_f32, dt);
        let out_buf = zeros_typed(runner, N_CHECK, dt);
        let ns      = runner.buffer_u32(N_CHECK as u32);
        let mt_vals = run_typed_once(runner, mk, &[&a_buf, &out_buf, &ns],
                                     &out_buf, N_CHECK,
                                     [N_CHECK.div_ceil(TPG), 1, 1], [TPG, 1, 1], dt);
        check_equiv(&cpu_ref, &mt_vals, tol)
    });
    //
    // Strategy B — MT vs MLX reference (when CPU oracle is complex or fp-order-sensitive):
    // let equiv = mk.as_ref().and_then(|mk| rk.as_ref().map(|rk| {
    //     /* run both on same input, check_equiv their outputs */
    // }));

    // ── Performance ──────────────────────────────────────────────────────────
    let mut results = Vec::new();
    for &n in SHAPES {
        // bytes = actual memory traffic per kernel invocation.
        // Elementwise read+write: n * eb * 2
        // Read-only (e.g. argmax):  n * eb
        // Two inputs + one output:  n * eb * 3
        let bytes = (n * eb * 2) as f64;

        let a_perf = buffer_typed(runner, &vec![1.0f32; n], dt);

        // Reference perf — dispatch must exactly match the MLX kernel's expected grid.
        let ref_perf = rk.as_ref().and_then(|rk| {
            let out = zeros_typed(runner, n, dt);
            let ns  = runner.buffer_u32(n as u32);
            to_gbps(&runner.bench(rk, &[&a_perf, &out, &ns],
                                  [n.div_ceil(TPG), 1, 1], [TPG, 1, 1], 3, 10), bytes)
        });

        // MT perf
        let mt_perf = mk.as_ref().and_then(|mk| {
            let out = zeros_typed(runner, n, dt);
            let ns  = runner.buffer_u32(n as u32);
            to_gbps(&runner.bench(mk, &[&a_perf, &out, &ns],
                                  [n.div_ceil(TPG), 1, 1], [TPG, 1, 1], 3, 10), bytes)
        });

        let shape = format!("N={n} {dlabel}");
        // For row-parallel ops: format!("B={b} N={n} {dlabel}")
        results.push(match mt_perf {
            Some(p) => BENCH.implemented(shape, ref_perf, p, equiv.clone().unwrap()),
            None    => BENCH.nyi(shape, ref_perf),
        });
    }
    results
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msl_generates_for_all_dtypes() {
        for &dt in FLOAT_DTYPES {
            let msl = <op>_msl_for(dt);
            assert!(!msl.trim().is_empty(), "MSL empty for {dt:?}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn kernels_compile() {
        let Ok(runner) = GpuRunner::new() else { return; };
        for &dt in FLOAT_DTYPES {
            let msl = <op>_msl_for(dt);
            runner.compile(&msl, "mt_<op>").unwrap();
        }
    }
}
```

---

## KernelMode

The default is `Elementwise`.  Set it before generating MSL when the MLX kernel
uses a different dispatch pattern:

| MLX dispatch pattern | KernelMode to use |
|---|---|
| One threadgroup per row: `[B, 1, 1] × [256, 1, 1]` | `Reduction` |
| Flat all-reduce: `[1, 1, 1] × [256, 1, 1]` | `Reduction` |
| 3-axis: `[GX, GY, GZ] × [1, 1, 1]` | `Grid3D` |
| One thread per element: `[N/TPG, 1, 1] × [TPG, 1, 1]` | `Elementwise` (default) |

```rust
let mut k = mt_<op>::kernel_ir_for(dt);
k.mode = KernelMode::Reduction;
let msl = MslGenerator::default().generate(&k).unwrap();
```

---

## MLX reference kernel names — how to find them

**Never guess the MLX function name.** grep the metal file:

```sh
grep 'kernel void\|host_name\|template.*kernel' src/metal/<op>.metal
```

Common patterns found in practice:

| Op file | MLX function name |
|---|---|
| unary | `v_{Op}{tname}{tname}` e.g. `v_Expfloat32float32` |
| softmax | `looped_softmax_{tname}` |
| reduce | `all_reduce_sum{tname}`, `row_reduce_simple_sum{tname}` |
| rms_norm | `rms{tn}` + bool constant slot 20 = true |
| rope | `rope_{tname}` + bool constants (forward=1, traditional=2, hs_transpose=3) |

When a kernel uses `[[function_constant(N)]]`, call:
```rust
runner.compile_with_bool_constants(SRC, "kernel_name", &[(slot, value), ...])
```

---

## Algorithm fidelity requirements

The module doc comment (`//!`) **must** document the MLX algorithm being matched:

```rust
//! MLX kernel: looped_softmax_float (softmax.metal)
//!   Params: (inp, out, n)  slots [0,1,2]
//!   Grid: [B, 1, 1] × [256, 1, 1]
//!   Algorithm: 2-pass (max then exp-sum), each thread strides over row,
//!              simd_sum + threadgroup merge for global max/sum.
//!
//! MetalTile: mt_softmax — same 2-pass algorithm.
//!   KernelMode::Reduction
```

If your DSL kernel implements a **different** algorithm than MLX (e.g. a faster
online variant), explain the deviation and why it is still functionally equivalent.

---

## Generics and precision coverage

Every kernel must be generic over `T` whenever the operation is dtype-agnostic:

```rust
#[kernel]
pub fn mt_<op><T>(a: Tensor<T>, out: Tensor<T>, ...) { ... }
```

- Use `FLOAT_DTYPES` (`[f32, f16, bf16]`) to drive all bench and test loops.
- Do **not** write separate `mt_<op>_f32` / `mt_<op>_f16` / `mt_<op>_bf16` kernels
  unless the MLX metal file itself has structurally different algorithms per dtype
  (rare — e.g. rope is f16-only because the benchmark targets a specific decode shape).
- A dtype-specific kernel (`<T>` replaced by a concrete type) is only acceptable
  when the algorithm is genuinely dtype-specific and there is no generic version in
  the MLX metal file either.
- Every dtype variant must appear in the bench output with its own row.  An op that
  only reports f32 numbers is incomplete.

---

## Absolute prohibitions

- **Never write hand-written MSL** in an op file.  The entire point of this
  project is to express every kernel in the `#[kernel]` DSL and measure the
  generated MSL against MLX's existing metal references.  If the DSL cannot yet
  express a required pattern, the op is NYI — fix the DSL or codegen instead.
- **Never call `BENCH.implemented` without a correctness check.**
- **Never guess an MLX function name** — always grep the metal file.

---

## Correctness rules

- **Every `BENCH.implemented` row requires a correctness check.**  `OpBench`
  panics if the `equiv` argument is not provided.  There are no exceptions.
- Run correctness on `N_CHECK` elements (256–4096), perf on full `SHAPES` size.
- Use `quantize_roundtrip` so the CPU reference operates on the same representable
  values the GPU receives after dtype conversion.
- For **elementwise** ops: use `dtype_tol(dt)`.
- For **reductions / norms / softmax**: use `dtype_tol_reduce(dt)`.  The tolerance
  accounts for accumulated floating-point error in the reduction path.
- When the CPU oracle is difficult to replicate exactly (complex fp order,
  e.g. strided reduce), use **MT vs MLX reference** as the correctness check
  instead of MT vs CPU.
- Check inputs must exercise the op's domain: use signed inputs for ops that
  accept negatives, positive inputs for `log`/`sqrt`/`rsqrt`, etc.

---

## Performance rules

- `bytes` must reflect **actual memory traffic**, not a template value:

  | Pattern | bytes expression |
  |---|---|
  | 1 read + 1 write | `n * eb * 2` |
  | 2 reads + 1 write | `n * eb * 3` |
  | read only (e.g. argmax) | `n * eb` |
  | row-parallel (B rows) | `b * n * eb * 2` |

- The reference `runner.bench` dispatch (`grid`, `tpg`) must exactly match what
  the MLX metal file expects for that kernel — read the metal file to confirm.
- MT dispatch must match the `KernelMode` used during MSL generation; mismatching
  these silently produces wrong results.
- Warmup=3, iterations=10 is the standard (`runner.bench(..., 3, 10)`).

---

## Shape string format

| Op kind | Shape string |
|---|---|
| 1-D flat | `format!("N={n} {dlabel}")` |
| Row-parallel | `format!("B={b} N={n} {dlabel}")` |
| Named sub-ops (e.g. unary) | `format!("N={n} {op_name} {dlabel}")` |
| Multi-dim (e.g. rope) | `format!("B{B}H{H}L{L}D{D}")` |

---

## Existing violations — technical debt, not patterns to follow

`softmax.rs`, `scan.rs`, and `arg_reduce.rs` contain hand-written inline MSL
constants (`ONLINE_SOFTMAX_MSL`, `PARALLEL_MSL`).  These are **technical debt**
that predates this rule.  They are not examples to copy.  When the DSL gains the
required primitives (e.g. `simd_shuffle_down`, two-phase scan), these files must
be converted to pure DSL kernels.

---

## Registration

1. Add `pub mod <op>;` to `src/ops/mod.rs`.
2. Add `pub use <op>::bench_<op>;` to `src/ops/mod.rs`.
3. Call `bench_<op>(runner)` in `src/main.rs` bench dispatch.
