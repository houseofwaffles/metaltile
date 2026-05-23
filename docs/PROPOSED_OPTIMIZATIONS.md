# Proposed Micro-Optimizations — metaltile perf kernels

Generated: 2026-05-23  
Branch: `ek/t4-microopts`  
Base: `feat/int-quant-perf`

These proposals were researched during the T4 micro-optimization pass.  Each
entry explains why the pattern cannot be applied cleanly today (missing DSL
primitives, codegen changes required, or benefit below threshold) and what
would need to change before it could land.

---

## Pattern 1 — Vectorized X loads (`float4` / `half4`) in GEMV/GEMM hot loops

**Status**: **Already implemented** — no action required.

**Targets**: `mt_qmv`, `mt_qmm{,_bm2,_bm4}`, `mt_qmv_int8_fast`,
`mt_qmm_int8_fast{,_bm2,_bm4}`, `mt_qmm_mma`, `mt_qmm_mma_m16`,
`mt_qmm_mma_int8`, `mt_qmm_mma_m16_int8`.

**Finding**: The `VectorizePass` in
`crates/metaltile-codegen/src/passes/vectorize.rs` already detects
consecutive `Load` ops at contiguous indices and promotes them to
`Op::VectorLoad` with width up to 4 (`MAX_VEC_RUN = 4`).  The MSL
emitter (`crates/metaltile-codegen/src/msl/emit_block.rs`) then lowers
`VectorLoad{len:4, dtype:F32}` → `float4`, `{len:4, dtype:F16}` →
`half4`, `{len:4, dtype:BF16}` → `bfloat4`.

All X-load sites in the GEMV/GEMM kernels are structured as consecutive
scalar `Load` calls at `base + 0 .. base + N` — explicitly sequenced to
be contiguous in the IR so the vectorize pass fuses them.  The kernel
comments confirm this (e.g. `"16 X loads — consecutive in IR for
vectorize fusion (4× float4)"` in `mt_qmv`; `"8 contiguous device
loads → 2× vec4 after vectorize pass"` in `mt_qmm_mma`).

**Verification**: Confirmed by inspecting the VectorizePass source, the
MSL emitter, and the kernel DSL code — no gap.

---

## Pattern 2 — `simd_broadcast` for scale/bias loads shared across lanes

**Status**: **Proposal only** — hardware already coalesces same-address
loads; benefit unconfirmed; implementation adds complexity without clear
profiling evidence.

**Targets**: `mt_qmv`, `mt_qmm{,_bm2,_bm4}`, `mt_qmv_int8_fast`,
`mt_qmm_int8_fast{,_bm2,_bm4}`.

**Background**: The DSL exposes `simd_broadcast(value, lane)` →
`simd_broadcast(v, lane)` in Metal Shading Language (see
`body_parser.rs:1315`, `ir.rs:986`, `emit_block.rs:1065`).  The
primitive exists and is used in the AURA codebook kernels.

**Where lanes share scale/bias addresses**:

*int4 kernels* (`mt_qmv`, `mt_qmm*`): K-block = 512 elements.  Each
lane owns 16 X values; group_size = 64.  Lanes 0–3 share group `g = 0`
for the first K-block, lanes 4–7 share group `g = 1`, etc.  Every 4
consecutive lanes read the same `scales[sb_base + g]` and
`biases[sb_base + g]` — 4 redundant device loads per 4-lane group, ×4
output rows = 32 redundant loads per K-block outer-iteration.

*int8 kernels* (`mt_qmv_int8_fast`, `mt_qmm_int8_fast*`): K-block =
128.  Each lane owns 4 X values; group_size = 64.  Lanes 0–15 share
group `g = _b / 64`, lanes 16–31 share group `g + 1`.  16 redundant
loads per K-block per output row.

**What the implementation would look like**:
```rust
// Determine the representative lane for this group (lowest lane in group).
// For int4 (16 X/lane, gs=64): base_lane = (lane / 4) * 4 = lane & !3
let base_lane = lane & !3u32;           // int4 groups: 4 lanes/group
// Only the base lane does the device load; others present 0.0.
let s0_raw = if lane == base_lane { load(scales[sb_base0 + g]).cast::<f32>() } else { 0.0f32 };
let s0 = simd_broadcast(s0_raw, base_lane);
let bi0_raw = if lane == base_lane { load(biases[sb_base0 + g]).cast::<f32>() } else { 0.0f32 };
let bi0 = simd_broadcast(bi0_raw, base_lane);
```

**Why not applied now**:

1. **Apple GPU cache broadcast semantics**: Apple Silicon GPUs (M1–M4)
   handle same-address loads from multiple threads in a simdgroup as a
   single cache-line fetch with broadcast.  The 4-lane (int4) or 16-lane
   (int8) same-address loads already cost ≈1 effective device access.
   The `simd_broadcast` reformulation saves instruction count (fewer
   load instructions in the shader) but does not reduce L2 traffic —
   only a profiler can show whether the instruction pressure is the
   actual bottleneck on these kernels.

2. **Implementation complexity**: The `if lane == base_lane … else
   0.0f32` conditional introduces a divergent branch + the `select`
   pattern; the existing DSL `if` lowers to an `if_conversion` pass
   that can generate predicated instructions, but the branch itself adds
   IR nodes and may perturb the schedule pass for the load/compute
   interleaving.

3. **`base_lane` is dynamic**: `simd_broadcast(val, lane)` where `lane`
   is a non-constant variable requires Metal 2.1+ which all M-series
   hardware satisfies — but the pass must handle the dynamic-lane
   argument in the `broadcast` cost model (currently the VectorizePass
   and SchedulePass have no knowledge of `simd_broadcast` dependencies).

**Prerequisite work**: Profile `mt_qmv_int8_fast` with Metal GPU counter
`LOAD_CACHE_MISS_RATE` to confirm same-address loads are not already
coalesced by hardware.  If they are not (i.e. older non-M-series
hardware or future non-Apple targets), add a `FastPath` flag to the
`BenchSpec` that enables the broadcast reformulation via a kernel-level
`constexpr` toggle rather than restructuring all kernels.

---

## Pattern 3 — `fast::` math intrinsics in audio + numeric paths

**Status**: **Proposal only** — `fast_exp`, `fast_log`, `fast_sin`,
`fast_cos` are not exposed by the DSL; adding them requires new
`UnaryOpKind` variants + codegen + precision characterization.

**Targets**: `mel_spectrogram` (sin/cos inner DFT, log filterbank),
`mt_softmax` (exp in both passes), `mt_logsumexp` (exp in loop, log at
store), `vocoder_istft` (sin/cos inner iDFT).

**Current emission**: `UnaryOpKind::{Exp,Log,Sin,Cos}` in
`crates/metaltile-core/src/ir.rs` emit `exp(arg)`, `log(arg)`,
`sin(arg)`, `cos(arg)` — IEEE-754 precise Metal built-ins.  No
`fast::exp`, `fast::log`, `fast::sin`, `fast::cos` variants exist
anywhere in the codegen stack.

**What Metal's fast math provides**: Metal's `fast` namespace (`#include
<metal_math>`) provides `fast::exp`, `fast::log`, `fast::sin`,
`fast::cos`, `fast::sqrt` — polynomial approximations that are:
- 1–3 ULP accurate (vs. ≤ 0.5 ULP for the IEEE versions)
- Approximately 1.5–2× faster in throughput-limited contexts
- Unsafe for edge cases: `fast::log(-1.0)` is undefined, `fast::exp`
  may not flush denormals

For `mel_spectrogram`: The direct DFT inner loop computes
`cos(angle) * xw` and `sin(angle) * xw` for each of the `n_fft` ≈ 400
samples per frequency bin.  The `angle` range is `[-2π, 0]`; inputs are
well-defined reals — fast math edge cases don't apply.  Expected speedup:
~1.5–1.8× on the DFT inner loop (the dominant cost).

For `mt_softmax`: `exp(v_i - max_i)` — arguments are always ≤ 0
(subtracted maximum), so the output is always in `(0, 1]`.  Fast exp is
safe here.  Expected speedup: ~1.4× on the reduction pass.

For `mt_logsumexp`: Same `exp` path as softmax (arguments ≤ 0), plus a
final `log(gs)` where `gs > 0` is always true.  Safe.

For `vocoder_istft`: `cos(angle)` and `sin(angle)` inner loop — same
argument range analysis as `mel_spectrogram`.  Safe.

**What needs to change**:

1. **New `UnaryOpKind` variants** in `metaltile-core/src/ir.rs`:
   ```rust
   FastExp,
   FastLog,
   FastSin,
   FastCos,
   FastSqrt,
   ```
   with `msl_emit` returning `fast::exp(arg)` etc.

2. **New DSL keywords** in `metaltile-macros/src/body_parser.rs`:
   ```rust
   "fast_exp"  => quote! { UnaryOpKind::FastExp },
   "fast_log"  => quote! { UnaryOpKind::FastLog },
   "fast_sin"  => quote! { UnaryOpKind::FastSin },
   "fast_cos"  => quote! { UnaryOpKind::FastCos },
   "fast_sqrt" => quote! { UnaryOpKind::FastSqrt },
   ```

3. **Precision validation**: The `mel_spectrogram_gpu_correctness` test
   tolerates `1e-3`; the audio end-to-end (waveform cosine similarity)
   is more forgiving.  The softmax / logsumexp tests use `1e-4`.  A
   numerical sweep of `fast::exp` error against the IEEE path for the
   actual argument ranges would be needed before landing.

4. **Kernel edits**: Replace `sin`/`cos`/`exp`/`log` with `fast_sin` /
   `fast_cos` / `fast_exp` / `fast_log` at the relevant call sites in:
   - `crates/metaltile-std/src/ffai/mel_spectrogram.rs`
   - `crates/metaltile-std/src/mlx/softmax.rs`
   - `crates/metaltile-std/src/mlx/logsumexp.rs`
   - `crates/metaltile-std/src/ffai/vocoder.rs`

**Risk**: If the fast-math approximation error accumulates over many
inner-loop iterations (e.g. 400 DFT taps), the cosine similarity of the
Mel spectrogram output may drop below 0.999.  The existing tests would
catch this.  A safe rollout would compare fast vs IEEE outputs on a
representative audio clip before committing.

---

## Pattern 4 — f16/bf16 accumulator for small-K shapes

**Status**: **Not applicable** — all production K shapes in the target
kernels accumulate across ≥ 64 elements under quantization noise;
switching to T-precision accumulators would degrade accuracy below the
0.999 cosine threshold.

**Targets examined**: `mt_qmv`, `mt_qmm{,_bm2,_bm4}`,
`mt_qmv_int8_fast`, `mt_qmm_int8_fast{,_bm2,_bm4}`, `mel_spectrogram`,
`mt_softmax`, `mt_logsumexp`, `vocoder_istft`.

**Analysis**:

*Quantized GEMV/GEMM kernels*: Every accumulator is `f32` by design.
The comment in `mt_qmv` is explicit: "accumulators stay in f32
regardless of T".  The reason is that int4 quantization introduces up to
`q_err = 15/(2^4) × scale = 0.93 × scale` per element; with 64 elements
per group and scale ≈ 1/16, accumulated error without f32 headroom
exceeds the cosine similarity gate.  For int8 the dynamic range is
similar: 255/256 × scale per element, same argument applies.

*`mel_spectrogram`*: The inner DFT accumulates `re += xw * cos(angle)`
over 400 samples.  Running sum of floats at f16 precision would
accumulate ≈ 0.001 × √400 ≈ 0.02 absolute error — audible in the
output spectrogram.  f32 accumulation is required.

*`mt_softmax`, `mt_logsumexp`*: Online softmax requires both a max
accumulator and a sum accumulator.  The numerical stability of the online
algorithm depends on the precision of `exp(v - m)` where `v` and `m` may
differ by many units.  f16 would lose the mantissa bits needed to recover
from the `exp(large_neg_val)` tail — catastrophic cancellation risk.

*`vocoder_istft`*: Same DFT accumulation argument as `mel_spectrogram`
but the n_fft is smaller (Kokoro: 20) — at n_fft=20, f16 might be
borderline, but the existing f32 path is not a bottleneck and the test
tolerance is `1e-3`.

**Conclusion**: fp32 accumulators are correctness-critical for all
identified kernel shapes.  The pattern is safe only if a kernel has both:
(a) K ≤ 16 per lane per accumulation, and (b) no quantization noise.
None of the production GEMV/GEMM targets satisfy (a); the audio paths
violate (b).

---

## Pattern 5 — K-loop software pipelining (prefetch-overlap)

**Status**: **Proposal only** — requires codegen support that does not
exist; implementing it without codegen changes would require hand-unrolling
the K-loop which defeats DSL portability.

**Targets**: `mt_qmm_mma`, `mt_qmm_mma_m16`, `mt_qmm_mma_int8`,
`mt_qmm_mma_m16_int8`, `mt_qmm_mma_mpp` (via `quantized_mpp.rs`),
`mt_qmm_nax`, `mt_qmm_nax_int8`, `mt_moe_gather_qmm_mma_int4`,
`mt_moe_gather_qmm_mma_int8` and the `_bm{8,16,64}_mpp` MoE variants.

**Pattern**: Apple's hand-tuned `mma_qmm` Metal shaders (and typical
CUTLASS-style GPU kernels) overlap loading the next K-block into
shared/threadgroup memory while the MMA units process the previous K-block:

```
// Pipelined form (2-stage):
load_to_tg(K=0)
threadgroup_barrier()
for k in 1..K_blocks:
    async_issue: load_to_tg(K=k)   // start next load
    mma(K=k-1)                      // compute on current
    threadgroup_barrier()            // wait for next load
mma(K=K_blocks-1)
```

vs. our current sequential form:
```
for k in 0..K_blocks:
    load_to_tg(K=k)
    threadgroup_barrier()
    mma(K=k)
    threadgroup_barrier()
```

**Why not done today**: The DSL's `for k in range(...)` block compiles
into a monolithic loop in the IR; the codegen schedule pass (`schedule.rs`)
is unaware of the "hoist-load-before-barrier" opportunity.  Expressing
software pipelining requires either:

1. **A `prefetch_async` IR op** — a new `Op::PrefetchAsync { src, tg_dst,
   ... }` that signals the schedule pass to emit the device load before
   the simdgroup barrier of the preceding block.  The schedule pass would
   need a new cost model entry for memory-latency hiding.

2. **A two-buffer (ping-pong) DSL primitive** — expose a `double_buffer`
   wrapper in the body_parser that allocates two threadgroup arrays
   (`Xs_a`, `Xs_b`, `Ws_a`, `Ws_b`) and auto-alternates `load` / `mma`
   assignments, emitting a two-trip peeled loop.  This is the
   `ldmatrix` / `wmma` pipelining pattern used by CUTLASS.

3. **Explicit loop-peeling at the kernel DSL level** — hand-peel the
   loop prologue and epilogue in the kernel source, using two named
   threadgroup buffers alternately.  This gives correct pipelining
   without codegen changes but roughly doubles the kernel source length
   and is fragile across K shapes.

**Expected benefit**: On M3 Pro with `mt_qmm_mma` (K=4096, N=4096, M=32),
an Apple hand-tuned pipelined MMA kernel typically runs 15–25% faster
than the non-pipelined equivalent (Apple's own `steel_gemm` docs reference
this range for their pipelined GEMM over the non-pipelined baseline).
The gain is larger for larger K (longer memory latency → more hiding
opportunity).

**Recommended approach**: Implement `Op::PrefetchAsync` as a new codegen
pass (`prefetch.rs`) that runs after `schedule.rs` and `licm.rs`,
identifies the pattern `[ThreadgroupStore* … ThreadgroupBarrier …
MMA*]` within a loop body, and hoists the store block one barrier ahead.
This is a well-understood transformation; the risk is incorrect barrier
placement (would produce wrong results, caught by existing MMA
correctness tests).  Estimated scope: 2–3 days of codegen work + 1 day
of correctness testing.

---

## Summary table

| Pattern | Status | Blocker |
|---|---|---|
| 1. `float4`/`half4` X loads | Already done (VectorizePass) | — |
| 2. `simd_broadcast` for scale/bias | Proposal only | Hardware coalesces same-address loads; no profiling evidence of bottleneck; implementation complexity |
| 3. `fast::` math in audio/softmax | Proposal only | Missing `FastExp`/`FastLog`/`FastSin`/`FastCos` `UnaryOpKind` variants + precision validation |
| 4. f16/bf16 accumulator small-K | Not applicable | All targets have K ≥ 64 with quantization noise; f32 required for correctness |
| 5. K-loop software pipelining | Proposal only | Needs `Op::PrefetchAsync` codegen pass or DSL double-buffer primitive |
