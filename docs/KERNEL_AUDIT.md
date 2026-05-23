# metaltile kernel-op coverage audit

Generated: 2026-05-18 · Refreshed: 2026-05-23 (full bf16 coverage + the
remaining CoopTile DSL migration via PR #152; int4 + int8 perf-path
build-out for dense GEMM / MoE BGEMM / RMSNorm-fused GEMV in this PR) ·
Updated: 2026-05-23 (`ek/t3-quant-completeness` — odd-bitwidth MMA for
int3/5/6, fp4/fp8 E4M3 simdgroup MMA, int8 RMSNorm-fused GEMV, fp8 KV
cache, short-prefill MoE m16/m32 batching)
Sources surveyed:
- MLX upstream `ml-explore/mlx@main` (commit `2414e5df`)
- MLX fork `ekryski/mlx@alpha` (commit `4919270e`)
- metaltile `thewafflehaus/metaltile:ek/aura-port` (the consolidated branch —
  `origin/dev` plus the Gemma / Nemotron-H / GPT-OSS-20B kernel work)

## Summary

- Total kernel-op rows in this audit (union): **90**
- metaltile-ported kernel ops: **89 / 90** — 89 full ✓, 0 partial ~
- **1 op intentionally out of scope** — `fence` (a GPU-side sync
  primitive, not a compute kernel; see
  [§ Fence ops](#fence-ops--intentionally-out-of-scope)). Every other op
  in the union is fully ported — kernel coverage is complete.
- Fully ported, with no remaining `✗` / `~` rows: the `steel_gemm`
  family (`fused`, `gather`, `masked`, `segmented`, `splitk + accum`
  and their `nax` siblings), the `steel_conv` family (2D / 3D / general
  + the 3×3 Winograd fast path), `fft` (radix-2 Cooley–Tukey), the full
  `quantized` / `fp_quantized` matrix (every bit-width + the NAX
  `mpp::matmul2d` variants + the new `fp_quantized_mma` simdgroup-MMA
  row for fp4/fp8 on non-NAX hardware), the AURA codec stack, the GDN /
  SSM / MoE kernels, the Vision / STT / TTS front-end kernels, and the
  model-review host-fallback closers.
- This is the `ek/aura-port` consolidation branch — every kernel PR
  (#115 / #137 / #144 / #145 and the Gemma / Nemotron-H / GPT-OSS-20B
  worktrees) is folded in; nothing is pending on an open PR.

> **Note on the 2026-05-21 consolidation pass.** The Gemma / Nemotron-H /
> GPT-OSS-20B kernel work, previously spread across separate worktrees, is now
> consolidated onto `ek/aura-port`. Two Gemma kernels — `sdpa_decode_d512` and
> `rms_norm_wide` — are added as ✓ rows. A model-side review of FFAI's decode
> path also surfaced several **host-side compute fallbacks** that existed only
> because a GPU kernel was missing; the kernels that close them
> (`gated_rmsnorm`, the `sdpa_decode` learned-sink term, the 2D-`A_log`
> `ssm_step_a2d` variant) are now all landed (✓ rows below), and the
> **Vision / STT / TTS** front-end kernels (`conv2d`, `patch_embed`,
> `rope_2d`, `mel_spectrogram`, `audio_conv1d`, `vocoder/iSTFT`) are ✓ rows
> for Phase 6.5 / 7.
> The MLX-upstream and MLX-alpha columns were **not** re-verified against those
> repos (not checked out) — only the metaltile column was re-surveyed.

> **Note on the 2026-05-23 refresh (PR #152 + int-quant perf PR).** Two
> follow-up landings refresh the perf / dtype-coverage columns without
> changing the op-count:
>
> 1. **PR #152 — bf16 coverage + CoopTile DSL migration.** Every
>    floating-point kernel in the library now exposes f32 / f16 / bf16
>    (the AURA stack and the mel front-end were the last `[F32, F16]`-only
>    rows). Every NAX cooperative-tensor kernel — `quantized_nax`,
>    `fp_quantized_nax`, `quantized_mpp` (`mt_qmm_mma_mpp`),
>    `steel_gemm_{fused,gather,splitk}_nax`, `steel_attention_nax`,
>    `hadamard_m` — was ported from hand-built `Op::InlineMsl` IR to the
>    `#[kernel]` DSL via the `coop_tile_*` intrinsics + the new
>    `coop_stage(T)` primitive that resolves bf16 activations through
>    `half` (Apple's `matmul2d` mishandles `bfloat` cooperative tensors;
>    `half`'s 10-bit mantissa losslessly covers bf16's 7, fp32
>    accumulation). The `quantized (gather_*)` MoE MPP kernels
>    (`bm8`/`bm16`/`bm64`) had already been ported in PR #149; the rest
>    of the NAX surface caught up in #152.
> 2. **int-quant perf build-out (this PR).** int8 was previously
>    correctness-only — the existing `mt_q{mv,vm,mm}_b8` were
>    one-TG-per-output scalar kernels; every int8 inference path fell
>    ~6–8× behind the int4 perf path. This refresh adds pack-aligned
>    int8 perf variants for: dense GEMM (`mt_qmv_int8_fast`,
>    `mt_qmm{,_bm2,_bm4}_int8_fast`, `mt_qmm_mma_int8` + `_m16_int8`,
>    `mt_qmm_mma_mpp_int8`, `mt_qmm_nax_int8`); MoE BGEMM
>    (`mt_moe_gather_qmm_mma_int8` + `_bm16_mpp` + `_bm8_mpp` +
>    `_bm64_mpp`); plus three int4 follow-ups flagged in prior audits
>    (`ffai_rms_norm_qgemv_fast`, `ffai_batched_qkv_qgemv_fast`,
>    `dequant_gemv_int4_fast`) and a new perf-tuned `mt_qvm_int4_fast`.
>    Each is purely additive — the correctness-first scalars stay for
>    callers that don't hit a perf-path shape.

## Op coverage table

| Op | MLX (upstream) | MLX (ekryski@alpha) | metaltile | Notes |
|---|---|---|---|---|
| arange | ✓ | ✓ | ✓ | `mlx/arange.rs` → `mt_arange`. Generic `T`. Direct port. |
| arg_reduce (argmax/argmin → float) | ✓ | ✓ | ✓ | `mlx/arg_reduce.rs` → `mt_argmax<T>` + `mt_argmin<T>`, both generic over `T` (f32/f16/bf16 — values widened to f32 for the comparison). Both emit the winning index as `u32` (MLX `arg_reduce_general` semantics); ties take the smallest index. Verified by `mt_arg_reduce_gpu_correctness` (CPU oracle, tie-break, all three dtypes, strided cover). |
| arg_reduce (argmax → u32 index) | ✗ | ✗ | ✓ | `ffai/arg_reduce.rs` → `ffai_argmax<T>`. FFAI-only; integer-index sampler workhorse. |
| binary (elementwise add/sub/mul/div/min/max) | ✓ | ✓ | ✓ | `mlx/binary.rs` → 6 kernels. Generic `T`. Direct port. |
| binary_two (fused two-output elementwise) | ✓ | ✓ | ✓ | `mlx/binary_two.rs` → `mt_binary_two<T>`. |
| copy (contiguous) | ✓ | ✓ | ✓ | `mlx/copy.rs` → `mt_copy<T>`. |
| copy (strided / general) | ✓ | ✓ | ✓ | `mlx/strided.rs` → `mt_strided_copy` (2-D padded) **plus** `mt_strided_copy_nd` — general arbitrary-rank strided copy. Each output element unravels its contiguous flat index against a runtime `shape` array and gathers `src[Σ coord_d · strides[d]]` — MLX's `elem_to_loc` / `copy_g`. Arbitrary source strides cover padded copies, transposes (permuted strides), broadcasts (stride 0), and dilated slices in one kernel; `rank` is a constexpr so the unravel loop fully unrolls. Verified by `mt_strided_copy_gpu_correctness` (1-D contiguous, 2-D padded, 3-D padded + transpose, 4-D broadcast-axis; f32/f16). |
| ternary (select) | ✓ | ✓ | ✓ | `mlx/ternary.rs` → `mt_select<T>`. |
| unary (exp/log/sqrt/rsqrt/abs/silu/etc.) | ✓ | ✓ | ✓ | `mlx/unary.rs` → 7+ kernels including `mt_silu`. |
| swiglu (`silu(gate)·up` fused MLP activation) | ✗ | ✗ | ✓ | `mlx/swiglu.rs` → `mt_swiglu<T>`. Fused element-wise `silu(gate) * up` — the standard modern-transformer MLP activation (Llama 4, Qwen3 dense + MoE, Gemma, Mistral). metaltile fuses what MLX expresses as separate `silu` + `mul` ops; no dedicated MLX kernel. The broader `fused_gate_activation` (gelu / clipped-swiglu variants) is still a separate ✗ row below. |
| random (key hash → u32) | ✓ | ✓ | ✓ | `mlx/random.rs` → `mt_random_hash`. |
| reduce (sum/prod/max/min — all + row + col) | ✓ | ✓ | ✓ | `mlx/reduce.rs` covers `all_reduce*`, `row_reduce*`, `col_reduce*` (Grid3D one-thread-per-column, `cols`-strided fold) and `seg_reduce*` (Grid3D one-thread-per-segment, contiguous fixed-length runs) — all four ops (sum/prod/max/min) for each shape. Verified by `reduce_col_seg_gpu_correctness`. |
| sort | ✓ | ✓ | ✓ | `mlx/sort.rs` → `mt_sort<T>` (single-block bitonic sort) + `mt_merge<T>` (multi-block bottom-up merge pass) + `mt_sort_segmented<T>` (per-row bitonic sort for `[batch, n]` matrices, `n ≤ 1024`, one TG per row). `mt_sort` sorts each 1024-element block; `mt_merge` merges adjacent sorted runs (caller ping-pongs two buffers); `mt_sort_segmented` uses `tgid_x` as the row index and a `+∞` sentinel for out-of-range slots — identical 10-stage bitonic network, Reduction-mode grid `[batch, 1, 1]`. Verified by `sort_gpu_correctness` (multi-block, reverse-sorted, f32+f16) and `sort_segmented_gpu_correctness` (n ∈ {64,256,512,1024}, reverse-sorted, two-row / four-row / eight-row, f32+f16+bf16, monotonicity). |
| scan (prefix sum) | ✓ | ✓ | ✓ | `mlx/scan.rs` → `mt_scan<T>` (inclusive) + `mt_scan_exclusive<T>` (exclusive) + `mt_scan_prod<T>` / `mt_scan_prod_exclusive<T>` (prefix product) + `mt_scan_max<T>` / `mt_scan_max_exclusive<T>` (running max) + `mt_scan_min<T>` / `mt_scan_min_exclusive<T>` (running min). Sum pair uses hardware `simd_scan_exclusive`; prod/max/min pairs use a `tgs[lsize]` threadgroup buffer for sequential cross-thread prefix reads (no missing intrinsics). Cross-chunk running prefix stored at `sgs[n_simd]` (identity element: 1.0 / −∞ / +∞). Required `msl/features.rs` fix: `Op::Load { src: "n_simd" }` now correctly sets `needs_simd_group = true` so the preamble emits `uint n_simd = lsize / 32u;`. Verified by `scan_exclusive_gpu_correctness` (sum, chunk-aligned + ragged) and `scan_multi_op_gpu_correctness` (prod/max/min × inclusive/exclusive × aligned/ragged, 12 tests). |
| softmax | ✓ | ✓ | ✓ | `mlx/softmax.rs` → `mt_softmax<T>` (looped + single-row collapsed). |
| logsumexp | ✓ | ✓ | ✓ | `mlx/logsumexp.rs` → `mt_logsumexp<T>`. |
| layer_norm | ✓ | ✓ | ✓ | `mlx/layer_norm.rs` → `mt_layer_norm<T>`. |
| rms_norm | ✓ | ✓ | ✓ | `mlx/rms_norm.rs` → `mt_rms_norm<T>` plus `mt_rms_norm_small<T>` (2-elem/thread small-head_dim variant for the per-head q_norm/k_norm dispatch). |
| rope (standard) | ✓ | ✓ | ✓ | `mlx/rope.rs` → `mt_rope` (fp16 only). |
| rope (Llama-3 banded) | ✗ | ✗ | ✓ | `ffai/rope_llama.rs` → `ffai_rope_llama<T>`. Decode-form, generic dtype, optional Llama-3 frequency-band scaling. No MLX counterpart. |
| sdpa_vector (prefill / generic) | ✓ | ✓ | ✓ | `mlx/scaled_dot_product_attention.rs` → `mt_sdpa<T>`. Scalar SDPA — sufficient for short sequences. |
| sdpa_vector (GQA decode, single pass) | ✓ | ✓ | ✓ | `mlx/sdpa_vector.rs` → `mt_sdpa_vector<T>` (head_dim=128) **plus** `mt_sdpa_vector_d{64,96,192,256}` — GQA single-pass decode at all production head_dims (64/96/192/256). Each variant scales the per-lane element count and reduction-phase count accordingly (2/3/6/8 elements per lane). All cover f32/f16/bf16; TPG=1024 throughout. Verified by `sdpa_vector_gpu_correctness` (f32, f16, GQA at each new head_dim). |
| sdpa_vector_2pass | ✓ | ✓ | ✓ | `ffai/sdpa_decode_2pass.rs` → pass1/pass2 pairs for head_dim ∈ {64, 96, 128, 256}. d=64 (2 elems/lane), d=96 (3 elems/lane), d=128 (4 elems/lane, original), d=256 (8 elems/lane, 4-buffer reuse across 2 phases to stay within the 32KB TG cap). Verified by `sdpa_decode_2pass_gpu` (f32 + GQA at each head_dim). |
| sdpa_decode (FFAI production decode, decoupled `kv_stride`) | ✗ | ✗ | ✓ | `ffai/sdpa_decode.rs` → `ffai_sdpa_decode<T>`, plus `ffai/sdpa_decode_d64.rs` / `sdpa_decode_d256.rs` for head_dim {64, 256}. FFAI-only variant with `kv_stride` ≠ `n_kv` (pre-allocated max-seq cache); now covers head_dim ∈ {64, 128, 256} and a sliding-window + sink-token path (`sink_end` / `window_start` constexprs). |
| sdpa_decode_batched (speculative-decode batched-Q decode) | ✗ | ✗ | ✓ | `ffai/sdpa_decode_batched.rs` → `sdpa_decode_batched_q{2,4,8}<T>` (+ `sdpa_decode_batched_prefill.rs`). K query positions share one KV walk per dispatch (M7 speculative decoding), amortizing KV memory bandwidth K× vs. K independent single-Q `sdpa_decode` dispatches. `q8` extends the pattern to K=8 (M7 branching factor 8): 8 independent `(run_max, run_sum)` tuples + 32 output accumulators updated in lockstep per KV position; 8 sequential output-reduction phases (A–H) reusing `tg_max/tg_sum/tg_out0..3` buffers. Q layout `[n_q_heads, 8, head_dim]`; output `[n_q_heads, 8, head_dim]`. TPG=256 (conservative for M1/M2/M3 register file). Verified by `sdpa_decode_batched_q8_gpu_correctness` (small n_kv=4, large n_kv=1024 GQA=4, identical-Q phase-aliasing guard; 3 tests). FFAI-only. |
| steel_attention (Flash, prefill) | ✓ | ✓ | ✓ | `mlx/steel/attn/steel_attention.rs` → `mt_sdpa_prefill<T>`. Scalar-flash prefill (BQ=4, online softmax, causal), generic `T`, head_dim=128. The old "`Op::FlashAttention` lowers to an error placeholder" blocker is resolved. |
| steel_attention_mma (Flash prefill, simdgroup-MMA) | ✓ | ✓ | ✓ | `mlx/steel/attn/steel_attention_mma.rs` → `mt_sdpa_prefill_mma<T>`. Real simdgroup-matrix MMA path; generic `T`, validated f32/f16/bf16, head_dim=128. A pre-M3 bf16-tuned sibling `mt_sdpa_prefill_mma_bf16` (`steel_attention_mma_bf16.rs`) is selected by `sdpa_prefill_mma_for()` — a perf specialization, not a separate op. |
| steel_attention_nax | ✓ | ✓ | ✓ | `mlx/steel/attn/steel_attention_nax.rs` → `mt_sdpa_prefill_nax<T>` (head_dim=32) **plus** `mt_sdpa_prefill_nax_d{64,128,256}` — flash-attention prefill via Apple `mpp::tensor_ops::matmul2d` (NAX tensor cores) at all production head_dims. The base d=32 kernel: BQ=16, BK=16, BD=32, tpg=32 (1 SG); the QK descriptor's K-dim is exactly 32 (Apple's "one of M/N/K=32" rule). The wide variants loop the QK contraction over `head_dim/32` consecutive 32-wide D-chunks inside the outer K-block loop: the first chunk uses an `overwrite` coop descriptor, subsequent chunks use an `accumulate` descriptor; the PV contraction stores each chunk into a scratch `Opv` tile (16×36) then manually accumulates into the full-width `Obk` output buffer (16×68/132/260 for d64/128/256). The S-tile remains 16×16 at all head_dims. QK descriptor `(16,16,32)` tb=true (Kᵀ transposed-B read); PV descriptor `(16,32,16)`. Per-block max-rescale of the running O accumulator gives correct online softmax. Causal masking + GQA. Expressed in the `#[kernel]` DSL via `coop_tile_*` + `coop_stage(T)` for bf16-safe staging (PR #152). `#[cfg(feature = "nax")]`-gated; needs macOS 26+ / Metal 4. Verified by `steel_attention_nax_gpu_correctness` across f32/f16 at each head_dim (single-tile, multi-tile causal, GQA). |
| steel_gemm_fused | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_fused.rs` → `mt_steel_gemm_{64x64x16_2x2,32x32x16_2x2,64x64x16_1x2,32x64x16_1x2}<T>`. Plain row-major `C = A·B` via Apple 8×8 simdgroup-matrix MMA; four block-shape instantiations (each mirrors an MLX `instantiate_gemm_shapes_helper` shape). Fixed a transposed-B fragment-load bug in the original `64×64×16_2x2` kernel (it loaded `B` with the `(fn,fm)` GEMM-transposed lane convention, shipping `Bᵀ`-shaped output) plus a missing K-accumulation loop (only summed K∈[0,16)). Verified by `steel_gemm_gpu_correctness` (all four transpose modes, f32/f16/bf16). |
| steel_gemm_fused_nax | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_fused_nax.rs` → `mt_steel_gemm_fused_nax<T>` — plain fused GEMM `C = A·B` via Apple `mpp::tensor_ops::matmul2d` (NAX tensor cores). Cooperative-tensor counterpart of `steel_gemm_fused`; expressed in the `#[kernel]` DSL via the `coop_tile_*` intrinsics + `coop_stage(T)` for bf16-safe matmul2d staging (PR #152 — no more `Op::InlineMsl`). Same machinery as `quantized_nax` minus the int4 dequant: B is dense `T`, coop-loaded transposed into the TG tile. `#[cfg(feature = "nax")]`-gated; needs macOS 26+ / Metal 4. Verified by `steel_gemm_fused_nax_gpu_correctness` across f32/f16/bf16 vs a naive triple-loop oracle. |
| steel_gemm_gather | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_gather.rs` → `mt_steel_gemm_gather_{64x64x16_2x2,32x32x16_2x2}<T>`. Row-major `C = A_gathered·B_gathered` (MLX `gather_mm`, the dense matmul of a MoE FFN): a `lhs_indices` buffer redirects each output row to a non-contiguous `A` row, a `rhs_indices` buffer selects which `[K,N]` `B` matrix each N-block multiplies against. No gather-load primitive needed — the redirection is one extra `u32` load before ordinary address arithmetic (the gather index is a per-row scalar, shared by every lane in the fragment row). Verified by `steel_gemm_gather_gpu_correctness` (identity, permuted lhs, rhs-select; f32/f16/bf16). |
| steel_gemm_gather_nax | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_gather_nax.rs` → `mt_steel_gemm_gather_nax<T>` — gather GEMM `C = A_gathered·B_gathered` via Apple `mpp::tensor_ops::matmul2d` (NAX tensor cores). Cooperative-tensor counterpart of `steel_gemm_gather`: exactly `steel_gemm_fused_nax` with two extra `u32` index loads (per-row `lhs_indices`, per-N-block `rhs_indices`) before the address arithmetic — no new codegen primitive. Expressed in the `#[kernel]` DSL + `coop_stage(T)` for bf16 (PR #152 — no more `Op::InlineMsl`). `#[cfg(feature = "nax")]`-gated; needs macOS 26+ / Metal 4. f32/f16/bf16. |
| steel_gemm_masked | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_masked.rs` → `mt_steel_gemm_masked_{64x64x16_2x2,32x32x16_2x2}<T>`. Block-masked row-major `C = A·B`: an output-block mask zeroes whole `BM×BN` blocks (uniform `if` around the K-loop + `select` on the store), an operand-block mask scales each `BM×BK`/`BK×BN` K-block contribution (a `0` mask multiplies the loaded fragment to zero — branchless). Both masks are plain `Tensor<T>` operands; no new codegen primitive needed. Verified by `steel_gemm_masked_gpu_correctness` (all-ones, checkerboard out-mask, partial op-mask; f32/f16/bf16). |
| steel_gemm_segmented | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_segmented.rs` → `mt_steel_gemm_segmented_{64x64x16_2x2,32x32x16_2x2}<T>`. Ragged-K batched matmul (MLX `segmented_mm`): each segment sums over its own `[k_start, k_end)` K-range of a shared `A`/`B`, output is `[n_segments, M, N]`. Expressed as the fused GEMM with a 3-D grid (`program_id<2>` = segment) and a K-loop whose bounds are read from a `segments` descriptor buffer instead of being a constexpr — `range(k_start, k_end, 16)` with variable bounds. No new codegen primitive needed. Verified by `steel_gemm_segmented_gpu_correctness` (single-full, disjoint, uneven ranges; f32/f16/bf16). |
| steel_gemm_splitk + accum | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_splitk.rs` → pass 1 `mt_steel_gemm_splitk_{64x64x16_2x2,32x32x16_2x2}<T>` + pass 2 `mt_steel_gemm_splitk_accum<T>` / `mt_steel_gemm_splitk_accum_axpby<T>`. Two-kernel split-K: pass 1 partitions K across a 3-D grid (`program_id<2>` = K-split, `range(k_start, k_end, 16)` clamped to `k`) and writes per-split fp32 partials to an `[n_splits, M, N]` buffer; pass 2 is a one-thread-per-output Elementwise reduce over the splits (plain sum, or `axpby` form `α·Σ + β·C_in`). The inter-kernel handoff is an ordinary fp32 device buffer — no split-K scheduling primitive needed; the partials stay fp32 so the cross-split sum keeps full precision for f16/bf16 inputs. Verified by `steel_gemm_splitk_gpu_correctness` (2-way, 3-way, axpby; f32/f16). |
| steel_gemm_splitk_nax | ✓ | ✓ | ✓ | `mlx/steel/gemm/steel_gemm_splitk_nax.rs` → pass 1 `mt_steel_gemm_splitk_nax<T>` + pass 2 `mt_steel_gemm_splitk_accum_nax<T>`. Two-kernel split-K via Apple `mpp::tensor_ops::matmul2d` (NAX tensor cores): pass 1 is `steel_gemm_fused_nax` with a 3-D grid (`tgid_z` = K-split, K-loop clamped to `k`) writing per-split fp32 partials to an `[n_splits, M, N]` buffer; pass 2 is a one-thread-per-output reduce over the splits (plain sum). The inter-kernel handoff is an ordinary fp32 device buffer; partials stay fp32 so the cross-split sum keeps full precision for f16/bf16 inputs. Expressed in the `#[kernel]` DSL + `coop_stage(T)` for bf16 (PR #152 — no more `Op::InlineMsl`). `#[cfg(feature = "nax")]`-gated; needs macOS 26+ / Metal 4. Verified by `steel_gemm_splitk_nax_gpu_correctness` (2-way, 3-way, multi-tile; f32/f16/bf16). |
| steel_conv 2D (implicit-GEMM) | ✓ | ✓ | ✓ | `ffai/conv2d.rs` → `conv2d_patch14` / `conv2d_patch16` / `conv2d_generic`. 2D convolution as a direct conv (implicit im2col, one thread per output) rather than MLX's explicit-im2col tiled GEMM — equivalent result, no im2col staging buffer. Covers fixed-patch and runtime-stride/pad configs. **MMA-tiled perf path**: `ffai/conv2d_mma.rs` → `conv2d_mma<T>` — implicit-im2col + 4-SG 2×2 simdgroup-matrix MMA, 32×32 output tile (stride=1/dilation=1/pad=0, out_ch and n_pixels divisible by 32); ~5–10× ALU utilisation gain for large-hidden ViT shapes. Verified by `conv2d_gpu_correctness` + `conv2d_mma_gpu_correctness`. |
| steel_conv 3D | ✓ | ✓ | ✓ | `ffai/conv3d.rs` → `conv3d_generic` (strided / padded dense 3D conv) + `conv3d_grouped` (adds dilation + grouped channels; `groups == in_ch` is depthwise). 5D NCDHW input, OIDHW weight — the volumetric counterpart of `conv2d.rs`: direct conv (implicit im2col), one thread per output voxel, fp32 accumulation, padding taps masked in the padded-input frame. Generic `T` (f32/f16/bf16). **MMA-tiled perf path**: `ffai/conv3d_mma.rs` → `conv3d_mma<T>` — implicit-im2col over `(kd, kh, kw, ic)` gather + 4-SG 2×2 simdgroup-matrix MMA, 32×32 output tile (stride=1/dilation=1/pad=0). Verified by `conv3d_gpu_correctness` + `conv3d_mma_gpu_correctness`. |
| steel_conv_general (strides/dilation/groups) | ✓ | ✓ | ✓ | `ffai/conv2d.rs` → `conv2d_grouped<T>`. Fully general 2D conv: strides, dilation (atrous), padding, and grouped channels (`groups == in_ch` is depthwise). NCHW input, OIHW weight with the I dimension = `in_ch/groups`. Direct conv, one thread per output, fp32 accumulation. Verified by `conv2d_gpu_correctness`. |
| conv (winograd + naive_unfold + depthwise) | ✓ | ✓ | ✓ | The `naive_unfold` + depthwise cases are covered for **both 2D and 3D** — `ffai/conv2d.rs` (`conv2d_generic` + `conv2d_grouped`) and `ffai/conv3d.rs` (`conv3d_generic` + `conv3d_grouped`); the `_grouped` kernels handle depthwise via `groups == in_ch` and dilation (atrous). The Winograd fast-conv path is `ffai/winograd_conv.rs` → `winograd_conv2d_3x3<T>` — the F(2×2, 3×3) minimal-filtering algorithm (input/filter/output transforms + a 4×4 element-wise product summed over `in_ch`), one thread per 2×2 output tile; requires even output dims (`conv2d_generic` covers odd outputs). The cuDNN-style split into separate filter-transform / batched-GEMM / untransform kernels is a perf follow-up. Verified by `winograd_conv_gpu_correctness`. The old `mlx/conv.rs` bench-crate stub is superseded. |
| gemv | ✓ | ✓ | ✓ | `mlx/gemv.rs` → `mt_gemv<T>`. |
| gemv_masked | ✓ | ✓ | ✓ | `mlx/gemv_masked.rs` → `mt_gemv_masked<T>` (no MLX comparison wired). |
| quantized (affine_quantize / affine_dequantize) | ✓ | ✓ | ✓ | `mlx/quantized.rs` → quantize **and** dequantize for all widths: int2/int4/int8 (power-of-2, pack-aligned) + int3/int5/int6 (byte-stream, non-power-of-2). All six quantize kernels (`mt_affine_quantize_int{2,3,4,5,6,8}`) + six dequantize kernels (`mt_affine_dequantize_int{2,3,4,5,6,8}`) are ported. The int3/5/6 quantize kernels use a bit-stream OR strategy (lane 0 iterates over all group_size elements, ORing each code into the correct uint32 word) to handle codes that straddle word boundaries — no atomics needed. Verified by `affine_int2_gpu_correctness` (int2 round-trip) + `affine_int356_quantize_gpu_correctness` (int3/5/6 quantize→dequantize round-trips). |
| quantized (affine_qmv / qvm / qmm — matvec / matmul) | ✓ | ✓ | ✓ | `mlx/quantized.rs` — **int4 perf**: hand-unrolled `mt_qmv` (8-row-per-TG decode, mirrors MLX `qmv_fast`) + `mt_qmm` / `_bm2` / `_bm4` M-batched variants + `mt_qmm_mma` / `_m16` simdgroup-matrix MMA prefill + `mt_qmm_mma_mpp` (MPP) + `mt_qmm_nax` (NAX). **int8 perf** (this PR): pack-aligned `mt_qmv_int8_fast` (8-row-per-TG decode) + `mt_qmm_int8_fast` / `_bm2` / `_bm4` + `mt_qmm_mma_int8` / `_m16_int8` (4-SG 2×2 MMA) + `mt_qmm_mma_mpp_int8` + `mt_qmm_nax_int8` (`mlx/quantized_mpp_int8.rs`, `mlx/quantized_nax_int8.rs`). int8 was previously correctness-only via the scalar `_b8` kernel below; the new pack-aligned variants (4 bytes/u32, byte-shift extract) close the ~6–8× perf gap. **All bit-widths × all dtypes**: generic `mt_{qmv,qvm,qmm}_b{3,4,5,6,8}` family (one-simdgroup correctness kernel, lane-strided K-walk + `simd_sum`, generic over `T` for bf16, macro-parameterised — pow2 widths pack-aligned, odd widths via the two-word bit-stream extract). **Odd-bitwidth MMA** (`ek/t3-quant-completeness`): `mt_qmm_mma_b{3,5,6}` — full 4-SG 2×2 simdgroup-matrix BM=BN=BK=32 MMA for int3/5/6, using the straddle-aware two-word bit-stream dequant in the K-loop inner body; closes the gap where int3/5/6 previously fell back to the scalar `_b{3,5,6}` kernel for any M>1 dispatch (verified by `qmm_mma_b356_gpu_correctness`). `qvm` is the transposed-W (`[K,N]`) vecmat — for int4 a new perf-tuned `mt_qvm_int4_fast` (8-col-per-TG, MLX `qvm_fast` shape) supplements the scalar family. Verified by `quantized_family_gpu_correctness` + the per-variant `qmv_int8_fast` / `qmm_int8_fast` / `qmm_mma_int8` / `qmm_mpp_int8` / `quantized_nax_int8` / `qvm_int4_fast` correctness oracles. The dynamic-M batched-prefill driver `mlx::quantized_mma_dynamic_m` is the M=2..16 selector. |
| quantized (gather_qmv / gather_qmm — gather variants) | ✓ | ✓ | ✓ | `ffai/moe.rs` → `mt_moe_gather_qmm_int4` (the int4 affine grouped-gather quantized matmul — per-row expert routing via a CSR `expert_offsets` walk, matching MLX's `gatherQuantizedMM`) **plus** the wider-precision family `mt_moe_gather_qmm_b{3,5,6,8}` closing the non-int4 gap (int8 pack-aligned, int3/5/6 via the two-word bit-stream extract; same routing + group-indexed scale/bias + `simd_sum` body). `gather_qmv` is the M=1 row of the per-row-routed `gather_qmm` body — no separate kernel needed. **int4 perf**: MMA path `mt_moe_gather_qmm_mma_int4{,_bm16}` + `_m8` (decode), MPP scale-ups `bm{8,16,64}_mpp` (`ffai/moe_mpp_bm{8,64}.rs`, `ffai/moe_mpp.rs`). **int8 perf** (this PR): pack-aligned `mt_moe_gather_qmm_mma_int8` (1-SG MMA, decode) + `mt_moe_gather_qmm_mma_int8_bm16_mpp` (`ffai/moe_mpp_int8.rs`) + `mt_moe_gather_qmm_mma_int8_bm8_mpp` (`ffai/moe_mpp_bm8_int8.rs`, direct-input cooperative tensors for tiny-M routing) + `mt_moe_gather_qmm_mma_int8_bm64_mpp` (`ffai/moe_mpp_bm64_int8.rs`, 4-SG 2×2 for long-context prefill). int8 MoE was previously routed through the slower bit-stream `_b8` MMA; the pack-aligned variants beat that by ~2× on top of the ~6–8× win the bit-stream kernel had over the scalar family. All MPP kernels stage `bf16` activations through `half` cooperative tensors via the DSL's `coop_stage(T)` (Apple `matmul2d` mishandles `bfloat` coop tensors); the MPP/NAX bodies are all expressed via `coop_tile_*` (PR #149 + #152, no `Op::InlineMsl`). **Short-prefill MoE m={16,32}** (`ek/moe-int4-m16-m32-unroll`): `mt_moe_gather_qmm_int4_m16` (grid `[m_out/16, T_rows, 1]`, 16 hand-unrolled cells, TPG=32) and `mt_moe_gather_qmm_int4_m32` (grid `[m_out/32, T_rows, 1]`, 32 hand-unrolled cells, TPG=32) — same expert routing + int4 dequant + `simd_sum` reduction as `_m8`, extended to 16/32 adjacent output cells per TG; 2×/4× fewer TGs than `_m8` at the same M, reducing dispatch overhead at short prefill. The cells are written out individually (`acc0..acc15` / `acc0..acc31`) because the DSL doesn't lower a runtime-indexed mutable array — the first T3 draft using `let mut acc = [0.0f32; m_batch]; acc[loop_var] = ...` emitted phantom SSA references the codegen never declared. Both generic over `T` (f32/f16/bf16). Verified by `moe_gather_qmm_int4_m16_m32_correctness` (cosine = 1.000, max |Δ_m8| = 0 across all 6 test cases). Plus `moe_gather_qmm_gpu_correctness` + the int8 oracles (`moe_gather_qmm_mma_int8_gpu_correctness`, `moe_gather_qmm_mpp_int8_correctness`, `moe_gather_qmm_mpp_bm{8,64}_int8_correctness`). Bare-tensor `ffai/gather.rs` exists but is non-quantized. |
| moe (router top-k + permute + unpermute orchestration) | ✗ | ✓ | ✓ | `ffai/moe.rs` → `mt_moe_router_topk<T>`, `mt_moe_permute<T>`, `mt_moe_unpermute<T>`. MoE expert-routing orchestration for Qwen3.6-35B-A3B / Qwen3-Coder-30B-A3B end-to-end serving. The grouped quantized BGEMM that fuses the per-expert FFN matmuls into one dispatch is landed — `mt_moe_gather_qmm_int4` + the MMA / MPP variants (see the `quantized (gather_*)` row). |
| dequant_gather (quantized embedding-table gather) | ✗ | ✗ | ✓ | `ffai/dequant_gather.rs`. int{3,4,5,6,8} all bit-widths. FFAI-specific, no MLX counterpart. |
| dequant_gemv (quantized GEMV, FFAI flavour) | ~ (subset of `quantized.metal`) | ~ | ✓ | `ffai/dequant_gemv.rs` — `dequant_gemv_int{3,4,5,6,8}<T>` (one-row-per-TG pack/element-strided, generic `T`) **plus** the perf-tuned `dequant_gemv_int4_fast<T>` (8-row-per-TG, mirrors MLX `qmv_fast` geometry — added in this PR). The original `dequant_gemv_int4` stays because FFAI's GPU-router opts into its `indirect` variant (`dequant_gemv_wants_indirect()`); the `_fast` variant is for callers that bench it directly. FFAI-tuned shape. Coexists with the partial `mt_qmv_f32` port. |
| fp_quantized (fp4/fp8 quant + dequant) | ✓ | ✓ | ✓ | `mlx/fp_quantized.rs` → `mt_fp4_quant_dequant` (fp4 E2M1) **plus** `mt_fp8_e4m3_quant_dequant` / `mt_fp8_e5m2_quant_dequant` — the fp8 quantize-dequantize round-trip for both MLX fp8 formats (e4m3: 3-mantissa, ±448; e5m2: 2-mantissa, ±57344). No new DSL dtype needed: fp8 quant-dequant is a pure arithmetic transform — per-group max-scale, then round each value's mantissa to the format's bit count via `floor`/`log2`/`exp2`/`round` (`e = clamp(floor(log2(norm)), e_min, e_max)`; `quantum = exp2(e − mantissa_bits)`; `q = round(norm/quantum)·quantum`). Exact for every fp8 normal/subnormal, saturating (no NaN/Inf) — matching MLX's `mxfp8`/`nvfp8`. Verified by `fp_quantized_fp8_gpu_correctness` (e4m3 + e5m2 round-trips, sign preservation, exact-value + saturation cases). |
| fp_quantized_nax | ✓ | ✓ | ✓ | `mlx/fp_quantized_nax.rs` → `mt_fp_qmm_nax<T>` — fp4 (E2M1) quantized matmul via Apple `mpp::tensor_ops::matmul2d` (NAX tensor cores). fp4 counterpart of `quantized_nax`: same dequant-into-TG-memory + one cooperative `matmul2d` per simdgroup per K-block, but the int4 affine nibble-dequant is swapped for an fp4 E2M1 codebook lookup (`{0,0.5,1,1.5,2,3,4,6}` magnitude LUT + sign bit, scale-only — no bias; see MLX `fp4.h`). 8 fp4 codes per `u32` pack; `GROUP_SIZE = 32` (one group per BK-block). Expressed in the `#[kernel]` DSL + `coop_stage(T)` for bf16 (PR #152 — no more `Op::InlineMsl`). `#[cfg(feature = "nax")]`-gated; needs macOS 26+ / Metal 4. Verified by `fp_quantized_nax_gpu_correctness` across f32/f16/bf16 vs a triple-loop fp4-dequant oracle. |
| fp_quantized_mma | ✗ | ✗ | ✓ | `mlx/fp_quantized_mma.rs` (`ek/t3-quant-completeness`) → `mt_fp4_qmm_mma<T>` + `mt_fp8_e4m3_qmm_mma<T>` — simdgroup-matrix BM=BN=BK=32 MMA for fp4 and fp8 E4M3 weight formats. Both are the same 4-SG 2×2 MMA scaffold as `mt_qmm_mma` / `mt_qmm_mma_b{3,5,6}`, but the weight dequant in the K-loop body is format-specific: **fp4** uses the fp4 E2M1 codebook (`two_m_int` trick — `{0,0.5,1,1.5,2,3,4,6}` magnitudes, scale-only, 8 codes/u32); **fp8 E4M3** is a two-pass byte-shift extract + biased-exponent decode (biased exp `e_biased = e - 3`, subnormal path when `e_biased == 0`, scale per group, 4 codes/u32). `GROUP_SIZE = 32` for both; scale-only (no bias), matching MLX's MX-format convention. No NAX/MPP gating — runs on any M1+. Fills the gap between the fp4/fp8 scalar round-trip kernels (`fp_quantized.rs`) and the NAX-gated `fp_quantized_nax`; provides a pure-simdgroup-MMA prefill path for fp4/fp8 weights on older hardware. Verified by `fp_quantized_mma_gpu_correctness` across f32/f16/bf16, multiple tile sizes, vs triple-loop dequant oracles. |
| quantized_nax | ✓ | ✓ | ✓ | `mlx/quantized_nax.rs` → `mt_qmm_nax<T>` — int4 quantized matmul via Apple `mpp::tensor_ops::matmul2d` (NAX tensor cores). Expressed in the `#[kernel]` DSL via the `coop_tile_*` intrinsics + `coop_stage(T)` for bf16-safe staging (PR #152 — no more `Op::InlineMsl`). MPP counterpart of `mt_qmm_mma` — same int4-dequant-into-TG-memory algorithm, one cooperative `matmul2d` per simdgroup per K-block. The int8 sibling `mt_qmm_nax_int8` (`mlx/quantized_nax_int8.rs`) lands in this PR — same algorithmic skeleton, byte-shift extract (2 packs/lane). `#[cfg(feature = "nax")]`-gated; needs macOS 26+ / Metal 4. Verified by `quantized_nax_gpu_correctness` + `quantized_nax_int8_gpu_correctness` across f32/f16/bf16. |
| fft (radix + readwrite) | ✓ | ✓ | ✓ | `mlx/fft.rs` → `mt_fft_n{32,64,128,256,512,1024}<T>`. Iterative radix-2 Cooley–Tukey FFT along the last axis (power-of-two N), one kernel covering forward + inverse via an `inv` constexpr. Complex numbers without a complex type: real / imaginary planes are two parallel real `f32` buffers, the butterfly's complex multiply expands to the four-real-mul form — the same representation `mel_spectrogram` / `vocoder` use. Bit-reversal load + `log2(N)` `threadgroup`-buffered butterfly stages; genuine O(N log N). **Bluestein chirp-Z path** (non-power-of-2): `mt_fft_bluestein_preprocess<T>` + `mt_fft_bluestein_chirp_filter` + `mt_fft_bluestein_cmul<T>` + `mt_fft_bluestein_postprocess<T>` — three pointwise kernels wrapping the existing pow2 FFT to implement arbitrary-length DFT in O(N log N); M=1024 padding covers N=400 (Whisper) and N=480 (Whisper large-v3). The prime-length (Rader) path remains a follow-up (Bluestein subsumes it for our target lengths). Verified by `fft_gpu_correctness` (pow2 forward/inverse/round-trip; f32/f16/bf16) + `fft_bluestein_gpu_correctness` (N=400, N=480 vs naive DFT; f32/f16/bf16). |
| hadamard (hadamard_n + hadamard_m) | ✓ | ✓ | ✓ | `mlx/hadamard.rs` → `mt_hadamard_n{64,128,256,512,1024}<T>` (power-of-2 FWHT via log2(N) butterfly passes). `mlx/hadamard_m.rs` → `mt_hadamard_m{12,20,28}<T>` (non-power-of-2 M factor; Sloane-table bitmask accumulate, expressed in the `#[kernel]` DSL — PR #152 ported from `Op::InlineMsl`; sign arrays verified orthogonal). Generic over `T` (f32/f16/bf16). Verified by `hadamard_m_gpu_correctness`. |
| fence | ✓ | ✓ | — | **Intentionally out of scope** — a GPU-side sync primitive, not a compute op, and not a `#[kernel]`. See [§ Fence ops](#fence-ops--intentionally-out-of-scope). |
| gather (bare-tensor embedding lookup) | ✓ (via indexing/) | ✓ | ✓ | `ffai/gather.rs` → `ffai_gather<T>`. FFAI's embedding-table gather. |
| indexing (scatter, scatter_axis, gather_axis, gather_front, masked_scatter) | ✓ | ✓ | ✓ | `mlx/gather_axis.rs` + `mlx/scatter_axis.rs` → `mt_gather_axis` / `mt_scatter_axis` (contiguous along-axis); `mlx/indexing.rs` → `mt_gather_front` (first-axis row gather), `mt_scatter` (first-axis row scatter, no-reduce assignment form), `mt_masked_scatter` (per-element masked gather-scatter). All five are one-thread-per-output Grid3D with an `n_elems` bounds guard. Verified by `gather_axis_gpu_correctness` / `scatter_axis_gpu_correctness` / `indexing_gpu_correctness`. |
| aura_encode (codebook quantize, fused) | ✗ | ✓ (`turbo_fused_encode` in `turbo_quant.metal`) | ✓ | `ffai/aura_encode.rs`. Bit-widths 2/3/4/8. Renamed turbo_*→aura_*. |
| aura_dequant_rotated (bulk dequant to rotated codec space) | ✗ | ✓ (`turbo_dequant_rotated` in `turbo_quant.metal`) | ✓ | `ffai/aura_dequant_rotated.rs`. bits ∈ {2,3,4,8}. Renamed. |
| aura_score (compressed-domain Q·K) | ✗ | ✓ (`turbo_score`) | ✓ | `ffai/aura_score.rs`. bits ∈ {2,3,4,8}. Generic over `T` — f32/f16/bf16 (PR #152). |
| aura_value (compressed-domain value aggregation) | ✗ | ✓ (`turbo_value` in `turbo_quant.metal`) | ✓ | `ffai/aura_value.rs`. Sparsity-threshold guard mirrors MLX upstream. Generic over `T` — f32/f16/bf16 (PR #152). |
| aura_flash_p1 (compressed-domain flash pass 1) | ✗ | ✓ (`turbo_flash_p1` in `turbo_flash.metal`) | ✓ | `ffai/aura_flash_p1.rs` → non-causal `aura_flash_p1_{kb4_vb2,kb4_vb4}_{d64,d128}` (4 instantiations) **plus** the causal variant `aura_flash_p1_causal_kb4_vb2_{d64,d128}`. The causal kernel clamps the per-token inner loop at `q_position + 1` (a constexpr-folded `causal_end` select) — every key strictly after the query token is masked out, matching `turbo_flash_p1`'s `causal` template flag. Generic over `T` — f32/f16/bf16 (PR #152, was f32-only before). Verified by `aura_flash_gpu_correctness` (end-to-end pair) + `aura_flash_p1_causal_gpu_correctness` (full-visibility ≡ non-causal, mid-cutoff masks later blocks). |
| aura_flash_pass2 (cross-block online-softmax merge) | ✗ | ✓ (`turbo_flash_pass2`) | ✓ | `ffai/aura_flash_pass2.rs`. fp32 accumulators → `T` final. Generic over `T` — f32/f16/bf16 (PR #152, was bf16-only before). |
| turbo_flash_sdpa (fused single-pass SDPA, sinks variant) | ✗ | ✓ (`turbo_flash_sdpa.metal`) | ✓ | `ffai/aura_flash_sdpa.rs` → `aura_flash_sdpa_kb*_vb*_d*<T>`. Single-pass online-softmax over compressed K/V with attention sinks + sliding-window causal mask. Single-simdgroup shape (token-parallelism a perf follow-up). |
| flash_quantized_sdpa (single-pass quantized SDPA, affine cache) | ✗ | ✓ (`flash_quantized_sdpa.metal`) | ✓ | `ffai/flash_quantized_sdpa.rs` → base `flash_quantized_sdpa_b{4,8}_d{64,96,128,256,512}<T>` (5 head dims × 2 bit-widths = 10 base kernels) + `flash_quantized_sdpa_bool_mask_b{4,8}_d{64,128,256}<T>` (bool mask gate) + `flash_quantized_sdpa_float_mask_b{4,8}_d{64,128,256}<T>` (float logit-bias). Single-pass online-softmax SDPA over affine-quant KV, with sinks + sliding-window. d=96 covers GPT-NeoX (group_size=32 since 96 isn't a multiple of 64); d=512 covers Gemma 4 global attention and dispatches at 256 threads/TG (16 elems/lane pushes `maxTotalThreadsPerThreadgroup` below 1024). Bool mask (`mask_bool: Tensor<u32>`, one `u32` per token, 0 = skip) ANDs with the causal gate; float mask (`mask_float: Tensor<T>`, shape `[B*nQ, tokens]`) adds a per-token logit bias (ALiBi / T5-relative). Float mask loads the same address on all 32 lanes so all get the same scalar bias after `simd_sum` — no `select(lane==0)` trick needed. Bool/float mask coverage at d={96,512} is a follow-up. Verified by `flash_quantized_sdpa_gpu_correctness` (base d=64..512: 6 tests) + `flash_quantized_sdpa_mask_gpu_correctness` (bool/float × b4/b8 × f32/bf16 + zero-bias / all-visible regression guards; 10 tests). |
| gated_delta (GatedDeltaNet recurrence) | ✗ | ✓ (`gated_delta.metal`) | ✓ | `ffai/gated_delta.rs` → `mt_gated_delta_step<T>` (single-token decode) + `mt_gated_delta_chunk<T>` (chunked-prefill). GDN linear-attention for the Qwen3.5 / 3.6 / 3.6-MoE hybrid models (≈75 % of layers). The MMA-tiled chunked-WY prefill perf variant `mt_gated_delta_wy_chunk` and the fused prep+recurrence variant `mt_gated_delta_prep_step` (`ffai/gated_delta_prep.rs`) are both landed — the latter collapses conv-split + per-head q/k RMSNorm + g/beta + the recurrence into one dispatch, cutting 3 host commit+wait pairs per GDN layer down to 1 (Qwen3.6-A3B decode unlock). |
| gated_delta_replay (tape capture + state replay) | ✗ | ✓ (`gated_delta_replay.metal`) | ✓ | `ffai/gated_delta_replay.rs` → `gated_delta_step_record<T>` (forward + delta-tape) + `state_replay<T>` (branchless accepted-prefix re-fold). Speculative-decode rollback on GDN. |
| ssm_step (Mamba 2 SSD single-token decode) | ✗ | ✓ (`ssm.metal`) | ✓ | `ffai/ssm.rs` → `ssm_step<T>`, `mt_ssm_step<T>`. Faithful port; `mlx_src: None` because pinned MLX upstream doesn't ship `ssm.metal`. Will graduate to `mlx/` when pin moves. |
| conv1d_causal_step (depthwise SSM conv stream) | ✗ | partial (subset of SSM toolchain) | ✓ | `ffai/ssm.rs` → `conv1d_causal_step<T>`. fp32 state recurrence. |
| ssm_replay (sequential tape capture + replay) | ✗ | ✓ (`ssm_replay.metal`) | ✓ | `ffai/ssm_replay.rs` → `ssm_step_record<T>` (SSD forward + dA/dBx tape) + `ssm_replay<T>` (re-fold first k entries). Spec 040 Mamba/Mamba2 state replay. |
| fused_gate_activation (silu/gelu × up gate) | ✗ | ✓ (`fused_gate_activation.metal`) | ✓ | `mlx/fused_gate_activation.rs` → `mt_fused_gate_gelu` (gelu-tanh approximation) + `mt_fused_gate_clipped_swiglu` (GPT-OSS clipped variant — `[-7,7]` clamp, `sigmoid(1.702·g)` gate, `+1` up bias). The `silu` variant ships separately as `mlx/swiglu.rs` (see the `swiglu` row). One-thread-per-output Grid3D; the MLX `single_row` / `looped` threadgroup-tiling split is a perf detail, not a separate op. Verified by `fused_gate_activation_gpu_correctness`. |
| rms_norm_residual (RMSNorm + residual add fused) | ✗ | ✓ (`rms_norm_residual.metal`) | ✓ | `ffai/rms_norm_residual.rs` → `ffai_rms_norm_residual<T>`. Reduction-mode, `N = TPG*4`; mirrors `mt_rms_norm` + a residual-add input. ~90 saved dispatches/token on Gemma4-30 type configs. |
| rms_norm_rope (RMSNorm + RoPE fused) | ✗ | ✓ (`rms_norm_rope.metal`) | ✓ | `ffai/rms_norm_rope.rs` → `ffai_rms_norm_rope<T>`. Reduction-mode, paired-layout RoPE; `TPG = axis_size/2`. Q/K post-projection norm+rope in one dispatch. |
| rms_norm_qgemv (RMSNorm + quantized GEMV fused) | ✗ | ✓ (`rms_norm_qgemv.metal`) | ✓ | `ffai/rms_norm_qgemv.rs` → `ffai_rms_norm_qgemv<T>` (one-row-per-TG correctness shape, int4) **plus** `ffai_rms_norm_qgemv_fast<T>` (8-row-per-TG, 2 SG × 4 rows, int4, mirrors MLX `rms_norm_qmm` — added in the int-quant perf PR to close the perf follow-up flagged in prior audits) **plus** `ffai_rms_norm_qgemv_int8_fast<T>` (8-row-per-TG int8 variant, `ek/t3-quant-completeness`): same 2-SG × 4-row geometry but with byte-shift int8 weight extract (4 bytes/u32, packs_per_row = in_dim/4); requires in_dim % 512 == 0, group_size == 64; closes the parity gap where the RMSNorm-fused path for int8 models had no perf variant. All three Reduction-mode; the non-fast kernel stays for callers that don't satisfy the 8-row shape constraint. Verified by `rms_norm_qgemv_int8_fast_gpu_correctness` (f32/f16/bf16, in_dim ∈ {512, 1024}). |
| batched_qkv_qgemv (Q/K/V 4-bit qGEMV → 1 dispatch) | ✗ | ✓ (`batched_qkv_qgemv.metal`) | ✓ | `ffai/batched_qkv_qgemv.rs` → `ffai_batched_qkv_qgemv<T>` (one-row-per-TG correctness shape) **plus** `ffai_batched_qkv_qgemv_fast<T>` (8-row-per-TG, 2 SG × 4 rows, GQA-guarded — added in this PR). Reduction-mode, int4; `program_id::<2>()` selects Q/K/V, output concatenated `[Q\|K\|V]`. Decode-form fused QKV projection. |
| kv_cache_update (raw bf16/fp16 single-token append) | ✗ | ✗ | ✓ | `ffai/kv_cache.rs` → `kv_cache_update<T>`. FFAI-only; raw cache append. |
| kv_cache (affine-quant int4/int8/fp8 quantize + bulk dequant) | ~ (via `quantized.metal` affine_quantize) | ~ | ✓ | `ffai/kv_cache.rs` — `quantize_kv` + `bulk_dequant_kv` for int4/int8. FFAI-specific cache layout. **fp8 KV cache** (`ek/t3-quant-completeness`): `quantize_kv_fp8_{e4m3,e5m2}` (single-token append — per-group amax → scale, 4 fp8 codes/u32, scale-only, no bias) + `bulk_dequant_kv_fp8_{e4m3,e5m2}` (bulk dequant: byte-shift extract + biased-exp decode). The fp8 macro takes the mantissa width as **two literals** (`$mant_f` float + `$mant_i` u32) instead of one — the DSL body parser doesn't handle Rust's `as u32` cast on literal floats (only `.cast::<T>()` on `Tensor`/`Value`), so a single `let mant_bits = $mant as u32;` lowers to a phantom SSA value the codegen references but never declares. Splitting the literal sidesteps the lowering gap. Format constants: E4M3 — mantissa_bits=3, e_bias=-6.0, max_val=448.0; E5M2 — mantissa_bits=2, e_bias=-14.0, max_val=57344.0. Closes the gap where fp8 KV cache required a host-side quantize+unpack round-trip; round-trip correctness verified by `kv_cache_fp8_gpu_correctness` (f32/f16/bf16 × {e4m3, e5m2}, cross-slot isolation). |
| sampling (softmax + categorical inverse-CDF) | ✗ | ✗ | ✓ | `ffai/sampling.rs` → `softmax_categorical_sample`. Companion to `ffai_argmax` for `T > 0` decode. |
| logits processors (temperature, repetition penalty, top-k / top-p / min-p masks) | ✗ | ✗ | ✓ | `ffai/logits_{processors,topk,top_p,min_p}.rs` → `logits_temperature`, `logits_repetition_penalty`, `logits_topk_mask`, `logits_top_p_mask`, `logits_min_p_mask` (all generic `T`). In-place decode-form sampler stages composed before `softmax_categorical_sample`. FFAI-only. |
| sdpa_decode_d512 (head_dim=512 SDPA decode — Gemma 4 global) | ✗ | ✗ | ✓ | `ffai/sdpa_decode_d512.rs` → `ffai_sdpa_decode_d512<T>`. head_dim=512 specialization for Gemma 4's global-attention layers; dispatches at 512 threads/TG (the 16-wide per-lane footprint caps the pipeline below 1024). FFAI-only; verified by `sdpa_decode_d512_gpu_correctness`. Consolidation pass (2026-05-21). |
| rms_norm_wide (RMSNorm for rows past the 4096-element cap) | ✗ | ✗ | ✓ | `mlx/rms_norm.rs` → `mt_rms_norm_wide<T>`. Strided wide-row variant for large-hidden models (Gemma 4 31B, hidden 5376) that exceed the standard `mt_rms_norm` 1024-thread × 4-element single-row cap. Verified by `rms_norm_wide_gpu_correctness`. Consolidation pass (2026-05-21). |
| sdpa_decode + learned attention sink (GPT-OSS-20B) | ✗ | ~ | ✓ | `ffai/sdpa_decode.rs` → `ffai_sdpa_decode` `has_sink` / `sink_logit` constexprs. GPT-OSS-20B's per-head learned attention-sink logit now folds into the cross-simdgroup softmax denominator on-GPU as a virtual key (score `sink_logit`, value 0) — removing the host-side post-hoc rescale that previously cost a CPU sync per attention layer. `has_sink == 0` masks the term out, keeping the dense / sliding-window paths bit-identical to the pre-sink kernel. Distinct from the `sink_end` sink-*token* range. Verified by `sdpa_decode_gpu_correctness` (`sdpa_decode_learned_sink_matches_cpu_f32`). |
| gated_rmsnorm (fp32-in gated RMSNorm → activation dtype) | ✗ | ✗ | ✓ | `ffai/gated_rmsnorm.rs` → `ffai_gated_rmsnorm<T>`. Fused Qwen3.5 / 3.6 GDN post-step `out = w·rmsNorm(y)·silu(z)`: `y` arrives fp32 (the `gated_delta` recurrence output), the gate `z` / weight `w` / output are activation-dtype `T`. Reduction-mode, `N = TPG*4`, mirrors `mt_rms_norm` with the fp32-in / `T`-out dtype split and the `silu(z)` gate. Closes the per-GDN-layer host-side CPU sync (≈75 % of Qwen3.5/3.6 layers). Verified by `gated_rmsnorm_gpu_correctness`. |
| ssm_step (2D `A_log` / per-(head,state) decay — Jamba) | ✗ | ~ | ✓ | `ffai/ssm.rs` → `ssm_step_a2d<T>`. The 2-D-`A_log` variant of `ssm_step`: carries a per-(channel, state) `A_log` of shape `[n_heads*head_dim, state_dim]` so the decay `exp(-exp(A_log)·dt)` varies with the state index, moving Jamba's Mamba 1 selective scan onto the GPU (it previously ran host-side). Same Grid3D geometry as `ssm_step` — one thread per `(head, d)`, state `h` in fp32. The other Mamba 2 families (Mamba2, FalconH1, NemotronH, GraniteMoeHybrid) use the scalar-`A` kernel and are unaffected. Verified by `ssm_step_a2d_gpu_correctness` (f32/f16/bf16). |
| conv2d (vision patch conv — im2col + tiled GEMM) | ✓ | ✓ | ✓ | `ffai/conv2d.rs` → `conv2d_patch14` / `conv2d_patch16` (fixed-patch variants, kernel + stride baked in) + `conv2d_generic` (runtime kh/kw/stride/pad). NCHW input, OIHW weight; direct conv (implicit im2col, one thread per output). Generic `T`; verified by `conv2d_gpu_correctness`. Phase 6.5 VLM. |
| patch_embed (fused image unfold + linear projection) | ✗ | ✗ | ✓ | `ffai/patch_embed.rs` → `patch_embed<T>`. Fused image-unfold + linear projection — gathers each patch's pixels and dots them with one weight row, no intermediate unfolded buffer. NCHW image, flat `[hidden, patch_dim]` weight, `[num_patches, hidden]` output. **MMA-tiled perf path**: `ffai/patch_embed_mma.rs` → `patch_embed_mma<T>` — implicit-patch-unfold + 4-SG 2×2 simdgroup-matrix MMA, 32×32 output tile (`hidden` and `num_patches` divisible by 32); targets ViT-L/H shapes (hidden=1024/1280). FFAI-specific; verified by `patch_embed_gpu_correctness` + `patch_embed_mma_gpu_correctness`. Phase 6.5 VLM. |
| rope_2d (2D positional RoPE for vision tokens) | ✓ | ✓ | ✓ | `ffai/rope_2d.rs` → `ffai_rope_2d<T>`. 2D RoPE over a (row, col) token grid — head_dim split into a row half and a column half, each running rotate-half RoPE. Consumes a per-token `(row, col)` pair. Generic `T`; verified by `rope_2d_gpu_correctness`. Phase 6.5 VLM. |
| mel_spectrogram (STFT + log-Mel filterbank) | ✓ | ✓ | ✓ | `ffai/mel_spectrogram.rs` → `mel_spectrogram<T>` (in-thread direct DFT, single-dispatch path) **plus** the radix-FFT path `mel_stft_window<T>` → `mt_fft_n{n_fft}<T>` → `mel_filterbank<T>` (three kernels, O(N log N) instead of O(N²)). All three are generic over `T` — f32/f16/bf16 (PR #152 — was f32/f16 only). Verified by `mel_spectrogram_gpu_correctness`. Phase 7. |
| audio_conv1d (wide-stride 1D conv — STT patch embed) | ✓ | ✓ | ✓ | `ffai/audio_conv1d.rs` → `audio_conv1d<T>`. Dense wide-stride multi-channel 1D conv (NCL); distinct from the depthwise `conv1d_causal_step` SSM-stream conv. Generic `T`; verified by `audio_conv1d_gpu_correctness`. Phase 7. |
| vocoder / iSTFT (TTS waveform synthesis) | ✓ | ✓ | ✓ | `ffai/vocoder.rs` → `vocoder_istft<T>`. Inverse-STFT overlap-add — one thread per output sample gathers every covering frame, inverse-DFTs with Hermitian symmetry, COLA-normalises (no atomics). Generic `T`; verified by `vocoder_gpu_correctness`. Phase 7. |

## Notes on counting decisions

A few rows mix multiple `.metal` files into one op or split one file into multiple ops:

- **`sdpa_vector*` rows.** Upstream `sdpa_vector.h` defines `sdpa_vector`, `sdpa_vector_2pass_1`, `sdpa_vector_2pass_2`. Counted as two ops: `sdpa_vector` (single pass) + `sdpa_vector_2pass` (two-pass pair).
- **AURA stack.** Each codec stage (`encode`, `dequant_rotated`, `score`, `value`, `flash_p1`, `flash_pass2`) is a separate row — they're separately compiled kernels with their own dispatch shapes. The `turbo_flash_sdpa` (sinks-fused single-pass) is also its own row.
- **`steel/` family.** Each kernel file in `steel/{attn,conv,gemm}/kernels/` becomes one op row; per-block-shape instantiations are not counted separately. `steel_attention` (scalar-flash) and `steel_attention_mma` (simdgroup-MMA) are counted as two rows because they are separately compiled kernels with different lowering strategies; the bf16-tuned `mt_sdpa_prefill_mma_bf16` is folded into the MMA row as a perf specialization.
- **`quantized.metal`.** Split into three rows by semantic operation (quant/dequant, qmv/qvm/qmm matmul, gather-qmv/qmm) rather than by template instantiation. Quantized-NAX and FP-quantized-NAX are separate rows because the metaltile modules exist (empty) and have separate feature gates. `fp_quantized_mma` is a distinct fourth row for the simdgroup-MMA fp4/fp8 matmul path (`mlx/fp_quantized_mma.rs`) that is not NAX-gated — it fills the M>1 perf slot for fp4/fp8 weights on M1+ hardware without the NAX cooperative-tensor overhead.
- **`indexing/`** is one row covering scatter / scatter_axis / gather_axis / gather_front / masked_scatter. Bare `gather` is its own row because metaltile has a dedicated FFAI port.
- **`moe`** is one row for the routing/permute/unpermute orchestration kernels in `ffai/moe.rs`. The grouped quantized BGEMM (`mt_moe_gather_qmm_*`, incl. the MMA / MPP-NAX variants) is counted under the `quantized (gather_*)` row.
- **`logits processors`** is one row for the FFAI sampler-stage kernels (`temperature`, `repetition_penalty`, `topk` / `top_p` / `min_p` masks). FFAI-only, no MLX counterpart.
- **Cells marked `~`** indicate metaltile has a partial port — typically one bit-width, one dtype, or one block shape where upstream has many. Read the notes column for the specific gap.

## Perf follow-ups — landed (throughput refinements)

Op coverage is complete — every row is ✓ except `fence` (out of scope).
The four throughput refinements below have **all landed**; each is
purely additive — the original kernels are kept for callers that don't
want the split / extra dispatch.

1. **`steel_gemm_fused` block-shape coverage** ✓ — added the
   `64×64×16 / 4×2` shape (8 simdgroups, TPG=256): **~40% faster** than
   the prior-best 2×2 on the 4096³ bench (f32 1.4 vs 1.0 GB/s) — the
   extra simdgroups hide the device-memory fragment loads. Also added
   the M-skewed `64×32 / 1×2` and low-TPG `32×32 / 1×2` tiles, for 7
   shapes total to feed a future per-shape dispatcher.
2. **`moe` gather_qmm bit-widths beyond int4** ✓ — the `gather_qmm_mma!`
   macro carries the tiled-MMA geometry for any bit-width via a
   bit-stream weight coop-dequant; instantiated as
   `mt_moe_gather_qmm_mma_b{3,5,6,8}`, so int3/5/6/8 MoE experts now hit
   the matrix engine instead of the scalar fallback.
3. **Winograd kernel split** ✓ — `winograd_filter_transform_3x3`
   pre-transforms every filter into its 4×4 `U` once;
   `winograd_conv2d_3x3_split` then loads the precomputed `U` instead of
   re-running `G·g·Gᵀ` per output tile, removing the O(tiles) redundant
   transform work.
4. **Radix-FFT STFT path** ✓ — `mel_stft_window` → `mt_fft_n{n_fft}` →
   `mel_filterbank` replaces `mel_spectrogram`'s in-thread direct DFT
   (recomputed per Mel bin) with one O(N log N) FFT per frame. `n_fft`
   must be a power of two; the single-kernel `mel_spectrogram` stays for
   non-pow2 sizes. **Non-pow2 Bluestein extension** ✓ (this PR) —
   `mt_fft_bluestein_{preprocess,chirp_filter,cmul,postprocess}<T>` in
   `mlx/fft.rs` add Whisper n_fft=400 and n_fft=480 support via chirp-Z
   transform padded to M=1024; verified by `fft_bluestein_gpu_correctness`.
5. **int4 + int8 quantized perf paths** ✓ (2026-05-23 — this PR) —
   closes the int8-has-no-perf-path gap and the four int4
   follow-ups flagged across prior audits. **int8 dense GEMM**:
   `mt_qmv_int8_fast`, `mt_qmm{,_bm2,_bm4}_int8_fast` (8-row-per-TG
   decode + small-batch prefill), `mt_qmm_mma_int8` + `_m16_int8`
   (4-SG 2×2 simdgroup-matrix MMA prefill), `mt_qmm_mma_mpp_int8`
   and `mt_qmm_nax_int8` (Apple `mpp::tensor_ops::matmul2d` int8
   prefill, new `mlx/quantized_{mpp,nax}_int8.rs` files).
   **int8 MoE BGEMM**: pack-aligned `mt_moe_gather_qmm_mma_int8`
   (1-SG MMA decode) + `_bm16_mpp` + `_bm8_mpp` (direct-input
   cooperative tensors for tiny-M routing) + `_bm64_mpp` (4-SG 2×2
   for long-context prefill, new `ffai/moe_mpp{,_bm8,_bm64}_int8.rs`
   files). int8 routes were previously through the slower bit-stream
   `_b8` MMA path; the new pack-aligned variants beat that by ~2×
   on top of the original ~6–8× win the bit-stream kernel had over
   the scalar `_b8`. **int4 polish**: `ffai_rms_norm_qgemv_fast`
   (8-row-per-TG tiling, the perf follow-up flagged in #122 of this
   audit's previous revision), `ffai_batched_qkv_qgemv_fast` (same
   tiling, GQA-guarded), `dequant_gemv_int4_fast` (new 8-row
   variant, the original kept for FFAI's indirect router), and
   `mt_qvm_int4_fast` (perf-tuned int4 vecmat — `y = xᵀ · W`, MLX
   `qvm_fast` shape, 8-col-per-TG; previously only the scalar
   `_b4` kernel existed). All variants pass cosine ≥ 0.999 (f32 /
   f16) and ≥ 0.997 (bf16) against the scalar dequant oracle.
6. **Quantization coverage completeness** ✓ (`ek/t3-quant-completeness`) —
   fills the remaining holes identified after the int-quant perf PR:
   **Odd-bitwidth MMA**: `mt_qmm_mma_b{3,5,6}` (`mlx/quantized.rs`) —
   straddle-aware two-word bit-stream dequant in the BM=BN=BK=32 4-SG
   MMA body; int3/5/6 now hit the matrix engine for M>1 prefill instead
   of falling back to the scalar `_b{3,5,6}` kernel. Verified by
   `qmm_mma_b356_gpu_correctness` (4 tests per bit-width: f32 small,
   f32 multi-tile, f16, bf16). **fp4/fp8 MMA**: `mt_fp4_qmm_mma` +
   `mt_fp8_e4m3_qmm_mma` (`mlx/fp_quantized_mma.rs`) — same 4-SG 2×2
   simdgroup-MMA scaffold but with fp4 E2M1 codebook dequant / fp8 E4M3
   biased-exponent decode respectively; runs on any M1+ (no NAX gating).
   Verified by `fp_quantized_mma_gpu_correctness`. **int8 RMSNorm-fused
   GEMV**: `ffai_rms_norm_qgemv_int8_fast` (`ffai/rms_norm_qgemv.rs`) —
   8-row-per-TG byte-shift int8 GEMV with in-kernel RMSNorm; the
   `ffai_rms_norm_qgemv_fast` perf path was int4-only before this PR.
   Verified by `rms_norm_qgemv_int8_fast_gpu_correctness`. **fp8 KV
   cache**: `quantize_kv_fp8_{e4m3,e5m2}` + `bulk_dequant_kv_fp8_{e4m3,e5m2}`
   (`ffai/kv_cache.rs`) — per-group amax → scale quantize, biased-exp
   decode dequant. The macro takes the mantissa bit count as both a
   float (`$mant_f`) and a u32 (`$mant_i`) since the DSL body parser
   doesn't lower a `let mant_bits = $mant as u32;` literal cast (the
   first draft emitted a phantom `v_mant_bits` the codegen never
   declared). Closes the host-side fp8 KV round-trip — verified by
   `kv_cache_fp8_gpu_correctness`.

7. **Short-prefill MoE m={16,32}** ✓ (`ek/moe-int4-m16-m32-unroll`) —
   `mt_moe_gather_qmm_int4_m16` and `mt_moe_gather_qmm_int4_m32`
   (`ffai/moe.rs`): hand-unrolled 16-cell and 32-cell variants of the
   `_m8` scalar decode kernel, eliminating the runtime-indexed mutable
   array that caused the DSL codegen failure in `ek/t3-quant-completeness`.
   Each cell is written out explicitly with individually-named accumulators
   (`acc0..acc15` / `acc0..acc31`), `simd_sum`'d and stored at the end.
   Grid `[m_out/16, T_rows, 1]` and `[m_out/32, T_rows, 1]` respectively,
   TPG=32 (one simdgroup). Generic over `T` (f32/f16/bf16). Verified by
   `moe_gather_qmm_int4_m16_m32_correctness` (6 tests: cosine = 1.000,
   max |Δ_m8| = 0 across all cases).

8. **Attention head_dim coverage** ✓ (2026-05-23 — `ek/t1-attn-headdim`) —
   closes the head_dim gaps flagged in prior audit notes for four kernel
   families:
   - `steel_attention_nax`: added `mt_sdpa_prefill_nax_d{64,128,256}` via
     a D-chunk loop (32-wide slices) inside the outer K-block loop.
   - `mt_sdpa_vector`: added `mt_sdpa_vector_d{64,96,192,256}` — single-pass
     GQA decode at every production head_dim.
   - `sdpa_vector_2pass`: added pass1/pass2 pairs for head_dim ∈ {64, 96, 256};
     d=256 uses 4-buffer TG reuse to stay within the 32KB cap.
   - `flash_quantized_sdpa`: added `b{4,8}_d{96,512}` instantiations —
     GPT-NeoX (d=96, group_size=32) and Gemma 4 global attention
     (d=512, 256 threads/TG due to register pressure).
   Each family is additive — the original kernels are kept for callers
   that dispatch the base head_dim.

(`fence` is **not** a next-up item — it is intentionally out of scope; see
[§ Fence ops](#fence-ops--intentionally-out-of-scope).)

## §5 Micro-optimizations across existing perf paths

Researched on branch `ek/t4-microopts` (2026-05-23).  For each of the
five standard micro-optimization patterns the investigation determined
whether the pattern applies, is already present, or requires new
infrastructure.  Full rationale and implementation sketches are in
[`docs/PROPOSED_OPTIMIZATIONS.md`](PROPOSED_OPTIMIZATIONS.md).

### Pattern 1 — `float4` / `half4` vectorized X loads

**Already implemented** — no action required.

The `VectorizePass`
(`crates/metaltile-codegen/src/passes/vectorize.rs`, `MAX_VEC_RUN = 4`)
detects consecutive scalar `Load` ops with contiguous indices and
promotes them to `Op::VectorLoad`.  The MSL emitter maps width-4 loads
to `float4` (F32), `half4` (F16), `bfloat4` (BF16, Metal 3.1+).  All
X-load sites in the GEMV/GEMM kernels are deliberately sequenced for
this pattern (code comments confirm; e.g. `mt_qmv`: "16 X loads —
consecutive in IR for vectorize fusion (4× float4)"; `mt_qmm_mma`:
"8 contiguous device loads → 2× vec4 after vectorize pass").

### Pattern 2 — `simd_broadcast` for scale/bias broadcast across lanes

**Proposal only** — see `PROPOSED_OPTIMIZATIONS.md §2`.

The DSL exposes `simd_broadcast(value, lane)` (used in AURA kernels);
the optimization applies in principle to the int4/int8 GEMV kernels
where 4 (int4) or 16 (int8) consecutive lanes share the same group
scale/bias address per K-block.  Not implemented because:
(a) Apple Silicon L1 cache natively broadcasts same-address loads from
threads in the same simdgroup — the hardware already coalesces the
redundant loads into a single fetch; (b) no profiling evidence that
instruction pressure (not memory bandwidth) is the bottleneck.

### Pattern 3 — `fast::` math intrinsics in audio + numeric paths

**Proposal only** — see `PROPOSED_OPTIMIZATIONS.md §3`.

`mel_spectrogram` (sin/cos/log), `mt_softmax` (exp), `mt_logsumexp`
(exp/log), `vocoder_istft` (sin/cos) all use IEEE-precise Metal
built-ins.  Metal's `fast::exp`, `fast::log`, `fast::sin`, `fast::cos`
would give ~1.5–2× speedup at 1–3 ULP instead of ≤ 0.5 ULP.  Not
implemented because the DSL has no `FastExp` / `FastLog` / `FastSin` /
`FastCos` `UnaryOpKind` variants — adding them requires new IR nodes,
codegen emission, and precision validation against the existing
test tolerances (`1e-3` for mel, `1e-4` for softmax/logsumexp).

### Pattern 4 — f16/bf16 accumulator path for small-K shapes

**Not applicable** — see `PROPOSED_OPTIMIZATIONS.md §4`.

All production GEMV/GEMM kernel shapes accumulate over K ≥ 64 elements
under int4/int8 quantization noise; switching accumulators from f32 to T
would push the accumulated error above the 0.999 cosine similarity gate.
The audio kernels (mel/vocoder) are also f32-accumulation-critical due to
catastrophic cancellation in online-softmax / DFT summation.  fp32
accumulators are correctness-required across all identified targets.

### Pattern 5 — K-loop software pipelining

**Proposal only** — see `PROPOSED_OPTIMIZATIONS.md §5`.

The MMA-tiled K-loop kernels (`mt_qmm_mma`, `mt_qmm_mma_m16`,
`mt_qmm_mma_int8`, `mt_qmm_mma_m16_int8`, `mt_qmm_mma_mpp_int8`,
`mt_qmm_nax_int8`, `mt_moe_gather_qmm_mma_*`) load the next K-block
into threadgroup memory and then compute MMA in strict sequence.
Overlapping the load of K=k with MMA of K=k-1 (two-stage ping-pong)
would hide ≈50% of the memory-latency gap and typically yields 15–25%
throughput improvement on M3 hardware (per Apple's `steel_gemm` docs).
Not implemented because it requires a new `Op::PrefetchAsync` IR op and
a `prefetch.rs` codegen pass that identifies and reorders the
`[ThreadgroupStore* … ThreadgroupBarrier … MMA*]` pattern; estimated
scope is 2–3 days of codegen work.

### Model-enablement kernels (separate track from generic-op completeness)

These don't move the coverage % much but each one unblocks a model family or
removes a measured per-layer CPU sync:

- **Vision (Phase 6.5)** — `conv2d`, `patch_embed`, `rope_2d`: **landed**
  (`ffai/conv2d.rs`, `ffai/patch_embed.rs`, `ffai/rope_2d.rs`). Unblocks the
  VLM vision encoders.
- **STT / TTS (Phase 7)** — `mel_spectrogram`, `audio_conv1d`,
  `vocoder/iSTFT`: **landed** (`ffai/mel_spectrogram.rs`,
  `ffai/audio_conv1d.rs`, `ffai/vocoder.rs`). Unblocks Whisper, Kokoro, and
  Qwen-Omni audio.
  - **Vision / STT / TTS perf follow-ups** ✓ (this PR) — MMA-tiled
    `conv2d_mma`, `conv3d_mma`, `patch_embed_mma` (implicit-im2col +
    4-SG 2×2 simdgroup-matrix MMA; `ffai/conv2d_mma.rs`,
    `ffai/conv3d_mma.rs`, `ffai/patch_embed_mma.rs`) and Bluestein
    non-pow2 FFT (`mt_fft_bluestein_{preprocess,chirp_filter,cmul,postprocess}`
    in `mlx/fft.rs`; covers Whisper n_fft=400/480 in O(N log N)).
- **Host-fallback closers** — all three **landed**: `gated_rmsnorm`
  (Qwen3.5/3.6 GDN post-step, `ffai/gated_rmsnorm.rs`), the
  `sdpa_decode` learned-sink term (GPT-OSS-20B, `has_sink` /
  `sink_logit` on `ffai/sdpa_decode.rs`), and the 2D-`A_log`
  `ssm_step` variant (Jamba, `ssm_step_a2d` in `ffai/ssm.rs`). Each
  was correctness-neutral (the host path worked) but cost a per-layer
  CPU↔GPU sync; folding them on-GPU is a decode-throughput win.

## Open uncertainties / counting caveats

- The four rows added in the 2026-05-21 refresh (`swiglu`,
  `sdpa_decode_batched`, `moe`, `logits processors`) had their metaltile column
  verified against source; their MLX-upstream / MLX-alpha columns are a
  best-effort read (those repos were not checked out) — treat them as
  provisional.
- The `nax`-gated kernels (`quantized_nax`, `fp_quantized_nax`,
  `steel_attention_nax`, `steel_gemm_{fused,gather,splitk}_nax`,
  `quantized_mpp`, `moe_mpp{,_bm{8,16,64}}{,_int8}`) are all
  expressed in the `#[kernel]` DSL via the `coop_tile_*` intrinsics
  (PR #149 + #152 ported every prior `Op::InlineMsl` MPP body to
  the DSL; their MPP `matmul2d` calls now come from
  `Op::CoopTileSetup` / `Run` / `LoadA` / `LoadB` / `StoreC`
  lowering). Each has a paired `*_nax_gpu_correctness` test
  (counted ✓). They are `#[cfg(feature = "nax")]`-gated in
  `mlx/mod.rs` and most do register `inventory::submit!`
  BenchSpecs; the `nax` feature is off in default / non-macOS CI
  builds, so the registry-consistency check never sees them.
- `mlx/strided.rs` (`mt_strided_copy`) covers strided copy but the stride
  dimensionalities were not audited — marked `~` defensively. Upstream
  `copy.metal` has multiple `copy_g_nd*` shapes.
- `ffai/sdpa_decode.rs` and `ffai/sdpa_decode_batched.rs` are FFAI-specific
  (`✗ / ✗ / ✓`) — not ports of upstream MLX kernels; they are derivatives of
  `mt_sdpa_vector` with a decoupled `kv_stride` and a batched-Q walk.
- `ffai/aura_flash_p1.rs` is now ✓ — non-causal `(kb=4, vb=2, dim={64,128})` +
  `(kb=4, vb=4, dim={64,128})` (4 instantiations) plus the causal variant
  `aura_flash_p1_causal_kb4_vb2_{d64,d128}` are all registered; generic over `T`
  (f32/f16/bf16) per PR #152.
- Coverage % treats the alpha-only kernels as in-scope (we maintain the fork,
  so they count toward the union).
- The Gemma / Nemotron-H / GPT-OSS-20B kernel work is now consolidated onto
  `ek/aura-port` and folded into this audit (the `sdpa_decode_d512` and
  `rms_norm_wide` rows). The three host-side fallbacks surfaced by the model
  review (`gated_rmsnorm`, the `sdpa_decode` learned-sink term, the 2D-`A_log`
  `ssm_step_a2d` variant) are now all landed as ✓ rows — they were
  correctness-neutral (the host path worked) but cost a CPU sync per layer
  on the affected models.
- The Vision / STT / TTS rows (`conv2d`, `patch_embed`, `rope_2d`,
  `mel_spectrogram`, `audio_conv1d`, `vocoder/iSTFT`) are scoped from the
  Phase 6.5 / 7 plan, not yet from checked-out reference source — treat their
  MLX columns as provisional.

## Fence ops — intentionally out of scope

MLX's `fence.metal` (`mlx/backend/metal/kernels/fence.metal`, ~52 lines) is
**not a compute kernel** — it is a GPU-side synchronisation primitive. It is
deliberately *not* ported to metaltile, and the `fence` audit row is marked
`—` rather than `✗`. This section records why.

### What the fence ops are

Three kernels: `input_coherent` (force input-buffer visibility),
`fence_update` (bump a counter in a shared buffer), and `fence_wait` — a
compute kernel that **spin-loops** reading that counter until it changes.
Together they let the GPU order work *across command buffers / streams*
without a CPU round-trip.

### How MLX actually uses them

`mlx/backend/metal/fence.cpp`'s `FenceImpl` has **two paths**:

- **Default:** `device->newSharedEvent()` — a standard **`MTLSharedEvent`**.
  The wait executes in the GPU *command processor*, not a shader core.
- **`use_fast` path** (the `fence.metal` spin-wait kernels): gated behind
  `GPUFamilyMetal3` **and** macOS 15 **and** an opt-in env var
  (`metal_fast_synch`). **Off by default.**

So MLX itself treats the GPU spin-wait fence as an *opt-in latency
micro-optimization* for its multi-stream `async_eval` workloads — not a
primitive every pipeline needs. Its default is `MTLSharedEvent`.

### Why FFAI does not need it

- FFAI's current pipeline is single-stream autoregressive decode. Within a
  forward pass, Metal's automatic hazard tracking orders kernels in a
  command buffer for free; across command buffers on one queue, submission
  order suffices. No GPU-side fence is involved.
- CPU/GPU pipelining (build command buffer N+1 while the GPU runs N) is
  `commit` + completion handlers — not a fence.
- For genuine cross-queue / cross-stream GPU sync, `MTLEvent` /
  `MTLSharedEvent` (encoder-level — `encodeWaitForEvent` /
  `encodeSignalEvent`) are the correct, power-efficient primitive, and they
  belong in `metaltile-runtime`'s dispatch layer, **not** as a `#[kernel]`.
- A `fence_wait` spin-wait is a deliberate near-infinite GPU loop: it burns
  a shader core + power, and a counter that never updates (a bug, a wrong
  dispatch) is a permanent GPU pin → hard reboot — the exact
  machine-freeze hazard documented in `developing.md`.

### When this could change

If FFAI later runs **multiple concurrent GPU streams** — e.g. speculative
decoding (draft/target overlap), prefill/decode overlap, or ANE+GPU
concurrency (Phase 8 / 9) — it will need cross-stream ordering. The right
implementation is `MTLEvent`-based encoder-level sync added to
`metaltile-runtime` (MLX's own default), **not** a spin-wait `#[kernel]`.
Only if profiling later shows that `MTLEvent`'s command-processor latency is
a measured bottleneck for an ultra-fine-grained sync pattern would the
opt-in spin-wait become worth revisiting — and even then it is a runtime
concern, not a metaltile kernel.
