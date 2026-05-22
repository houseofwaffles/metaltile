# metaltile kernel-op coverage audit

Generated: 2026-05-18 · Refreshed: 2026-05-21
Sources surveyed:
- MLX upstream `ml-explore/mlx@main` (commit `2414e5df`)
- MLX fork `ekryski/mlx@alpha` (commit `4919270e`)
- metaltile `0xClandestine/metaltile:dev` (commit `dd4c2ef`)

## Summary

- Total kernel-op rows in this audit (union): **78**
- metaltile-ported kernel ops: **50 / 78 = 64 %** — 40 full ✓ (51 %), 10 partial ~ (13 %)
- **Still to cover: 28 ops not ported (✗)**, plus **10 partial ports** still to finish
- 3 in-flight kernel families have an **open PR** (not yet landed) — see
  [Kernels with open PRs](#kernels-with-open-prs).

> **Note on the 2026-05-21 refresh.** The metaltile column was re-surveyed
> against source at `dev` HEAD `dd4c2ef`. Since the previous refresh, PRs
> #94–#99, #110, #128, #129, #134, #135 landed; the `gated_delta` row was
> stale (the kernel had landed but the table still showed ✗), and four
> kernel families that ship on `dev` had no row at all (`swiglu`,
> `sdpa_decode_batched`, `moe`, the logits-processor stack). Those are added
> below. The MLX-upstream and MLX-alpha columns were **not** re-verified
> against those repos (not checked out) — only the metaltile column was
> re-surveyed. More rows are expected: the Gemma / Nemotron-H / GPT-OSS-20B
> kernel work lives in separate worktrees and will be folded in once it is
> consolidated onto a branch and PR'd upstream.

## Op coverage table

| Op | MLX (upstream) | MLX (ekryski@alpha) | metaltile | Notes |
|---|---|---|---|---|
| arange | ✓ | ✓ | ✓ | `mlx/arange.rs` → `mt_arange`. Generic `T`. Direct port. |
| arg_reduce (argmax/argmin → float) | ✓ | ✓ | ~ | `mlx/arg_reduce.rs` → `mt_argmax_f32` only. f32 argmax only; argmin and bf16/f16 not yet. |
| arg_reduce (argmax → u32 index) | ✗ | ✗ | ✓ | `ffai/arg_reduce.rs` → `ffai_argmax<T>`. FFAI-only; integer-index sampler workhorse. |
| binary (elementwise add/sub/mul/div/min/max) | ✓ | ✓ | ✓ | `mlx/binary.rs` → 6 kernels. Generic `T`. Direct port. |
| binary_two (fused two-output elementwise) | ✓ | ✓ | ✓ | `mlx/binary_two.rs` → `mt_binary_two<T>`. |
| copy (contiguous) | ✓ | ✓ | ✓ | `mlx/copy.rs` → `mt_copy<T>`. |
| copy (strided / general) | ✓ | ✓ | ~ | `mlx/strided.rs` → `mt_strided_copy`. Limited stride dimensionality. |
| ternary (select) | ✓ | ✓ | ✓ | `mlx/ternary.rs` → `mt_select<T>`. |
| unary (exp/log/sqrt/rsqrt/abs/silu/etc.) | ✓ | ✓ | ✓ | `mlx/unary.rs` → 7+ kernels including `mt_silu`. |
| swiglu (`silu(gate)·up` fused MLP activation) | ✗ | ✗ | ✓ | `mlx/swiglu.rs` → `mt_swiglu<T>`. Fused element-wise `silu(gate) * up` — the standard modern-transformer MLP activation (Llama 4, Qwen3 dense + MoE, Gemma, Mistral). metaltile fuses what MLX expresses as separate `silu` + `mul` ops; no dedicated MLX kernel. The broader `fused_gate_activation` (gelu / clipped-swiglu variants) is still a separate ✗ row below. |
| random (key hash → u32) | ✓ | ✓ | ✓ | `mlx/random.rs` → `mt_random_hash`. |
| reduce (sum/prod/max/min — all + row + col) | ✓ | ✓ | ~ | `mlx/reduce.rs` covers `all_reduce*` and `row_reduce`. Column-reduce partial; segmented-reduce missing. |
| sort | ✓ | ✓ | ~ | `mlx/sort.rs` → `mt_sort<T>`. Single-block path only; multi-block / segmented not yet. |
| scan (prefix sum) | ✓ | ✓ | ~ | `mlx/scan.rs` → `mt_scan<T>`. Inclusive sum only; exclusive / multi-op not yet. |
| softmax | ✓ | ✓ | ✓ | `mlx/softmax.rs` → `mt_softmax<T>` (looped + single-row collapsed). |
| logsumexp | ✓ | ✓ | ✓ | `mlx/logsumexp.rs` → `mt_logsumexp<T>`. |
| layer_norm | ✓ | ✓ | ✓ | `mlx/layer_norm.rs` → `mt_layer_norm<T>`. |
| rms_norm | ✓ | ✓ | ✓ | `mlx/rms_norm.rs` → `mt_rms_norm<T>` plus `mt_rms_norm_small<T>` (2-elem/thread small-head_dim variant for the per-head q_norm/k_norm dispatch). |
| rope (standard) | ✓ | ✓ | ✓ | `mlx/rope.rs` → `mt_rope` (fp16 only). |
| rope (Llama-3 banded) | ✗ | ✗ | ✓ | `ffai/rope_llama.rs` → `ffai_rope_llama<T>`. Decode-form, generic dtype, optional Llama-3 frequency-band scaling. No MLX counterpart. |
| sdpa_vector (prefill / generic) | ✓ | ✓ | ✓ | `mlx/scaled_dot_product_attention.rs` → `mt_sdpa<T>`. Scalar SDPA — sufficient for short sequences. |
| sdpa_vector (GQA decode, single pass) | ✓ | ✓ | ✓ | `mlx/sdpa_vector.rs` → `mt_sdpa_vector<T>`. head_dim=128 only; covers f32/f16/bf16. |
| sdpa_vector_2pass | ✓ | ✓ | ✓ | `ffai/sdpa_decode_2pass.rs`. head_dim=128 only. Upstream supports {64,96,128,256}. |
| sdpa_decode (FFAI production decode, decoupled `kv_stride`) | ✗ | ✗ | ✓ | `ffai/sdpa_decode.rs` → `ffai_sdpa_decode<T>`, plus `ffai/sdpa_decode_d64.rs` / `sdpa_decode_d256.rs` for head_dim {64, 256}. FFAI-only variant with `kv_stride` ≠ `n_kv` (pre-allocated max-seq cache); now covers head_dim ∈ {64, 128, 256} and a sliding-window + sink-token path (`sink_end` / `window_start` constexprs). |
| sdpa_decode_batched (speculative-decode batched-Q decode) | ✗ | ✗ | ✓ | `ffai/sdpa_decode_batched.rs` → `sdpa_decode_batched_q{2,4}<T>` (+ `sdpa_decode_batched_prefill.rs`). K query positions share one KV walk per dispatch (M7 speculative decoding), amortizing KV memory bandwidth K× vs. K independent single-Q `sdpa_decode` dispatches. FFAI-only. |
| steel_attention (Flash, prefill) | ✓ | ✓ | ✓ | `mlx/steel/attn/steel_attention.rs` → `mt_sdpa_prefill<T>`. Scalar-flash prefill (BQ=4, online softmax, causal), generic `T`, head_dim=128. The old "`Op::FlashAttention` lowers to an error placeholder" blocker is resolved. |
| steel_attention_mma (Flash prefill, simdgroup-MMA) | ✓ | ✓ | ✓ | `mlx/steel/attn/steel_attention_mma.rs` → `mt_sdpa_prefill_mma<T>`. Real simdgroup-matrix MMA path; generic `T`, validated f32/f16/bf16, head_dim=128. A pre-M3 bf16-tuned sibling `mt_sdpa_prefill_mma_bf16` (`steel_attention_mma_bf16.rs`) is selected by `sdpa_prefill_mma_for()` — a perf specialization, not a separate op. |
| steel_attention_nax | ✓ | ✓ | ✗ | Header-only stub + `nax` feature gate. |
| steel_gemm_fused | ✓ | ✓ | ~ | `mlx/steel/gemm/steel_gemm_fused.rs` → `mt_steel_gemm_64x64x16_2x2<T>`. One block-shape variant; upstream has many. |
| steel_gemm_fused_nax | ✓ | ✓ | ✗ | Blocker: `nax` feature gate. (Simdgroup-matrix primitive now exists — see `steel_attention_mma`.) |
| steel_gemm_gather | ✓ | ✓ | ✗ | Blocker: indirect (gather) indexing of the matmul operands. |
| steel_gemm_gather_nax | ✓ | ✓ | ✗ | Same + NAX feature gate. |
| steel_gemm_masked | ✓ | ✓ | ✗ | Blocker: block-level predication. |
| steel_gemm_segmented | ✓ | ✓ | ✗ | Blocker: ragged batched matmul. |
| steel_gemm_splitk + accum | ✓ | ✓ | ✗ | Blocker: two-kernel split-K dispatch + accumulator pass. |
| steel_gemm_splitk_nax | ✓ | ✓ | ✗ | Same + NAX feature gate. |
| steel_conv 2D (implicit-GEMM) | ✓ | ✓ | ✗ | Blocker: im2col primitives missing. |
| steel_conv 3D | ✓ | ✓ | ✗ | Same blocker + 3D `MLXConvParams<3>` indexing. |
| steel_conv_general (strides/dilation/groups) | ✓ | ✓ | ✗ | Same blockers as steel_conv. |
| conv (winograd + naive_unfold + depthwise) | ✓ | ✓ | ✗ | `crates/metaltile-std/src/mlx/conv.rs` is a stub left from the old bench crate, not declared in `mod.rs`. No DSL port. |
| gemv | ✓ | ✓ | ✓ | `mlx/gemv.rs` → `mt_gemv<T>`. |
| gemv_masked | ✓ | ✓ | ✓ | `mlx/gemv_masked.rs` → `mt_gemv_masked<T>` (no MLX comparison wired). |
| quantized (affine_quantize / affine_dequantize) | ✓ | ✓ | ~ | `mlx/quantized.rs` → quantize **and** dequantize for int4/int8, plus dequantize for int3/int5/int6 (`mt_affine_{quantize,dequantize}_int{3,4,5,6,8}`). Gap: int2, and the quantize side of int3/5/6. |
| quantized (affine_qmv / qvm / qmm — matvec / matmul) | ✓ | ✓ | ~ | `mlx/quantized.rs` → `mt_qmv` + `mt_qmm` / `mt_qmm_bm2` / `mt_qmm_bm4` (3 M-batch tiles) with an `mt_qmm_for` selector, all f32+f16, int4. `mt_qmm_mma` covers the simdgroup-matrix MMA path; `mt_qmm_mma_mpp` (#137, merged) wires the Apple `mpp::tensor_ops::matmul2d` NAX path for MLX-parity. Dynamic-M batched-prefill driver `mlx::quantized_mma_dynamic_m` (host-side `T → ceil(T/32)·32` pad + `mt_qmm_mma` dispatch) is open in **PR [#144](https://github.com/0xClandestine/metaltile/pull/144)** — collapses T per-token int4 dispatches into one batched call for the 70× T=32K Qwen3.6-A3B prefill cell. Gap: `qvm` absent, bit-widths other than int4 absent, bf16 on `mt_qmm_mma_mpp` absent. |
| quantized (gather_qmv / gather_qmm — gather variants) | ✓ | ✓ | ~ | Bare-tensor `ffai/gather.rs` exists but is non-quantized. The MoE grouped-gather quantized matmul stack lives across `ffai/moe.rs` (`mt_moe_gather_qmm_mma_int4_bm{1,16}` scalar + MMA), `ffai/moe_mpp.rs` (`bm16_mpp` MPP/NAX), `ffai/moe_mpp_bm64.rs` (`bm64_mpp` 4-SG WM=WN=2 scale-up for prefill), and `ffai/moe_mpp_bm8.rs` (`bm8_mpp` half-height BM=8 for topK=8 decode where m_total=8 — uses destination-only-cooperative MPP descriptor `(M=8, N=32, K=16)`). bm16/bm64_mpp + bm8_mpp open in **PR [#144](https://github.com/0xClandestine/metaltile/pull/144)**; bm1/bm16 originated in [#125](https://github.com/0xClandestine/metaltile/pull/125) / [#136](https://github.com/0xClandestine/metaltile/pull/136). Affine `qvm` flavour absent. |
| moe (router top-k + permute + unpermute orchestration) | ✗ | ✓ | ✓ | `ffai/moe.rs` → `mt_moe_router_topk<T>`, `mt_moe_permute<T>`, `mt_moe_unpermute<T>`. MoE expert-routing orchestration for Qwen3.6-35B-A3B / Qwen3-Coder-30B-A3B end-to-end serving. The grouped quantized BGEMM that fuses the per-expert FFN matmuls into one dispatch is **open in PR [#125](https://github.com/0xClandestine/metaltile/pull/125) / [#136](https://github.com/0xClandestine/metaltile/pull/136)**. |
| dequant_gather (quantized embedding-table gather) | ✗ | ✗ | ✓ | `ffai/dequant_gather.rs`. int{3,4,5,6,8} all bit-widths. FFAI-specific, no MLX counterpart. |
| dequant_gemv (quantized GEMV, FFAI flavour) | ~ (subset of `quantized.metal`) | ~ | ✓ | `ffai/dequant_gemv.rs`. int{3,4,5,6,8}, generic `T`. Coexists with the partial `mt_qmv_f32` port; FFAI-tuned shape. |
| fp_quantized (fp4/fp8 quant + dequant) | ✓ | ✓ | ~ | `mlx/fp_quantized.rs` → `mt_fp4_quant_dequant` (f32 only). fp8 path and other dtypes missing. |
| fp_quantized_nax | ✓ | ✓ | ✗ | Module file present but empty (no `#[kernel]` defs). NAX-gated. |
| quantized_nax | ✓ | ✓ | ✗ | Module file present but empty (no `#[kernel]` defs). NAX-gated. |
| fft (radix + readwrite) | ✓ | ✓ | ✗ | Stub file in repo, not declared. No DSL port. |
| hadamard (hadamard_n + hadamard_m) | ✓ | ✓ | ~ | `mlx/hadamard.rs` → `mt_hadamard_n{64,128,256,512,1024}<T>`. Power-of-2 FWHT via log2(N) butterfly passes. The non-power-of-2 `hadamard_m` factor (M ∈ {12,20,28}) is a follow-up. |
| fence | ✓ | ✓ | ✗ | Stub file in repo, not declared. Synchronization primitive. |
| gather (bare-tensor embedding lookup) | ✓ (via indexing/) | ✓ | ✓ | `ffai/gather.rs` → `ffai_gather<T>`. FFAI's embedding-table gather. |
| indexing (scatter, scatter_axis, gather_axis, gather_front, masked_scatter) | ✓ | ✓ | ~ | `mlx/gather_axis.rs` + `mlx/scatter_axis.rs` → `mt_gather_axis` / `mt_scatter_axis`. Contiguous gather/scatter-along-axis. The general strided forms (scatter, gather_front, masked_scatter) need strided-indexing infra — follow-up. |
| aura_encode (codebook quantize, fused) | ✗ | ✓ (`turbo_fused_encode` in `turbo_quant.metal`) | ✓ | `ffai/aura_encode.rs`. Bit-widths 2/3/4/8. Renamed turbo_*→aura_*. |
| aura_dequant_rotated (bulk dequant to rotated codec space) | ✗ | ✓ (`turbo_dequant_rotated` in `turbo_quant.metal`) | ✓ | `ffai/aura_dequant_rotated.rs`. bits ∈ {2,3,4,8}. Renamed. |
| aura_score (compressed-domain Q·K) | ✗ | ✓ (`turbo_score`) | ✓ | `ffai/aura_score.rs`. bits ∈ {2,3,4,8}. Renamed. |
| aura_value (compressed-domain value aggregation) | ✗ | ✓ (`turbo_value` in `turbo_quant.metal`) | ✓ | `ffai/aura_value.rs`. Sparsity-threshold guard mirrors MLX upstream. Renamed. |
| aura_flash_p1 (compressed-domain flash pass 1) | ✗ | ✓ (`turbo_flash_p1` in `turbo_flash.metal`) | ~ | `ffai/aura_flash_p1.rs`. Only the `(kb=4, vb=2, dim=128)` aura4v2/Qwen3-128 instantiation today; causal-variant from upstream not ported. |
| aura_flash_pass2 (cross-block online-softmax merge) | ✗ | ✓ (`turbo_flash_pass2`) | ✓ | `ffai/aura_flash_pass2.rs`. fp32 accums → bf16 final. Renamed. |
| turbo_flash_sdpa (fused single-pass SDPA, sinks variant) | ✗ | ✓ (`turbo_flash_sdpa.metal`) | ✓ | `ffai/aura_flash_sdpa.rs` → `aura_flash_sdpa_kb*_vb*_d*<T>`. Single-pass online-softmax over compressed K/V with attention sinks + sliding-window causal mask. Single-simdgroup shape (token-parallelism a perf follow-up). |
| flash_quantized_sdpa (single-pass quantized SDPA, affine cache) | ✗ | ✓ (`flash_quantized_sdpa.metal`) | ✓ | `ffai/flash_quantized_sdpa.rs` → `flash_quantized_sdpa_b{4,8}_d{64,128,256}<T>`. Single-pass online-softmax SDPA over affine-quant KV, with sinks + sliding-window. head_dim {96,512} and bool/float masks are a follow-up. |
| gated_delta (GatedDeltaNet recurrence) | ✗ | ✓ (`gated_delta.metal`) | ✓ | `ffai/gated_delta.rs` → `mt_gated_delta_step<T>` (single-token decode) + `mt_gated_delta_chunk<T>` (chunked-prefill). GDN linear-attention for the Qwen3.5 / 3.6 / 3.6-MoE hybrid models (≈75 % of layers). The MMA-tiled chunked-WY prefill perf variant `mt_gated_delta_wy_chunk` landed in [#115](https://github.com/0xClandestine/metaltile/pull/115). The fused prep+recurrence variant `mt_gated_delta_prep_step` (`ffai/gated_delta_prep.rs`) is open in **PR [#144](https://github.com/0xClandestine/metaltile/pull/144)** — one dispatch absorbs conv-split + per-head q/k RMSNorm + g/beta + the recurrence, collapsing 3 host commit+wait pairs per GDN layer down to 1 (Qwen3.6-A3B decode unlock). |
| gated_delta_replay (tape capture + state replay) | ✗ | ✓ (`gated_delta_replay.metal`) | ✓ | `ffai/gated_delta_replay.rs` → `gated_delta_step_record<T>` (forward + delta-tape) + `state_replay<T>` (branchless accepted-prefix re-fold). Speculative-decode rollback on GDN. |
| ssm_step (Mamba 2 SSD single-token decode) | ✗ | ✓ (`ssm.metal`) | ✓ | `ffai/ssm.rs` → `ssm_step<T>`, `mt_ssm_step<T>`. Faithful port; `mlx_src: None` because pinned MLX upstream doesn't ship `ssm.metal`. Will graduate to `mlx/` when pin moves. |
| conv1d_causal_step (depthwise SSM conv stream) | ✗ | partial (subset of SSM toolchain) | ✓ | `ffai/ssm.rs` → `conv1d_causal_step<T>`. fp32 state recurrence. |
| ssm_replay (sequential tape capture + replay) | ✗ | ✓ (`ssm_replay.metal`) | ✓ | `ffai/ssm_replay.rs` → `ssm_step_record<T>` (SSD forward + dA/dBx tape) + `ssm_replay<T>` (re-fold first k entries). Spec 040 Mamba/Mamba2 state replay. |
| fused_gate_activation (silu/gelu × up gate) | ✗ | ✓ (`fused_gate_activation.metal`) | ✗ | NOT PORTED. The `silu` variant is covered by `mlx/swiglu.rs` (see the `swiglu` row); the gelu-approx and clipped-swiglu variants, plus the single-row / looped dispatch forms, are not. |
| rms_norm_residual (RMSNorm + residual add fused) | ✗ | ✓ (`rms_norm_residual.metal`) | ✓ | `ffai/rms_norm_residual.rs` → `ffai_rms_norm_residual<T>`. Reduction-mode, `N = TPG*4`; mirrors `mt_rms_norm` + a residual-add input. ~90 saved dispatches/token on Gemma4-30 type configs. |
| rms_norm_rope (RMSNorm + RoPE fused) | ✗ | ✓ (`rms_norm_rope.metal`) | ✓ | `ffai/rms_norm_rope.rs` → `ffai_rms_norm_rope<T>`. Reduction-mode, paired-layout RoPE; `TPG = axis_size/2`. Q/K post-projection norm+rope in one dispatch. |
| rms_norm_qgemv (RMSNorm + 4-bit quantized GEMV fused) | ✗ | ✓ (`rms_norm_qgemv.metal`) | ✓ | `ffai/rms_norm_qgemv.rs` → `ffai_rms_norm_qgemv<T>`. Reduction-mode, int4, one row/threadgroup; eliminates the global RT of the normalized activation. MLX's 8-row-per-TG tiling is a perf follow-up. |
| batched_qkv_qgemv (Q/K/V 4-bit qGEMV → 1 dispatch) | ✗ | ✓ (`batched_qkv_qgemv.metal`) | ✓ | `ffai/batched_qkv_qgemv.rs` → `ffai_batched_qkv_qgemv<T>`. Reduction-mode, int4; `program_id::<2>()` selects Q/K/V, output concatenated `[Q\|K\|V]`. Decode-form fused QKV projection. |
| kv_cache_update (raw bf16/fp16 single-token append) | ✗ | ✗ | ✓ | `ffai/kv_cache.rs` → `kv_cache_update<T>`. FFAI-only; raw cache append. |
| kv_cache (affine-quant int4/int8 quantize + bulk dequant) | ~ (via `quantized.metal` affine_quantize) | ~ | ✓ | `ffai/kv_cache.rs` — `quantize_kv` + `bulk_dequant_kv` for int4/int8. FFAI-specific cache layout. |
| sampling (softmax + categorical inverse-CDF) | ✗ | ✗ | ✓ | `ffai/sampling.rs` → `softmax_categorical_sample`. Companion to `ffai_argmax` for `T > 0` decode. |
| logits processors (temperature, repetition penalty, top-k / top-p / min-p masks) | ✗ | ✗ | ✓ | `ffai/logits_{processors,topk,top_p,min_p}.rs` → `logits_temperature`, `logits_repetition_penalty`, `logits_topk_mask`, `logits_top_p_mask`, `logits_min_p_mask` (all generic `T`). In-place decode-form sampler stages composed before `softmax_categorical_sample`. FFAI-only. |

## Kernels with open PRs

These are tracked above with an inline link in the Notes column; collected here
for quick scanning. Status reflects the open PRs as of 2026-05-21.

| PR | Kernel(s) | Affects row | State |
|---|---|---|---|
| [#115](https://github.com/0xClandestine/metaltile/pull/115) | `mt_gated_delta_wy_chunk` — chunked-WY GDN prefill (scalar foundation) | `gated_delta` | Draft / WIP; CI green, needs rebase onto current `dev`. |
| [#125](https://github.com/0xClandestine/metaltile/pull/125) | `mt_moe_gather_qmm_int4` — grouped MoE quantized matmul (m1 scalar) | `quantized (gather_*)`, `moe` | Draft; fmt/clippy/commit-hygiene red. Overlaps #136. |
| [#136](https://github.com/0xClandestine/metaltile/pull/136) | MoE gather BGEMM stack (m8 / MMA / MPP-NAX bm16 + bm64) | `quantized (gather_*)`, `moe` | Draft / WIP — surfaced for visibility; currently regresses vs MLX. |
| [#137](https://github.com/0xClandestine/metaltile/pull/137) | `mt_qmm_mma_mpp` + `mt_mpp_matmul_smoke` — int4 qmm via Apple `mpp::tensor_ops::matmul2d` | `quantized (qmm)` | Draft; MLX-parity, needs rebase + CI. |

## Notes on counting decisions

A few rows mix multiple `.metal` files into one op or split one file into multiple ops:

- **`sdpa_vector*` rows.** Upstream `sdpa_vector.h` defines `sdpa_vector`, `sdpa_vector_2pass_1`, `sdpa_vector_2pass_2`. Counted as two ops: `sdpa_vector` (single pass) + `sdpa_vector_2pass` (two-pass pair).
- **AURA stack.** Each codec stage (`encode`, `dequant_rotated`, `score`, `value`, `flash_p1`, `flash_pass2`) is a separate row — they're separately compiled kernels with their own dispatch shapes. The `turbo_flash_sdpa` (sinks-fused single-pass) is also its own row.
- **`steel/` family.** Each kernel file in `steel/{attn,conv,gemm}/kernels/` becomes one op row; per-block-shape instantiations are not counted separately. `steel_attention` (scalar-flash) and `steel_attention_mma` (simdgroup-MMA) are counted as two rows because they are separately compiled kernels with different lowering strategies; the bf16-tuned `mt_sdpa_prefill_mma_bf16` is folded into the MMA row as a perf specialization.
- **`quantized.metal`.** Split into three rows by semantic operation (quant/dequant, qmv/qvm/qmm matmul, gather-qmv/qmm) rather than by template instantiation. Quantized-NAX and FP-quantized-NAX are separate rows because the metaltile modules exist (empty) and have separate feature gates.
- **`indexing/`** is one row covering scatter / scatter_axis / gather_axis / gather_front / masked_scatter. Bare `gather` is its own row because metaltile has a dedicated FFAI port.
- **`moe`** is one row for the routing/permute/unpermute orchestration kernels in `ffai/moe.rs`. The grouped quantized BGEMM that the open PRs add is counted under the `quantized (gather_*)` row.
- **`logits processors`** is one row for the FFAI sampler-stage kernels (`temperature`, `repetition_penalty`, `topk` / `top_p` / `min_p` masks). FFAI-only, no MLX counterpart.
- **Cells marked `~`** indicate metaltile has a partial port — typically one bit-width, one dtype, or one block shape where upstream has many. Read the notes column for the specific gap.

## Highest-value un-ported ops (next-up recommendations)

Roughly ordered by FFAI-impact × tractability. The fused-norm/-act family is
largely landed now (`rms_norm_residual` / `_rope` / `_qgemv`,
`batched_qkv_qgemv`, `aura_flash_sdpa`, `flash_quantized_sdpa`, `gated_delta`,
`ssm_replay` all ✓). The DSL has a working simdgroup-matrix MMA path
(`steel_attention_mma`, the `probe/mma_layout_probe.rs` layout probe), so the
remaining `steel_gemm_*` / `steel_conv*` rows are no longer blocked on the
primitive itself — only on the gather / masked / split-K / im2col logic layered
on top.

1. **`fused_gate_activation`** — gelu-approx and clipped-swiglu variants + the
   single-row / looped dispatch forms. The `silu` case already ships as
   `mlx/swiglu.rs`; finishing the row is a small elementwise port.
2. **`quantized` gather_qmm / gather_qmv** — the affine grouped-gather matmul.
   In flight in PRs #125 / #136; landing it closes the MoE FFN dispatch-count
   win (one kernel for the whole expert projection).
3. **`steel_gemm_fused` shape coverage** — only `64×64×16` is wired today;
   prefill perf needs more block shapes.
4. **`steel_gemm_splitk` + accum** — two-kernel split-K dispatch + accumulator
   pass. Infra-gated (split-K scheduling primitive).
5. **`steel_gemm_masked`** — block-level predication. Infra-gated.
6. **`steel_conv` 2D / 3D / general** + `conv` — all blocked on im2col / unfold
   primitives. One infra PR unblocks the family.
7. **`indexing` (scatter, gather_front, masked_scatter)** — the strided forms,
   needed for any cache update path that isn't a simple append (sliding-window
   evict, prefix-cache splice, batched scatter).
8. **NAX feature family** — `steel_attention_nax`, `steel_gemm_*_nax`,
   `quantized_nax`, `fp_quantized_nax`. PR #137 demonstrates the Apple
   `mpp::tensor_ops::matmul2d` path; the `nax`-gated rows can follow once the
   feature scaffolding lands.
9. **`fft`** — radix + readwrite. Needs an FFT codegen path (complex types,
   bit-reversal indexing). Lowest FFAI priority.
10. **`fence`** — synchronization primitive. Needs atomics / device-memory
    fence primitives in the DSL; infrastructure, not a compute op.

## Open uncertainties / counting caveats

- The four rows added in the 2026-05-21 refresh (`swiglu`,
  `sdpa_decode_batched`, `moe`, `logits processors`) had their metaltile column
  verified against source; their MLX-upstream / MLX-alpha columns are a
  best-effort read (those repos were not checked out) — treat them as
  provisional.
- `quantized_nax.rs` and `fp_quantized_nax.rs` were re-checked: both are still
  empty (TODO comment only, zero `#[kernel]`) and both are
  `#[cfg(feature = "nax")]`-gated in `mlx/mod.rs`. Counted as `✗` for metaltile.
- `mlx/strided.rs` (`mt_strided_copy`) covers strided copy but the stride
  dimensionalities were not audited — marked `~` defensively. Upstream
  `copy.metal` has multiple `copy_g_nd*` shapes.
- `ffai/sdpa_decode.rs` and `ffai/sdpa_decode_batched.rs` are FFAI-specific
  (`✗ / ✗ / ✓`) — not ports of upstream MLX kernels; they are derivatives of
  `mt_sdpa_vector` with a decoupled `kv_stride` and a batched-Q walk.
- `ffai/aura_flash_p1.rs` is marked `~` because only the `(kb=4, vb=2, dim=128)`
  instantiation is registered; the causal variant from `turbo_flash.metal` and
  other `(kb, vb, dim)` combos aren't ported yet.
- Coverage % treats the alpha-only kernels as in-scope (we maintain the fork,
  so they count toward the union).
- More rows are pending: the Gemma / Nemotron-H / GPT-OSS-20B kernel work is
  spread across separate worktrees and will be folded into this audit once it
  is consolidated onto a branch and PR'd upstream.
