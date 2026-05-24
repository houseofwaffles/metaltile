//! Gated DeltaNet — **fused** prep + chunked-prefill kernel.
//!
//! `mt_gated_delta_prep_chunk` extends
//! [`mt_gated_delta_prep_step`](super::gated_delta_prep::mt_gated_delta_prep_step)
//! over a chunk of `T` tokens, mirroring the relationship between
//! [`mt_gated_delta_step`](super::gated_delta::mt_gated_delta_step) and
//! [`mt_gated_delta_chunk`](super::gated_delta::mt_gated_delta_chunk).
//!
//! State stays register-resident across the entire `T`-loop — one
//! load_state at entry and one store_state at exit, regardless of `T`.
//! This collapses the dominant `mt_gated_delta_prep_step`-per-token T-loop
//! in `Qwen35GDNMixer.forwardMany` to a single dispatch per layer.
//!
//! Inputs (note the added `T` dimension on conv_out / a_raw / b_raw):
//!   - `conv_out`     : Tensor<T> [B, T, 2·Hk·Dk + Hv·Dv]   q | k | v slabs
//!   - `a_log`        : Tensor<T> [Hv]                      per-Hv learnable
//!   - `dt_bias`      : Tensor<T> [Hv]
//!   - `a_raw`        : Tensor<T> [B, T, Hv]
//!   - `b_raw`        : Tensor<T> [B, T, Hv]
//!   - `q_norm_weight`: Tensor<T> [Hk·Dk]   (pass 1.0×invKeyScale² for unweighted q-scale)
//!   - `k_norm_weight`: Tensor<T> [Hk·Dk]   (pass 1.0×invKeyScale for unweighted k-scale)
//!   - `state_in`     : Tensor<T> [B, Hv, Dv, Dk]           (one state per (b, hv))
//!   - `t_len`        : Tensor<u32> [1]                     runtime chunk length
//!
//! Outputs:
//!   - `state_out`    : Tensor<T> [B, Hv, Dv, Dk]
//!   - `y`            : Tensor<T> [B, T, Hv, Dv]
//!
//! ## DISPATCH INVARIANTS (identical to `mt_gated_delta_prep_step`)
//!
//! - **Mode: Reduction.** Each TG is one simdgroup (32 threads).
//! - **Grid: `[Dv, B·Hv, 1]`, TG: `[32, 1, 1]`.**
//! - **`Dk % 32 == 0`.** Each lane owns `n_per_t = Dk / 32` slots.
//! - **Hv divisible by Hk.** GQA: `hk_idx = hv_idx / (Hv/Hk)`.
//! - **`t_len` is runtime u32** so a single PSO compiles for every chunk size.
//!
//! ## Per-iter cost vs prep_step
//!
//! Prep-step pays:
//!   - 1× state-load + 1× state-store (Dk floats per lane)
//!   - prep math + recurrence math
//!
//! Prep-chunk pays:
//!   - 1× state-load + 1× state-store (Dk floats per lane), TOTAL — not per-t
//!   - T × (prep math + recurrence math)
//!
//! State traffic per layer drops by `T`× at the dispatch boundary. For
//! Qwen3.6-A3B (Dk=256, Dv=128, Hv=16, B=1): state size = 16·128·256·4 B =
//! 2 MiB per direction. At T=512 the per-token loop did `T × (state R+W) = 2
//! GiB device traffic per layer per direction × 30 GDN layers = 120 GiB
//! per prefill step in state traffic alone. The chunked variant does
//! 2 MiB × 30 = 60 MiB.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="gated_delta",
    subop="prep_chunk",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Reduction,
)]
#[kernel]
pub fn mt_gated_delta_prep_chunk<T>(
    conv_out: Tensor<T>,      // [B, T, 2·Hk·Dk + Hv·Dv]
    a_log: Tensor<T>,         // [Hv]
    dt_bias: Tensor<T>,       // [Hv]
    a_raw: Tensor<T>,         // [B, T, Hv]
    b_raw: Tensor<T>,         // [B, T, Hv]
    q_norm_weight: Tensor<T>, // [Hk·Dk]
    k_norm_weight: Tensor<T>, // [Hk·Dk]
    state_in: Tensor<T>,      // [B, Hv, Dv, Dk]
    mut state_out: Tensor<T>, // [B, Hv, Dv, Dk]
    mut y: Tensor<T>,         // [B, T, Hv, Dv]
    t_len: Tensor<u32>,       // [1] scalar
    #[constexpr] dk: u32,
    #[constexpr] dv: u32,
    #[constexpr] hv: u32,
    #[constexpr] hk: u32,
) {
    let dv_idx = tgid_x;
    let n = tgid_y;
    let dk_idx = tid;

    // GQA decomposition.
    let hv_idx = n - (n / hv) * hv;
    let b = n / hv;
    let hk_per_hv = hv / hk;
    let hk_idx = hv_idx / hk_per_hv;

    let n_per_t = dk / 32u32;
    let t_total = load(t_len[0]);

    let stride_b = 2u32 * hk * dk + hv * dv;
    let eps = 0.000001f32;
    let dk_f = dk.cast::<f32>();

    // Per-layer constants (loaded once per TG).
    let a_log_val = load(a_log[hv_idx]).cast::<f32>();
    let dt_bias_val = load(dt_bias[hv_idx]).cast::<f32>();
    let exp_a_log = exp(a_log_val);

    let state_base = n * dv * dk + dv_idx * dk;

    // ─── Load state into per-lane registers ONCE — persists across the T-loop.
    stack_alloc("state_reg", 8u32, "f32");
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let val = load(state_in[state_base + s_idx]).cast::<f32>();
        stack_store("state_reg", i, val);
    }

    // q_w / k_w are static across the T-loop (one row of weights per
    // hk_idx); load them once into per-lane stack so the inner T-loop
    // doesn't re-read.
    stack_alloc("q_w", 8u32, "f32");
    stack_alloc("k_w", 8u32, "f32");
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        let qw = load(q_norm_weight[hk_idx * dk + s_idx]).cast::<f32>();
        let kw = load(k_norm_weight[hk_idx * dk + s_idx]).cast::<f32>();
        stack_store("q_w", i, qw);
        stack_store("k_w", i, kw);
    }

    // Stack arrays reused per-token: q_raw / k_raw / k_cache.
    stack_alloc("q_raw", 8u32, "f32");
    stack_alloc("k_raw", 8u32, "f32");
    stack_alloc("k_cache", 8u32, "f32");

    // ─── Inner T-loop: prep + recurrence per token ──────────────────────
    for t in range(0u32, t_total, 1u32) {
        let bt = b * t_total + t;
        let conv_base = bt * stride_b;
        let q_off = conv_base + hk_idx * dk;
        let k_off = conv_base + hk * dk + hk_idx * dk;
        let v_off = conv_base + 2u32 * hk * dk + hv_idx * dv;
        let gbeta_idx = bt * hv + hv_idx;

        // ─── Phase 0a: Per-head RMSNorm of q / k ─────────────────────────
        let mut q_ssq = 0.0f32;
        let mut k_ssq = 0.0f32;
        for i in range(0u32, n_per_t, 1u32) {
            let s_idx = n_per_t * dk_idx + i;
            let qv = load(conv_out[q_off + s_idx]).cast::<f32>();
            let kv = load(conv_out[k_off + s_idx]).cast::<f32>();
            stack_store("q_raw", i, qv);
            stack_store("k_raw", i, kv);
            q_ssq = q_ssq + qv * qv;
            k_ssq = k_ssq + kv * kv;
        }
        let q_ssq_sum = simd_sum(q_ssq);
        let k_ssq_sum = simd_sum(k_ssq);
        let q_inv = rsqrt(q_ssq_sum / dk_f + eps);
        let k_inv = rsqrt(k_ssq_sum / dk_f + eps);

        // ─── Phase 0b: g / beta ──────────────────────────────────────────
        let a_raw_val = load(a_raw[gbeta_idx]).cast::<f32>();
        let b_raw_val = load(b_raw[gbeta_idx]).cast::<f32>();
        let pre_softplus = a_raw_val + dt_bias_val;
        let dt_val = log(exp(pre_softplus) + 1.0f32);
        let g_val = exp(0.0f32 - exp_a_log * dt_val);
        let beta_val = 1.0f32 / (1.0f32 + exp(0.0f32 - b_raw_val));

        // v: one read per Dv slot per token.
        let v_val = load(conv_out[v_off + dv_idx]).cast::<f32>();

        // ─── Phase 1: decay state + accumulate kv_mem; cache k_normed ────
        let mut kv_mem = 0.0f32;
        for i in range(0u32, n_per_t, 1u32) {
            let s_old = stack_load("state_reg", i);
            let s_decayed = s_old * g_val;
            stack_store("state_reg", i, s_decayed);
            let k_normed = stack_load("k_raw", i) * k_inv * stack_load("k_w", i);
            stack_store("k_cache", i, k_normed);
            kv_mem = kv_mem + s_decayed * k_normed;
        }
        let kv_mem_sum = simd_sum(kv_mem);
        let delta = (v_val - kv_mem_sum) * beta_val;

        // ─── Phase 2: rank-1 update + output projection ──────────────────
        let mut out_acc = 0.0f32;
        for i in range(0u32, n_per_t, 1u32) {
            let s_decayed = stack_load("state_reg", i);
            let k_normed = stack_load("k_cache", i);
            let s_new = s_decayed + k_normed * delta;
            stack_store("state_reg", i, s_new);
            let q_normed = stack_load("q_raw", i) * q_inv * stack_load("q_w", i);
            out_acc = out_acc + s_new * q_normed;
        }
        let out_sum = simd_sum(out_acc);

        // ─── Phase 3: lane 0 writes y[t, n, dv_idx] ──────────────────────
        if dk_idx == 0u32 {
            store(y[(bt * hv + hv_idx) * dv + dv_idx], out_sum.cast::<T>());
        }
    }

    // ─── Write final state ONCE at the end ──────────────────────────────
    for i in range(0u32, n_per_t, 1u32) {
        let s_idx = n_per_t * dk_idx + i;
        store(state_out[state_base + s_idx], stack_load("state_reg", i).cast::<T>());
    }
}

#[cfg(test)]
mod tests {
    use metaltile_core::ir::KernelMode;

    use super::*;
    use crate::bench_types::DType;

    /// Developer aid — dump the full generated MSL for inspection.
    /// `cargo test -p metaltile-std --lib --release -- ffai::gated_delta_prep_chunk::tests::dump --nocapture`
    #[test]
    fn dump() {
        use metaltile_codegen::msl::MslGenerator;
        let mut k = mt_gated_delta_prep_chunk::kernel_ir_for(DType::F32);
        k.mode = KernelMode::Reduction;
        let msl = MslGenerator::default().generate(&k).expect("codegen");
        println!("===== BEGIN MSL =====\n{}\n===== END MSL =====", msl);
    }
}
