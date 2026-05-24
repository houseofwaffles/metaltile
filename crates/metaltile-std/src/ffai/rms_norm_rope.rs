//! Fused RMSNorm + RoPE — normalizes a Q/K head then applies the
//! rotary position embedding, in one dispatch. Saves a kernel launch
//! per Q and per K vs separate `rms_norm` + `rope` calls (the
//! post-projection q_norm/k_norm path in Qwen3-style models).
//!
//! Non-traditional (paired) RoPE layout: element `i` rotates with
//! element `i + half`. One threadgroup per row — a row is one
//! `(batch, seq_pos, head)` slice of length `axis_size`; thread `lid`
//! owns the pair `(lid, lid + half)`.
//!
//! Phase 1 — `inv_rms = rsqrt(mean(x²) + eps)` via `mt_rms_inv_scalar`
//! cross-kernel call with `partial_ssq = v1² + v2²` as the Value arg.
//! Phase 2 — `normed = w * x * inv_rms`, then rotate:
//!   `out[lid]      = normed_a·cos θ − normed_b·sin θ`
//!   `out[lid+half] = normed_a·sin θ + normed_b·cos θ`
//! where `θ = pos · inv_freqs[lid]` and the row's position is
//! `pos = offset + (row / n_heads) mod seq_len`.
//!
//! ## DISPATCH INVARIANTS
//!
//! Reduction-mode kernel.
//!
//! - **`TPG = axis_size / 2`** — one thread per rotation pair.
//! - **`axis_size` must be a multiple of 64** so `TPG` is a multiple
//!   of 32 (a whole number of simdgroups) and `TPG ≥ 32`. Common head
//!   dims 64 / 128 / 256 satisfy this.
//! - **`TPG ≤ 1024`** → `axis_size ≤ 2048`.
//! - **Grid: 1 threadgroup per row**, `program_id::<0>()` = row index.
//!
//! Codegen-only; correctness pinned by
//! `tests/rms_norm_rope_gpu_correctness.rs`.

use metaltile::{bench_kernel, kernel};

/// Fused RMSNorm + paired-layout RoPE for one Q/K head per threadgroup.
#[bench_kernel(
    op="rms_norm_rope",
    subop="rms_norm_rope",
    class=GenericEmpty,
    tol=1e-4,
    kernel_mode=Reduction,
)]
#[kernel]
pub fn ffai_rms_norm_rope<T>(
    x: Tensor<T>,
    w: Tensor<T>,
    inv_freqs: Tensor<f32>,
    out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] axis_size: u32,
    #[constexpr] offset: u32,
    #[constexpr] n_heads: u32,
    #[constexpr] seq_len: u32,
) {
    let row = program_id::<0>();
    let half = axis_size / 2u32;
    let rs = row * axis_size;
    let lid = tid;

    // Phase 1: per-thread pair → threadgroup-wide inv_rms via cross-kernel call.
    // partial_ssq is a Value arg; eps_buf and axis_size are Tensor args whose
    // names are substituted into mt_rms_inv_scalar's callee loads.
    let v1 = load(x[rs + lid]).cast::<f32>();
    let v2 = load(x[rs + lid + half]).cast::<f32>();
    let partial_ssq = v1 * v1 + v2 * v2;
    let inv_rms = mt_rms_inv_scalar(partial_ssq, eps_buf, axis_size);

    // Phase 2: weight scale + RoPE rotation.
    let l = (row / n_heads) % seq_len;
    let pos = (offset + l).cast::<f32>();
    let theta = pos * load(inv_freqs[lid]);
    let cos_t = cos(theta);
    let sin_t = sin(theta);

    let normed_a = v1 * load(w[lid]).cast::<f32>() * inv_rms;
    let normed_b = v2 * load(w[lid + half]).cast::<f32>() * inv_rms;

    store(out[rs + lid], (normed_a * cos_t - normed_b * sin_t).cast::<T>());
    store(out[rs + lid + half], (normed_a * sin_t + normed_b * cos_t).cast::<T>());
}
