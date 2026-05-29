//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
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

use metaltile::kernel;

/// Fused RMSNorm + paired-layout RoPE for one Q/K head per threadgroup.
#[kernel(
    bench(
        op="rms_norm_rope",
        subop="rms_norm_rope",
        class=GenericEmpty,
        tol=1e-4,
        kernel_mode=Reduction,
    )
)]
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

/// New-syntax correctness for `ffai_rms_norm_rope` (Reduction mode, one
/// threadgroup per row, `tpg = axis_size/2` — `axis_size` a multiple of 64,
/// `axis_size ≤ 2048`). Per-row oracle replays the whole-row RMSNorm scale,
/// the per-row position `pos = offset + (row / n_heads) mod seq_len`, and the
/// paired-layout rotation on dtype-rounded `x` / `w` (`inv_freqs` stays f32).
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_rms_norm_rope;
    use crate::utils::{pack_f32, unpack_f32};

    /// Small, deterministic inverse-frequency table (length `half`).
    fn inv_freq_table(half: usize) -> Vec<f32> {
        (0..half).map(|i| 1.0 / 10000.0_f32.powf(i as f32 / half as f32)).collect()
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 1e-2, 1e-1])]
    fn test_ffai_rms_norm_rope(dt: DType) -> TestSetup {
        let (axis, n_heads, seq_len, offset, eps) = (128usize, 4usize, 8usize, 5usize, 1e-5f32);
        let rows = n_heads * seq_len; // one batch
        let half = axis / 2;
        let x_raw: Vec<f32> = (0..rows * axis).map(|i| ((i % 53) as f32) * 0.07 - 1.8).collect();
        let w_raw: Vec<f32> = (0..axis).map(|i| 1.0 + 0.02 * ((i % 11) as f32 - 5.0)).collect();
        let inv_freqs = inv_freq_table(half);
        let x = unpack_f32(&pack_f32(&x_raw, dt), dt);
        let w = unpack_f32(&pack_f32(&w_raw, dt), dt);

        let mut expected = vec![0.0f32; rows * axis];
        for r in 0..rows {
            let base = r * axis;
            let ssq: f32 = (0..axis).map(|i| x[base + i] * x[base + i]).sum();
            let inv_rms = 1.0 / (ssq / axis as f32 + eps).sqrt();
            let pos = (offset + (r / n_heads) % seq_len) as f32;
            for lid in 0..half {
                let theta = pos * inv_freqs[lid];
                let (s, c) = theta.sin_cos();
                let na = x[base + lid] * w[lid] * inv_rms;
                let nb = x[base + lid + half] * w[lid + half] * inv_rms;
                expected[base + lid] = na * c - nb * s;
                expected[base + lid + half] = na * s + nb * c;
            }
        }
        TestSetup::new(ffai_rms_norm_rope::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x_raw, dt), dt))
            .input(TestBuffer::from_vec("w", pack_f32(&w_raw, dt), dt))
            .input(TestBuffer::from_vec("inv_freqs", pack_f32(&inv_freqs, DType::F32), DType::F32))
            .input(TestBuffer::zeros("out", rows * axis, dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .constexpr("axis_size", axis as u32)
            .constexpr("offset", offset as u32)
            .constexpr("n_heads", n_heads as u32)
            .constexpr("seq_len", seq_len as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [(axis / 2) as u32, 1, 1])
    }
}

/// New-syntax benchmark for `ffai_rms_norm_rope` (fused RMSNorm + RoPE, one
/// `(batch, seq, head)` row per threadgroup, axis_size=128, tpg=64).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_rms_norm_rope;

    fn f32_bytes(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[bench(name = "ffai/rms_norm_rope/rms_norm_rope", dtypes = [f32, f16, bf16])]
    fn bench_rms_norm_rope(dt: DType) -> BenchSetup {
        let (axis, n_heads, seq_len) = (128usize, 32usize, 128usize);
        let rows = n_heads * seq_len;
        let half = axis / 2;
        let inv_freqs: Vec<f32> =
            (0..half).map(|i| 1.0 / 10000.0_f32.powf(i as f32 / half as f32)).collect();
        BenchSetup::new(ffai_rms_norm_rope::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", rows * axis, dt))
            .buffer(BenchBuffer::random("w", axis, dt))
            .buffer(BenchBuffer::from_vec("inv_freqs", f32_bytes(&inv_freqs), DType::F32))
            .buffer(BenchBuffer::zeros("out", rows * axis, dt).output())
            .buffer(BenchBuffer::from_vec("eps_buf", 1e-5f32.to_le_bytes().to_vec(), DType::F32))
            .constexpr("axis_size", axis as u32)
            .constexpr("offset", 0u32)
            .constexpr("n_heads", n_heads as u32)
            .constexpr("seq_len", seq_len as u32)
            .grid_3d(rows as u32, 1, 1, [(axis / 2) as u32, 1, 1])
            .bytes_moved((2 * rows * axis * dt.size_bytes()) as u64)
    }
}
