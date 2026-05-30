//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! DSv4 CSA / HCA compressor — softmax-gated weighted pool + APE.
//!
//! Per-layer KV compressor used on CSA (`compress_ratio=4`) and HCA
//! (`compress_ratio=128`) layers. CSA uses **overlap** pooling
//! (`2 * ratio = 8` raw tokens per compressed entry); HCA uses
//! **non-overlap** pooling. Both reduce a pool window of post-`wkv`-
//! projected hidden states into one compressed KV entry per stride.
//!
//! ```text
//!   for each pool window of `pool_len` raw KV entries:
//!     g[pool_len] = softmax(gate_proj @ raw)        # softmax-gated weights
//!     out         = sum_w g[w] * (kv_proj @ raw[w] + ape[w])
//! ```
//!
//! Inputs:
//!   - `raw_kv:    [pool_len, head_dim]` — already-`wkv`-projected KV
//!     stream for the pool window (host slices the cache).
//!   - `gate:      [pool_len]`           — pre-softmax gate weights
//!     (`gate_proj @ x` per raw token, gathered into the pool).
//!   - `ape:       [pool_len, head_dim]` — learned absolute-position
//!     embedding added per pool slot.
//!
//! Output:
//!   - `compressed: [head_dim]` — one compressed KV entry.
//!
//! Pool semantics — overlap (CSA, m=4): pool_len=8 raw entries per
//! one compressed entry, stride=4 between successive compressed
//! entries. Non-overlap (HCA, m=128): pool_len=128, stride=128.
//! Wrappers in Ops.swift slice `raw_kv` to the right pool window
//! per decode step; this kernel just reduces.
//!
//! The compressor's RoPE base differs per layer-type — `rope_theta=10K`
//! for CSA, `compress_rope_theta=160K` for HCA — and is applied BEFORE
//! this kernel by the upstream wrapper (rotating the last
//! `qk_rope_head_dim=64` of each raw KV row).
//!
//! ## Dispatch
//!
//! 1D grid over output `head_dim` elements. Each thread computes one
//! output position. **Single-pass online-softmax** — Flash-style
//! running (max, sum, weighted-acc) in one loop over the pool. Avoids
//! the codegen-CSE hazard with sequential `for _w` loops sharing a
//! reused `gate[_w]` load (DSL collapses them and references variables
//! out of scope across loop boundaries).

use metaltile::kernel;

#[kernel]
pub fn ffai_dsv4_compressor_pool<T>(
    raw_kv: Tensor<T>,
    gate: Tensor<f32>,
    ape: Tensor<T>,
    mut compressed: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] pool_len: u32,
) {
    let d = tid;
    if d < head_dim {
        // Online-softmax single pass: maintain running (max, sum) of
        // gate exponentials and a running weighted accumulator of
        // (raw + ape) at this thread's output dim. Rescale running
        // state by `exp(old_max - new_max)` when a larger gate is
        // seen — identical numerics to the two-pass form.
        let mut g_max = neg_infinity();
        let mut g_sum = 0.0f32;
        let mut acc = 0.0f32;
        for _w in range(0u32, pool_len, 1u32) {
            let g = load(gate[_w]);
            let new_max = select(g > g_max, g, g_max);
            let factor = exp(g_max - new_max);
            let weight = exp(g - new_max);
            let raw_val = load(raw_kv[_w * head_dim + d]).cast::<f32>();
            let ape_val = load(ape[_w * head_dim + d]).cast::<f32>();
            g_sum = g_sum * factor + weight;
            acc = acc * factor + weight * (raw_val + ape_val);
            g_max = new_max;
        }
        // Implicit narrowing per playbook §"DSL implicit Store coercion".
        store(compressed[d], acc / g_sum);
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_dsv4_compressor_pool;
    use crate::utils::{pack_f32, unpack_f32};

    fn cpu_reference(
        raw_kv: &[f32],
        gate: &[f32],
        ape: &[f32],
        pool_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut soft = vec![0f32; pool_len];
        let g_max = gate.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut g_sum = 0f32;
        for w in 0..pool_len {
            soft[w] = (gate[w] - g_max).exp();
            g_sum += soft[w];
        }
        for w in 0..pool_len {
            soft[w] /= g_sum;
        }
        let mut out = vec![0f32; head_dim];
        for d in 0..head_dim {
            let mut acc = 0f32;
            for w in 0..pool_len {
                acc += soft[w] * (raw_kv[w * head_dim + d] + ape[w * head_dim + d]);
            }
            out[d] = acc;
        }
        out
    }

    fn setup(pool_len: usize, head_dim: usize, dt: DType) -> TestSetup {
        let raw_kv: Vec<f32> =
            (0..pool_len * head_dim).map(|i| (i as f32 * 0.013 - 1.7).sin() * 1.2).collect();
        let ape: Vec<f32> =
            (0..pool_len * head_dim).map(|i| (i as f32 * 0.021 - 0.4).cos() * 0.6).collect();
        let gate: Vec<f32> = (0..pool_len).map(|w| (w as f32 - 4.0) * 0.3).collect();
        let raw_dt = unpack_f32(&pack_f32(&raw_kv, dt), dt);
        let ape_dt = unpack_f32(&pack_f32(&ape, dt), dt);
        let expected = cpu_reference(&raw_dt, &gate, &ape_dt, pool_len, head_dim);
        TestSetup::new(ffai_dsv4_compressor_pool::kernel_ir_for(dt))
            .input(TestBuffer::from_vec("raw_kv", pack_f32(&raw_kv, dt), dt))
            .input(TestBuffer::from_vec("gate", pack_f32(&gate, DType::F32), DType::F32))
            .input(TestBuffer::from_vec("ape", pack_f32(&ape, dt), dt))
            .input(TestBuffer::zeros("compressed", head_dim, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("pool_len", pool_len as u32)
            .expect(TestBuffer::from_vec("compressed", pack_f32(&expected, dt), dt))
            .grid_1d(head_dim, 256)
    }

    /// CSA overlap-pool — 8 raw tokens, DSv4 head_dim=512.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_compressor_pool_csa(dt: DType) -> TestSetup { setup(8, 512, dt) }

    /// HCA non-overlap-pool — 128 raw tokens, DSv4 head_dim=512.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_compressor_pool_hca(dt: DType) -> TestSetup { setup(128, 512, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::ffai_dsv4_compressor_pool;

    #[bench(name = "ffai/dsv4_compressor_pool_csa", dtypes = [f32, f16, bf16])]
    fn bench_csa(dt: DType) -> BenchSetup {
        let (pool, head_dim) = (8usize, 512usize);
        BenchSetup::new(ffai_dsv4_compressor_pool::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("raw_kv", pool * head_dim, dt))
            .buffer(BenchBuffer::random("gate", pool, DType::F32))
            .buffer(BenchBuffer::random("ape", pool * head_dim, dt))
            .buffer(BenchBuffer::zeros("compressed", head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("pool_len", pool as u32)
            .grid_1d(head_dim, 256)
            .bytes_moved(
                ((2 * pool * head_dim + pool) * dt.size_bytes() + head_dim * dt.size_bytes())
                    as u64,
            )
    }

    #[bench(name = "ffai/dsv4_compressor_pool_hca", dtypes = [f32, f16, bf16])]
    fn bench_hca(dt: DType) -> BenchSetup {
        let (pool, head_dim) = (128usize, 512usize);
        BenchSetup::new(ffai_dsv4_compressor_pool::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("raw_kv", pool * head_dim, dt))
            .buffer(BenchBuffer::random("gate", pool, DType::F32))
            .buffer(BenchBuffer::random("ape", pool * head_dim, dt))
            .buffer(BenchBuffer::zeros("compressed", head_dim, dt).output())
            .constexpr("head_dim", head_dim as u32)
            .constexpr("pool_len", pool as u32)
            .grid_1d(head_dim, 256)
            .bytes_moved(
                ((2 * pool * head_dim + pool) * dt.size_bytes() + head_dim * dt.size_bytes())
                    as u64,
            )
    }
}
