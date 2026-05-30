//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Scaled dot-product attention benchmark — #[kernel] DSL vs MLX metal/scaled_dot_product_attention.metal

use metaltile::kernel;

#[kernel]
pub fn mt_sdpa<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] n_kv: u32,
    #[constexpr] scale: f32,
) {
    let head = program_id::<0>();
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out0", 1024);
    threadgroup_alloc("tg_out1", 1024);
    threadgroup_alloc("tg_out2", 1024);
    threadgroup_alloc("tg_out3", 1024);
    let q_off = head * 128u32;
    let kv_base = head * n_kv * 128u32;
    let d0 = lane * 4u32;
    let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
    let q1 = load(q[q_off + d0 + 1u32]).cast::<f32>() * scale;
    let q2 = load(q[q_off + d0 + 2u32]).cast::<f32>() * scale;
    let q3 = load(q[q_off + d0 + 3u32]).cast::<f32>() * scale;
    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    let mut o3 = 0.0f32;
    for _t in range(sg, n_kv, ns) {
        let base = kv_base + _t * 128u32;
        let partial = q0 * load(k[base + d0]).cast::<f32>()
            + q1 * load(k[base + d0 + 1u32]).cast::<f32>()
            + q2 * load(k[base + d0 + 2u32]).cast::<f32>()
            + q3 * load(k[base + d0 + 3u32]).cast::<f32>();
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        o0 = o0 * factor + weight * load(v[base + d0]).cast::<f32>();
        o1 = o1 * factor + weight * load(v[base + d0 + 1u32]).cast::<f32>();
        o2 = o2 * factor + weight * load(v[base + d0 + 2u32]).cast::<f32>();
        o3 = o3 * factor + weight * load(v[base + d0 + 3u32]).cast::<f32>();
    }
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max);
        threadgroup_store("tg_sum", sg, run_sum);
    }
    threadgroup_barrier();
    if sg == 0 {
        let g_max_in = select(lane < ns, threadgroup_load("tg_max", lane), neg_infinity());
        let g_max = simd_max(g_max_in);
        let g_sum_in =
            select(lane < ns, threadgroup_load("tg_sum", lane) * exp(g_max_in - g_max), 0.0f32);
        let g_sum = simd_sum(g_sum_in);
        if lane == 0 {
            threadgroup_store("tg_max", 0, g_max);
            threadgroup_store("tg_sum", 0, g_sum);
        }
    }
    threadgroup_barrier();
    let g_max = threadgroup_load("tg_max", 0);
    let g_sum = threadgroup_load("tg_sum", 0);
    let rescale = exp(run_max - g_max) / g_sum;
    let idx = lane * ns + sg;
    threadgroup_store("tg_out0", idx, o0 * rescale);
    threadgroup_store("tg_out1", idx, o1 * rescale);
    threadgroup_store("tg_out2", idx, o2 * rescale);
    threadgroup_store("tg_out3", idx, o3 * rescale);
    threadgroup_barrier();
    if sg == 0 {
        let mut so0 = 0.0f32;
        let mut so1 = 0.0f32;
        let mut so2 = 0.0f32;
        let mut so3 = 0.0f32;
        for _g in range(0u32, ns, 1u32) {
            let ri = lane * ns + _g;
            so0 = so0 + threadgroup_load("tg_out0", ri);
            so1 = so1 + threadgroup_load("tg_out1", ri);
            so2 = so2 + threadgroup_load("tg_out2", ri);
            so3 = so3 + threadgroup_load("tg_out3", ri);
        }
        let out_off = q_off + d0;
        store(out[out_off], so0.cast::<T>());
        store(out[out_off + 1u32], so1.cast::<T>());
        store(out[out_off + 2u32], so2.cast::<T>());
        store(out[out_off + 3u32], so3.cast::<T>());
    }
}

/// New-syntax correctness + benchmark for `mt_sdpa` — the baseline decode SDPA
/// (head_dim hardcoded 128, one KV head per Q head, constexprs `n_kv` +
/// `scale`). Reuses the triple-loop `softmax(Q·Kᵀ·scale)·V` oracle from
/// `sdpa_vector` (gqa=1). Pre-this migration `mt_sdpa` had no CPU-oracle test
/// — validation flowed only through the `tile bench` head-to-head against MLX.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_sdpa;
    use crate::{
        mlx::sdpa_vector::kernel_tests::cpu_sdpa,
        utils::{pack_f32, unpack_f32},
    };

    const HEAD_DIM: usize = 128;

    fn setup(n_heads: usize, n_kv: usize, dt: DType) -> TestSetup {
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let q_f: Vec<f32> =
            (0..n_heads * HEAD_DIM).map(|i| ((i % 17) as f32 - 8.0) * 0.02).collect();
        let kv_len = n_heads * n_kv * HEAD_DIM;
        let k_f: Vec<f32> = (0..kv_len).map(|i| ((i % 23) as f32 - 11.0) * 0.02).collect();
        let v_f: Vec<f32> = (0..kv_len).map(|i| ((i % 19) as f32 - 9.0) * 0.03).collect();
        let q = unpack_f32(&pack_f32(&q_f, dt), dt);
        let k = unpack_f32(&pack_f32(&k_f, dt), dt);
        let v = unpack_f32(&pack_f32(&v_f, dt), dt);
        // gqa=1: one KV head per Q head.
        let expected = cpu_sdpa(&q, &k, &v, n_heads, n_heads, n_kv, HEAD_DIM);
        TestSetup::new(mt_sdpa::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("q", pack_f32(&q_f, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k_f, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v_f, dt), dt))
            .input(TestBuffer::zeros("out", n_heads * HEAD_DIM, dt))
            .constexpr("n_kv", n_kv as u32)
            .constexpr("scale", scale)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(n_heads as u32, 1, 1, [1024, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-3, 1e-2])]
    fn test_sdpa(dt: DType) -> TestSetup { setup(8, 64, dt) }
}

/// New-syntax benchmark for `mt_sdpa` (decode attention, head_dim 128).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_sdpa;

    const HEAD_DIM: usize = 128;

    #[bench(name = "mlx/sdpa/sdpa", dtypes = [f32, f16, bf16])]
    fn bench_sdpa(dt: DType) -> BenchSetup {
        let (n_heads, n_kv) = (32usize, 4096usize);
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let kv_len = n_heads * n_kv * HEAD_DIM;
        let bytes = 2 * kv_len * dt.size_bytes() + 2 * n_heads * HEAD_DIM * dt.size_bytes();
        BenchSetup::new(mt_sdpa::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", n_heads * HEAD_DIM, dt))
            .buffer(BenchBuffer::random("k", kv_len, dt))
            .buffer(BenchBuffer::random("v", kv_len, dt))
            .buffer(BenchBuffer::zeros("out", n_heads * HEAD_DIM, dt).output())
            .constexpr("n_kv", n_kv as u32)
            .constexpr("scale", scale)
            .with_shape_label(format!(
                "h{HEAD_DIM} kv{n_kv} nh{n_heads} {}",
                crate::bench_types::dtype_label(dt)
            ))
            .grid_3d(n_heads as u32, 1, 1, [1024, 1, 1])
            .bytes_moved(bytes as u64)
    }
}
