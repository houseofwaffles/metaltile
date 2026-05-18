//! Quantized MatVec benchmark — #[kernel] DSL vs MLX metal/quantized.metal

use metaltile::{bench_kernel, kernel};
static QUANTIZED_SHAPES: &[(usize, usize)] = &[(4096, 4096)];

#[bench_kernel(
    op="quantized",
    subop="qmv",
    class=QuantizedMatVec,
    shapes=&QUANTIZED_SHAPES,
    group_size=64,
    tpg=64,
    tol=1e-3,
    mlx="affine_qmv_fast_float16_t_gs_64_b_4_batch_0",
    metal_file="quantized.metal",
    dtypes=crate::spec::F32_ONLY,
)]
#[kernel]
pub fn mt_qmv_f32(
    w: Tensor<u32>,
    scales: Tensor<f32>,
    biases: Tensor<f32>,
    x: Tensor<f32>,
    out: Tensor<f32>,
    #[constexpr] k: u32,
    #[constexpr] gs_per_row: u32,
) {
    let row = program_id::<0>();
    let packs_per_row = k / 8u32;
    let w_base = row * packs_per_row;
    let sb_base = row * gs_per_row;
    let mut acc = 0.0f32;
    for _g in range(tid, gs_per_row, lsize) {
        let s = load(scales[sb_base + _g]);
        let bias = load(biases[sb_base + _g]);
        let g_w_base = w_base + _g * 8u32;
        let g_x_base = _g * 64u32;
        for _p in range(0u32, 8u32, 1u32) {
            let packed = load(w[g_w_base + _p]);
            let xb = g_x_base + _p * 8u32;
            for _b in range(0u32, 8u32, 1u32) {
                let shift = _b * 4u32;
                let int4_val = (packed >> shift) & 15u32;
                let xi = load(x[xb + _b]);
                acc = acc + (s * (int4_val * 1.0f32) + bias) * xi;
            }
        }
    }
    let result = reduce_sum(acc);
    store(out[row], result);
}

// ─── mt_affine_dequantize_int4 ─────────────────────────────────────────
//
// One thread per pack (8 nibbles in one uint32). For each output i in
// 0..8: `q = (val >> (i*4)) & 0xf`, then `out[oindex+i] = scale * q + bias`
// where scale/bias are looked up by group index `oindex / group_size`.
//
// Faithful port of MLX `affine_dequantize<T, group_size, 4>` from
// `quantized.h`. Both kernels read the same byte stream and produce the
// same output (MLX views weights as `uint8_t*`, ours as `Tensor<u32>` —
// same bits, different lens).
#[bench_kernel(
    op="affine",
    subop="dequantize_int4",
    class=AffineDequantize,
    bits=4,
    group_size=64,
    n_groups=4096,
    batch=1,
    tpg=32,
    // tol=1e-2 — bf16 round-trip error scales with max_q (= 15). At
    // n_groups=4096 the worst-case absolute drift is ~3e-3.
    tol=1e-2,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_dequantize_int4<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let pack_idx = program_id::<0>();
    let pack_factor = 8u32;
    let oindex = pack_idx * pack_factor;
    let g_idx = oindex / group_size;

    let scale = load(scales[g_idx]).cast::<f32>();
    let bias = load(biases[g_idx]).cast::<f32>();
    let val = load(w[pack_idx]);

    let q0 = (val >> 0u32) & 15u32;
    let q1 = (val >> 4u32) & 15u32;
    let q2 = (val >> 8u32) & 15u32;
    let q3 = (val >> 12u32) & 15u32;
    let q4 = (val >> 16u32) & 15u32;
    let q5 = (val >> 20u32) & 15u32;
    let q6 = (val >> 24u32) & 15u32;
    let q7 = (val >> 28u32) & 15u32;

    store(out[oindex + 0u32], (scale * q0.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 1u32], (scale * q1.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 2u32], (scale * q2.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 3u32], (scale * q3.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 4u32], (scale * q4.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 5u32], (scale * q5.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 6u32], (scale * q6.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 7u32], (scale * q7.cast::<f32>() + bias).cast::<T>());
}

// ─── mt_affine_quantize_int4 ───────────────────────────────────────────
//
// Inverse of dequantize: one threadgroup per group, finds min/max over
// the group, computes scale/bias, then packs 8 nibbles per uint32. The
// per-group nature means no cross-threadgroup sync is needed.
//
// MLX's `affine_quantize` uses a 32-thread simd-group cooperative reduce
// across `group_size` elements; we use the same shape (one threadgroup
// of 32 threads per group) and reduce via `simd_min` / `simd_max`. After
// the reduction lane 0 writes the scale + bias and packs the nibbles
// (serial per lane but small — `group_size / 8` packs per group).
//
// Restriction: hardcodes group_size=64 and bits=4 in the unrolling
// (`group_size / 32 = 2` values per thread, 8 nibbles per uint32).
// Bigger group sizes or other bit widths follow the same template with
// different constants.
#[bench_kernel(
    op="affine",
    subop="quantize_int4",
    class=AffineQuantize,
    bits=4,
    group_size=64,
    n_groups=4096,
    batch=1,
    tpg=32,
    tol=1e-1,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_quantize_int4<T>(
    w: Tensor<T>,
    mut out: Tensor<u32>,
    mut scales: Tensor<T>,
    mut biases: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let g_idx = tgid_x;
    let lane = tid;
    let in_base = g_idx * group_size;

    let v0 = load(w[in_base + lane * 2u32]).cast::<f32>();
    let v1 = load(w[in_base + lane * 2u32 + 1u32]).cast::<f32>();
    let local_min = select(v0 < v1, v0, v1);
    let local_max = select(v0 > v1, v0, v1);
    let w_min = simd_min(local_min);
    let w_max = simd_max(local_max);

    let n_bins = 15.0f32;
    let raw_scale = (w_max - w_min) / n_bins;
    let eps = 1.0e-7f32;
    let scale = select(raw_scale < eps, 1.0f32, raw_scale);
    let inv_scale = 1.0f32 / scale;
    let bias = w_min;

    // Lane 0 packs serially. `simd_or` isn't exposed in the DSL today,
    // so we trade a tiny amount of parallelism for codegen simplicity —
    // packing cost is negligible against the cooperative reduction above.
    if lane == 0u32 {
        store(scales[g_idx], scale.cast::<T>());
        store(biases[g_idx], bias.cast::<T>());

        let packs_per_group = group_size / 8u32;
        let out_base = g_idx * packs_per_group;
        for p in range(0u32, packs_per_group, 1u32) {
            let mut acc = 0u32;
            for k in range(0u32, 8u32, 1u32) {
                let v = load(w[in_base + p * 8u32 + k]).cast::<f32>();
                let q_f = (v - bias) * inv_scale + 0.5f32;
                let q_c = select(q_f > 15.0f32, 15.0f32, select(q_f < 0.0f32, 0.0f32, q_f));
                let q = q_c.cast::<u32>();
                acc = acc | (q << (k * 4u32));
            }
            store(out[out_base + p], acc);
        }
    }
}

// ─── mt_affine_dequantize_int8 ─────────────────────────────────────────
//
// One thread per pack (4 bytes in one uint32). Same shape as int4 but
// each pack covers 4 output values instead of 8, and bit-extraction
// shifts by multiples of 8 instead of 4.
//
// Faithful port of MLX `affine_dequantize<T, group_size, 8>`.
#[bench_kernel(
    op="affine",
    subop="dequantize_int8",
    class=AffineDequantize,
    bits=8,
    group_size=64,
    n_groups=4096,
    batch=1,
    tpg=32,
    // tol=1e-1 — int8 max_q=255 amplifies bf16 round-trip drift; the
    // worst case at n_groups=4096 is ~5e-2.
    tol=1e-1,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_dequantize_int8<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let pack_idx = program_id::<0>();
    let pack_factor = 4u32;
    let oindex = pack_idx * pack_factor;
    let g_idx = oindex / group_size;

    let scale = load(scales[g_idx]).cast::<f32>();
    let bias = load(biases[g_idx]).cast::<f32>();
    let val = load(w[pack_idx]);

    let q0 = (val >> 0u32) & 255u32;
    let q1 = (val >> 8u32) & 255u32;
    let q2 = (val >> 16u32) & 255u32;
    let q3 = (val >> 24u32) & 255u32;

    store(out[oindex + 0u32], (scale * q0.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 1u32], (scale * q1.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 2u32], (scale * q2.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 3u32], (scale * q3.cast::<f32>() + bias).cast::<T>());
}

// ─── mt_affine_quantize_int8 ───────────────────────────────────────────
#[bench_kernel(
    op="affine",
    subop="quantize_int8",
    class=AffineQuantize,
    bits=8,
    group_size=64,
    n_groups=4096,
    batch=1,
    tpg=32,
    tol=1e-1,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_quantize_int8<T>(
    w: Tensor<T>,
    mut out: Tensor<u32>,
    mut scales: Tensor<T>,
    mut biases: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let g_idx = tgid_x;
    let lane = tid;
    let in_base = g_idx * group_size;

    let v0 = load(w[in_base + lane * 2u32]).cast::<f32>();
    let v1 = load(w[in_base + lane * 2u32 + 1u32]).cast::<f32>();
    let local_min = select(v0 < v1, v0, v1);
    let local_max = select(v0 > v1, v0, v1);
    let w_min = simd_min(local_min);
    let w_max = simd_max(local_max);

    let n_bins = 255.0f32;
    let raw_scale = (w_max - w_min) / n_bins;
    let eps = 1.0e-7f32;
    let scale = select(raw_scale < eps, 1.0f32, raw_scale);
    let inv_scale = 1.0f32 / scale;
    let bias = w_min;

    if lane == 0u32 {
        store(scales[g_idx], scale.cast::<T>());
        store(biases[g_idx], bias.cast::<T>());

        let packs_per_group = group_size / 4u32;
        let out_base = g_idx * packs_per_group;
        for p in range(0u32, packs_per_group, 1u32) {
            let mut acc = 0u32;
            for k in range(0u32, 4u32, 1u32) {
                let v = load(w[in_base + p * 4u32 + k]).cast::<f32>();
                let q_f = (v - bias) * inv_scale + 0.5f32;
                let q_c = select(q_f > 255.0f32, 255.0f32, select(q_f < 0.0f32, 0.0f32, q_f));
                let q = q_c.cast::<u32>();
                acc = acc | (q << (k * 8u32));
            }
            store(out[out_base + p], acc);
        }
    }
}

// ─── Byte-stream dequant variants (int3 / int5 / int6) ───────────────
//
// Non-power-of-2 bit widths can't pack cleanly into a uint32, so each
// pack spans `bytes_per_pack` bytes that may cross a uint32 boundary.
// The runner allocates a one-uint32 sentinel past the end so the always-
// on `w[u_idx0 + 1]` load is safe even for the last pack.
//
// Bit layouts match MLX `affine_dequantize<T, group_size, {3,5,6}>`
// exactly.

#[bench_kernel(
    op="affine",
    subop="dequantize_int3",
    class=AffineDequantize,
    bits=3,
    group_size=32,
    n_groups=4096,
    batch=1,
    tpg=16,
    // tol=5e-3 — int3 max_q=7; worst-case bf16 drift at n_groups=4096
    // is ~1e-3.
    tol=5e-3,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_dequantize_int3<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let pack_idx = program_id::<0>();
    let pack_factor = 8u32;
    let bytes_per_pack = 3u32;
    let oindex = pack_idx * pack_factor;
    let g_idx = oindex / group_size;

    let scale = load(scales[g_idx]).cast::<f32>();
    let bias = load(biases[g_idx]).cast::<f32>();

    let byte_off = pack_idx * bytes_per_pack;
    let u_idx0 = byte_off / 4u32;
    let u0 = load(w[u_idx0]);
    let u1 = load(w[u_idx0 + 1u32]);

    let s0 = byte_off & 3u32;
    let s1 = (byte_off + 1u32) & 3u32;
    let s2 = (byte_off + 2u32) & 3u32;
    let in0_0 = (byte_off + 0u32) / 4u32 == u_idx0;
    let in0_1 = (byte_off + 1u32) / 4u32 == u_idx0;
    let in0_2 = (byte_off + 2u32) / 4u32 == u_idx0;
    let b0 = (select(in0_0, u0, u1) >> (s0 * 8u32)) & 255u32;
    let b1 = (select(in0_1, u0, u1) >> (s1 * 8u32)) & 255u32;
    let b2 = (select(in0_2, u0, u1) >> (s2 * 8u32)) & 255u32;

    let q0 = b0 & 7u32;
    let q1 = (b0 >> 3u32) & 7u32;
    let q2 = ((b0 >> 6u32) & 3u32) | ((b1 & 1u32) << 2u32);
    let q3 = (b1 >> 1u32) & 7u32;
    let q4 = (b1 >> 4u32) & 7u32;
    let q5 = ((b1 >> 7u32) & 1u32) | ((b2 & 3u32) << 1u32);
    let q6 = (b2 >> 2u32) & 7u32;
    let q7 = (b2 >> 5u32) & 7u32;

    store(out[oindex + 0u32], (scale * q0.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 1u32], (scale * q1.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 2u32], (scale * q2.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 3u32], (scale * q3.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 4u32], (scale * q4.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 5u32], (scale * q5.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 6u32], (scale * q6.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 7u32], (scale * q7.cast::<f32>() + bias).cast::<T>());
}

#[bench_kernel(
    op="affine",
    subop="dequantize_int5",
    class=AffineDequantize,
    bits=5,
    group_size=32,
    n_groups=4096,
    batch=1,
    tpg=16,
    tol=1e-2,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_dequantize_int5<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let pack_idx = program_id::<0>();
    let pack_factor = 8u32;
    let bytes_per_pack = 5u32;
    let oindex = pack_idx * pack_factor;
    let g_idx = oindex / group_size;

    let scale = load(scales[g_idx]).cast::<f32>();
    let bias = load(biases[g_idx]).cast::<f32>();

    let byte_off = pack_idx * bytes_per_pack;
    let u_idx0 = byte_off / 4u32;
    let u0 = load(w[u_idx0]);
    let u1 = load(w[u_idx0 + 1u32]);

    let s0 = byte_off & 3u32;
    let s1 = (byte_off + 1u32) & 3u32;
    let s2 = (byte_off + 2u32) & 3u32;
    let s3 = (byte_off + 3u32) & 3u32;
    let s4 = (byte_off + 4u32) & 3u32;
    let in0_0 = (byte_off + 0u32) / 4u32 == u_idx0;
    let in0_1 = (byte_off + 1u32) / 4u32 == u_idx0;
    let in0_2 = (byte_off + 2u32) / 4u32 == u_idx0;
    let in0_3 = (byte_off + 3u32) / 4u32 == u_idx0;
    let in0_4 = (byte_off + 4u32) / 4u32 == u_idx0;
    let b0 = (select(in0_0, u0, u1) >> (s0 * 8u32)) & 255u32;
    let b1 = (select(in0_1, u0, u1) >> (s1 * 8u32)) & 255u32;
    let b2 = (select(in0_2, u0, u1) >> (s2 * 8u32)) & 255u32;
    let b3 = (select(in0_3, u0, u1) >> (s3 * 8u32)) & 255u32;
    let b4 = (select(in0_4, u0, u1) >> (s4 * 8u32)) & 255u32;

    let q0 = b0 & 31u32;
    let q1 = ((b0 >> 5u32) & 7u32) | ((b1 & 3u32) << 3u32);
    let q2 = (b1 >> 2u32) & 31u32;
    let q3 = ((b1 >> 7u32) & 1u32) | ((b2 & 15u32) << 1u32);
    let q4 = ((b2 >> 4u32) & 15u32) | ((b3 & 1u32) << 4u32);
    let q5 = (b3 >> 1u32) & 31u32;
    let q6 = ((b3 >> 6u32) & 3u32) | ((b4 & 7u32) << 2u32);
    let q7 = (b4 >> 3u32) & 31u32;

    store(out[oindex + 0u32], (scale * q0.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 1u32], (scale * q1.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 2u32], (scale * q2.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 3u32], (scale * q3.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 4u32], (scale * q4.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 5u32], (scale * q5.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 6u32], (scale * q6.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 7u32], (scale * q7.cast::<f32>() + bias).cast::<T>());
}

#[bench_kernel(
    op="affine",
    subop="dequantize_int6",
    class=AffineDequantize,
    bits=6,
    group_size=32,
    n_groups=4096,
    batch=1,
    tpg=16,
    // tol=5e-2 — int6 max_q=63; worst-case bf16 drift at n_groups=4096
    // is ~1.3e-2.
    tol=5e-2,
    metal_file="quantized.metal",
)]
#[kernel]
pub fn mt_affine_dequantize_int6<T>(
    w: Tensor<u32>,
    scales: Tensor<T>,
    biases: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] group_size: u32,
) {
    let pack_idx = program_id::<0>();
    let pack_factor = 4u32;
    let bytes_per_pack = 3u32;
    let oindex = pack_idx * pack_factor;
    let g_idx = oindex / group_size;

    let scale = load(scales[g_idx]).cast::<f32>();
    let bias = load(biases[g_idx]).cast::<f32>();

    let byte_off = pack_idx * bytes_per_pack;
    let u_idx0 = byte_off / 4u32;
    let u0 = load(w[u_idx0]);
    let u1 = load(w[u_idx0 + 1u32]);

    let s0 = byte_off & 3u32;
    let s1 = (byte_off + 1u32) & 3u32;
    let s2 = (byte_off + 2u32) & 3u32;
    let in0_0 = (byte_off + 0u32) / 4u32 == u_idx0;
    let in0_1 = (byte_off + 1u32) / 4u32 == u_idx0;
    let in0_2 = (byte_off + 2u32) / 4u32 == u_idx0;
    let b0 = (select(in0_0, u0, u1) >> (s0 * 8u32)) & 255u32;
    let b1 = (select(in0_1, u0, u1) >> (s1 * 8u32)) & 255u32;
    let b2 = (select(in0_2, u0, u1) >> (s2 * 8u32)) & 255u32;

    let q0 = b0 & 63u32;
    let q1 = ((b0 >> 6u32) & 3u32) | ((b1 & 15u32) << 2u32);
    let q2 = ((b1 >> 4u32) & 15u32) | ((b2 & 3u32) << 4u32);
    let q3 = (b2 >> 2u32) & 63u32;

    store(out[oindex + 0u32], (scale * q0.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 1u32], (scale * q1.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 2u32], (scale * q2.cast::<f32>() + bias).cast::<T>());
    store(out[oindex + 3u32], (scale * q3.cast::<f32>() + bias).cast::<T>());
}
