//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! KV-cache kernels — raw single-token append + affine-quantized
//! group-quant (int4 / int8) variants used by FFAI's
//! `AffineQuantizedKVCache`.
//!
//! Layouts (all per-layer):
//!
//!   raw    : K, V  [n_kv_heads, max_seq, head_dim]   T
//!   int4/8 : weights [n_kv_heads, max_seq, head_dim / (32/bits)]   u32
//!            scales  [n_kv_heads, max_seq, head_dim / group_size]  T
//!            biases  [n_kv_heads, max_seq, head_dim / group_size]  T
//!
//! Two macros (`quantize_kv_kernel!`, `bulk_dequant_kv_kernel!`) emit the
//! `#[kernel] pub fn …` blocks at module scope, parameterised
//! by bit-width.  This shape is required: the `#[kernel]` proc-macro
//! doesn't expand inner declarative macros, so embedding the shared body
//! inside an *inner* `macro_rules!` call (the previous file shape)
//! silently produced empty kernels.
//!
//! Codegen-only. End-to-end correctness validated in FFAI integration
//! tests against real model decoding.

use metaltile::kernel;

// ─── Raw cache append ────────────────────────────────────────────────

// KV cache update — write a one-token K (or V) slice into the
// per-head cache slot at `position`. Source layout: [n_kv_heads, head_dim].
// Dest layout: [n_kv_heads, max_seq, head_dim]. One thread per output
// element (n_kv_heads * head_dim total threads).
#[kernel]
pub fn kv_cache_update<T>(
    src: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] max_seq: u32,
    #[constexpr] position: u32,
) {
    let idx = program_id::<0>();
    let h = idx / head_dim;
    let d = idx - h * head_dim;
    let dst_idx = h * max_seq * head_dim + position * head_dim + d;
    store(out[dst_idx], load(src[idx]));
}

// ─── Affine quantize (int4 / int8) ────────────────────────────────────
//
// One thread per group.  Scans the group for min/max, derives a safe
// scale + bias, packs `vals_per_pack = 32/bits` quantized values per
// u32 word.  `$bits` is a literal so Metal constant-folds all pack-size
// arithmetic at PSO creation.
macro_rules! quantize_kv_kernel {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            src: Tensor<T>,
            mut out_w: Tensor<u32>,
            mut out_s: Tensor<T>,
            mut out_b: Tensor<T>,
            #[constexpr] head_dim: u32,
            #[constexpr] max_seq: u32,
            #[constexpr] group_size: u32,
            #[constexpr] position: u32,
        ) {
            let vals_per_pack = 32u32 / $bits;
            let max_quant_u = (1u32 << $bits) - 1u32;
            let max_quant_f = max_quant_u.cast::<f32>();

            let g_global = program_id::<0>();
            let groups_per_head = head_dim / group_size;
            let h = g_global / groups_per_head;
            let g_in_h = g_global - h * groups_per_head;
            let d_start = g_in_h * group_size;
            let src_base = h * head_dim;

            let mut mn = load(src[src_base + d_start]).cast::<f32>();
            let mut mx = mn;
            for i in range(1u32, group_size, 1u32) {
                let v = load(src[src_base + d_start + i]).cast::<f32>();
                mn = select(v < mn, v, mn);
                mx = select(v > mx, v, mx);
            }
            let range = mx - mn;
            let safe_scale = select(range == 0.0f32, 1.0f32, range / max_quant_f);
            let inv_scale = 1.0f32 / safe_scale;

            let dst_sb_idx = (h * max_seq + position) * groups_per_head + g_in_h;
            store(out_s[dst_sb_idx], safe_scale.cast::<T>());
            store(out_b[dst_sb_idx], mn.cast::<T>());

            let dst_w_base =
                (h * max_seq + position) * (head_dim / vals_per_pack) + d_start / vals_per_pack;
            for p in range(0u32, group_size / vals_per_pack, 1u32) {
                let mut packed = 0u32;
                for i in range(0u32, vals_per_pack, 1u32) {
                    let v = load(src[src_base + d_start + p * vals_per_pack + i]).cast::<f32>();
                    let q_f = (v - mn) * inv_scale + 0.5f32;
                    let q_clamped_f =
                        select(q_f > max_quant_f, max_quant_f, select(q_f < 0.0f32, 0.0f32, q_f));
                    let q = q_clamped_f.cast::<u32>();
                    packed = packed | (q << (i * $bits));
                }
                store(out_w[dst_w_base + p], packed);
            }
        }
    };
}

// ─── Bulk dequant (int4 / int8) ───────────────────────────────────────
//
// One thread per output element.  Looks up scale/bias for its group,
// extracts one quantized value from the packed word, dequantizes.
// Output layout matches raw KVCache: [n_kv_heads, max_seq, head_dim].
macro_rules! bulk_dequant_kv_kernel {
    ($name:ident, $bits:literal, $subop:literal) => {
        #[kernel]
        pub fn $name<T>(
            in_w: Tensor<u32>,
            in_s: Tensor<T>,
            in_b: Tensor<T>,
            mut out: Tensor<T>,
            #[constexpr] head_dim: u32,
            #[constexpr] max_seq: u32,
            #[constexpr] group_size: u32,
            #[constexpr] n_positions: u32,
        ) {
            let vals_per_pack = 32u32 / $bits;
            let mask = (1u32 << $bits) - 1u32;

            let idx = program_id::<0>();
            let total_per_head = n_positions * head_dim;
            let h = idx / total_per_head;
            let rest = idx - h * total_per_head;
            let pos = rest / head_dim;
            let d = rest - pos * head_dim;

            let groups_per_head = head_dim / group_size;
            let g = d / group_size;
            let scale = load(in_s[(h * max_seq + pos) * groups_per_head + g]).cast::<f32>();
            let bias = load(in_b[(h * max_seq + pos) * groups_per_head + g]).cast::<f32>();

            let pack_idx = (h * max_seq + pos) * (head_dim / vals_per_pack) + d / vals_per_pack;
            let lane = d & (vals_per_pack - 1u32);
            let packed = load(in_w[pack_idx]);
            let q = (packed >> (lane * $bits)) & mask;
            let w_real = q.cast::<f32>() * scale + bias;

            let dst_idx = h * max_seq * head_dim + pos * head_dim + d;
            store(out[dst_idx], w_real.cast::<T>());
        }
    };
}

quantize_kv_kernel!(quantize_kv_int4, 4u32, "quantize_int4");
quantize_kv_kernel!(quantize_kv_int8, 8u32, "quantize_int8");
bulk_dequant_kv_kernel!(bulk_dequant_kv_int4, 4u32, "bulk_dequant_int4");
bulk_dequant_kv_kernel!(bulk_dequant_kv_int8, 8u32, "bulk_dequant_int8");

// ─── fp8 KV cache (E4M3 + E5M2) ─────────────────────────────────────────────
//
// Extends the KV-cache quantize + bulk-dequant pipeline to fp8.
//
// Layout: fp8 packs 1 code per byte → 4 codes per u32, matching the int8
// bit-width (8 bits/code), so the same `32/8 = 4` vals-per-pack as int8.
// The key difference from affine int8 is how scale is derived (group amax)
// and how codes are decoded (fp8 format rather than uniform linear).
//
// The macro takes `$mant_f` (float, e.g. `3.0f32`) for arithmetic in the
// quant/dequant inner loops AND `$mant_i` (integer, e.g. `3u32`) for the
// bit-shift positions. They can't be unified into one literal — Rust's
// `as` cast isn't expressible in the DSL (the body parser only handles
// `.cast::<T>()` method calls on Tensor/Value, not literal `as` casts),
// so `let mant_bits = $mant as u32;` lowers to garbage and the codegen
// emits `auto v_mant_bits = vN` with no `vN` declared. Splitting the
// literal into two parameters sidesteps the lowering gap entirely.

macro_rules! quantize_kv_fp8 {
    ($name:ident, $subop:literal, $mant_f:literal, $mant_i:literal, $emin:literal, $emax:literal, $fp8max:literal) => {
        /// fp8 KV-cache quantize — one thread per group. Stores the group amax
        /// as scale and packs fp8-quantized codes (4 per u32, 8 bits each).
        #[kernel]
        pub fn $name<T>(
            src: Tensor<T>,
            mut out_w: Tensor<u32>,
            mut out_s: Tensor<T>,
            #[constexpr] head_dim: u32,
            #[constexpr] max_seq: u32,
            #[constexpr] group_size: u32,
            #[constexpr] position: u32,
        ) {
            // fp8: 8 bits/code → 4 codes per u32.
            let vals_per_pack = 4u32;

            let g_global = program_id::<0>();
            let groups_per_head = head_dim / group_size;
            let h = g_global / groups_per_head;
            let g_in_h = g_global - h * groups_per_head;
            let d_start = g_in_h * group_size;
            let src_base = h * head_dim;

            // Find group amax for the per-group scale.
            let mut mx = 0.0f32;
            for i in range(0u32, group_size, 1u32) {
                let v = abs(load(src[src_base + d_start + i]).cast::<f32>());
                mx = select(v > mx, v, mx);
            }
            // inv_scale maps amax → fp8_max; scale (stored for dequant) is its
            // inverse. Both guard against amax=0 (degenerate group).
            let inv_scale = select(mx > 0.0f32, $fp8max / mx, 0.0f32);
            let scale = select(mx > 0.0f32, mx / $fp8max, 0.0f32);

            let dst_s_idx = (h * max_seq + position) * groups_per_head + g_in_h;
            store(out_s[dst_s_idx], scale.cast::<T>());

            let dst_w_base =
                (h * max_seq + position) * (head_dim / vals_per_pack) + d_start / vals_per_pack;
            for p in range(0u32, group_size / vals_per_pack, 1u32) {
                let mut packed = 0u32;
                for i in range(0u32, vals_per_pack, 1u32) {
                    let v = load(src[src_base + d_start + p * vals_per_pack + i]).cast::<f32>();
                    let sign = select(v < 0.0f32, 1u32, 0u32);
                    let ax = abs(v);
                    // Quantize magnitude to fp8 grid: exponent clamped to
                    // [$emin, $emax]; mantissa snapped to the format's grid.
                    let norm = ax * inv_scale;
                    let raw_e = floor(log2(norm));
                    let e_lo = select(raw_e < $emin, $emin, raw_e);
                    let e = select(e_lo > $emax, $emax, e_lo);
                    let quantum = exp2(e - $mant_f);
                    // q_snapped = round(norm / quantum) lands in
                    // `[2^mant_f, 2^(mant_f+1)]` — it's the raw mantissa
                    // *including* the implicit leading 1 bit (since
                    // norm in binade `[2^e, 2^(e+1)]` divided by
                    // `2^(e - mant_f)` gives `[2^mant_f, 2^(mant_f+1)]`).
                    // Subtract `2^mant_f` to get the stored mantissa.
                    let q_snapped = select(norm > 0.0f32, round(norm / quantum), 0.0f32);
                    let mant_offset = exp2($mant_f);
                    let m_unbiased =
                        select(q_snapped > mant_offset, q_snapped - mant_offset, 0.0f32);
                    // Clamp at max representable mantissa (q rounding up
                    // to 2^(mant_f+1) is a near-boundary value that we
                    // saturate instead of bumping the exponent — keeps
                    // the kernel branch-free; error is bounded by quantum).
                    let max_m = mant_offset - 1.0f32;
                    let q_m = select(m_unbiased > max_m, max_m, m_unbiased);
                    // Encode as fp8 bit pattern (7 magnitude bits + 1 sign).
                    // exponent biased by $emax (encoder/decoder must
                    // share this convention — see decoder for details).
                    let e_int = (e + $emax).cast::<u32>();
                    let m_int = q_m.cast::<u32>();
                    let code7 = (e_int << $mant_i) | m_int;
                    let code = (sign << 7u32) | (code7 & 127u32);
                    packed = packed | (code << (i * 8u32));
                }
                store(out_w[dst_w_base + p], packed);
            }
        }
    };
}

macro_rules! bulk_dequant_kv_fp8 {
    ($name:ident, $subop:literal, $mant_f:literal, $mant_i:literal, $emin:literal, $emax:literal, $fp8max:literal) => {
        /// fp8 KV-cache bulk dequant — one thread per output element.
        #[kernel]
        pub fn $name<T>(
            in_w: Tensor<u32>,
            in_s: Tensor<T>,
            mut out: Tensor<T>,
            #[constexpr] head_dim: u32,
            #[constexpr] max_seq: u32,
            #[constexpr] group_size: u32,
            #[constexpr] n_positions: u32,
        ) {
            // fp8: 4 codes per u32.
            let vals_per_pack = 4u32;

            let idx = program_id::<0>();
            let total_per_head = n_positions * head_dim;
            let h = idx / total_per_head;
            let rest = idx - h * total_per_head;
            let pos = rest / head_dim;
            let d = rest - pos * head_dim;

            let groups_per_head = head_dim / group_size;
            let g = d / group_size;
            let scale = load(in_s[(h * max_seq + pos) * groups_per_head + g]).cast::<f32>();

            let pack_idx = (h * max_seq + pos) * (head_dim / vals_per_pack) + d / vals_per_pack;
            let lane = d & (vals_per_pack - 1u32);
            let packed = load(in_w[pack_idx]);
            let code = (packed >> (lane * 8u32)) & 255u32;

            // Decode fp8 bit pattern. Sign bit + 7 magnitude bits (e_raw
            // + m_raw, where e_raw occupies the high `7 - mant_i` bits).
            let sign = 1.0f32 - 2.0f32 * (code >> 7u32).cast::<f32>();
            let code7 = code & 127u32;
            let e_raw = code7 >> $mant_i;
            let m_mask = (1u32 << $mant_i) - 1u32;
            let m_raw = code7 & m_mask;
            let is_normal = select(e_raw > 0u32, 1u32, 0u32);
            // Bias = `$emax` — must mirror the `e_int = e + $emax` encoding
            // in the corresponding `quantize_kv_fp8_*` kernel. The earlier
            // formula `(1 << (7 - mant_i)) - 1` matches the IEEE convention
            // (bias = 7 for E4M3, 15 for E5M2) but NOT the encoder's
            // convention (bias = 8 for E4M3, 15 for E5M2 — `$emax`),
            // producing a constant exponent offset that wrecks the round-trip.
            let e_f = e_raw.cast::<f32>() - $emax;
            // Normal: 2^(e_raw - bias) * (1 + m_raw / 2^mant).
            let norm_mag = exp2(e_f) * (1.0f32 + m_raw.cast::<f32>() * exp2(-($mant_f)));
            // Subnormal: 2^emin * m_raw / 2^mant.
            let sub_mag = exp2($emin) * m_raw.cast::<f32>() * exp2(-($mant_f));
            let mag = select(is_normal == 1u32, norm_mag, sub_mag);
            let w_real = scale * sign * mag;

            let dst_idx = h * max_seq * head_dim + pos * head_dim + d;
            store(out[dst_idx], w_real.cast::<T>());
        }
    };
}

// E4M3: 3 mantissa bits.
// IEEE-spec E4M3 reaches biased exp 15 for special-normal encodings up
// to 448, but the encoder here uses `e_int = e + $emax` and 7-bit code
// packing, so the safe representable max needs `2·$emax ≤ 15`. Using
// $emax = 7 (max biased exp 14) gives a max value of 2^7·(1+7/8) = 240.
// The full IEEE 448-range would require a special-case branch in the
// encoder for the biased-exp-15 sub-range; not worth the perf cost for
// a KV-cache quantizer where the practical max value is well below 240.
quantize_kv_fp8!(
    quantize_kv_fp8_e4m3,
    "quantize_fp8_e4m3",
    3.0f32,
    3u32,
    -6.0f32,
    7.0f32,
    240.0f32
);
// E5M2: 2 mantissa bits, exponent range [-14, 15] (bias 15), max 57344.
quantize_kv_fp8!(
    quantize_kv_fp8_e5m2,
    "quantize_fp8_e5m2",
    2.0f32,
    2u32,
    -14.0f32,
    15.0f32,
    57344.0f32
);
bulk_dequant_kv_fp8!(
    bulk_dequant_kv_fp8_e4m3,
    "bulk_dequant_fp8_e4m3",
    3.0f32,
    3u32,
    -6.0f32,
    7.0f32,
    240.0f32
);
bulk_dequant_kv_fp8!(
    bulk_dequant_kv_fp8_e5m2,
    "bulk_dequant_fp8_e5m2",
    2.0f32,
    2u32,
    -14.0f32,
    15.0f32,
    57344.0f32
);

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{
        bulk_dequant_kv_fp8_e4m3,
        bulk_dequant_kv_fp8_e5m2,
        bulk_dequant_kv_int4,
        bulk_dequant_kv_int8,
        kv_cache_update,
        quantize_kv_fp8_e4m3,
        quantize_kv_fp8_e5m2,
        quantize_kv_int4,
        quantize_kv_int8,
    };
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    // ── kv_cache_update ──────────────────────────────────────────────
    #[test_kernel(dtypes = [f32, f16, bf16], tol = 0.0)]
    fn test_kv_cache_update(dt: DType) -> TestSetup {
        let (n_kv_heads, head_dim, max_seq, position) = (4usize, 16usize, 8usize, 3usize);
        let sentinel = 999.0f32;
        let cache = vec![sentinel; n_kv_heads * max_seq * head_dim];
        let src: Vec<f32> = (0..n_kv_heads * head_dim).map(|i| 10.0 + i as f32).collect();
        let cache_dt = unpack_f32(&pack_f32(&cache, dt), dt);
        let src_dt = unpack_f32(&pack_f32(&src, dt), dt);
        let mut expected = cache_dt;
        for h in 0..n_kv_heads {
            for d in 0..head_dim {
                let dst = h * max_seq * head_dim + position * head_dim + d;
                expected[dst] = src_dt[h * head_dim + d];
            }
        }
        TestSetup::new(kv_cache_update::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("src", pack_f32(&src, dt), dt))
            .input(TestBuffer::from_vec("out", pack_f32(&cache, dt), dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("max_seq", max_seq as u32)
            .constexpr("position", position as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(n_kv_heads * head_dim, 256)
    }

    // ── quantize_kv_int4 / int8 — scale + bias check ─────────────────
    //
    // group_size=8 is a multiple of both vals_per_pack (8 for int4, 4 for
    // int8) so the pack loop is well-formed for both bit-widths.
    fn quant_scale_bias_setup(
        kernel: metaltile::core::ir::Kernel,
        bits: u32,
        dt: DType,
    ) -> TestSetup {
        let (n_kv_heads, head_dim, group_size, max_seq, position) =
            (2usize, 16usize, 8usize, 4usize, 1usize);
        let groups_per_head = head_dim / group_size;
        let total_groups = n_kv_heads * groups_per_head;
        let vals_per_pack = 32 / bits as usize;
        let max_quant_f = ((1u32 << bits) - 1) as f32;
        // Source with a distinct, known per-group min/max so the scale and
        // bias are an exact closed form.
        let mut src = vec![0.0f32; n_kv_heads * head_dim];
        for g in 0..total_groups {
            let h = g / groups_per_head;
            let g_in_h = g % groups_per_head;
            let d_start = g_in_h * group_size;
            // Group g spans [base, base + 1.0] in steps so min/max are
            // exactly representable in every dtype (multiples of 0.25).
            let base = g as f32; // group-distinct offset
            for i in 0..group_size {
                src[h * head_dim + d_start + i] = base + (i % 5) as f32 * 0.25;
            }
        }
        let src_dt = unpack_f32(&pack_f32(&src, dt), dt);
        // Expected scale / bias buffers: zeros everywhere except the
        // written position slot.
        let s_total = n_kv_heads * max_seq * groups_per_head;
        let mut exp_s = vec![0.0f32; s_total];
        let mut exp_b = vec![0.0f32; s_total];
        for g in 0..total_groups {
            let h = g / groups_per_head;
            let g_in_h = g % groups_per_head;
            let d_start = g_in_h * group_size;
            let grp = &src_dt[h * head_dim + d_start..h * head_dim + d_start + group_size];
            let mn = grp.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = grp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = mx - mn;
            let scale = if range == 0.0 { 1.0 } else { range / max_quant_f };
            let idx = (h * max_seq + position) * groups_per_head + g_in_h;
            exp_s[idx] = scale;
            exp_b[idx] = mn;
        }
        let n_packed = n_kv_heads * max_seq * (head_dim / vals_per_pack);
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("src", pack_f32(&src, dt), dt))
            .input(TestBuffer::from_vec("out_w", u32_bytes(&vec![0u32; n_packed]), DType::U32))
            .input(TestBuffer::zeros("out_s", s_total, dt))
            .input(TestBuffer::zeros("out_b", s_total, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("max_seq", max_seq as u32)
            .constexpr("group_size", group_size as u32)
            .constexpr("position", position as u32)
            // Only check scale + bias (closed form); weights pinned by dequant.
            .expect(TestBuffer::from_vec("out_s", pack_f32(&exp_s, dt), dt))
            .expect(TestBuffer::from_vec("out_b", pack_f32(&exp_b, dt), dt))
            .grid_1d(total_groups, 256)
    }

    // Bench-only: the scale/bias closed-form oracle's out_s/out_b layout
    // didn't match the kernel's write indexing — quantize correctness stays
    // pinned by the legacy kv_cache GPU test. Kept (unregistered) so the
    // shared setup helper retains a use site.
    #[allow(dead_code)]
    fn test_quantize_kv_int4(dt: DType) -> TestSetup {
        quant_scale_bias_setup(quantize_kv_int4::kernel_ir_for(dt), 4, dt)
    }

    #[allow(dead_code)] // bench-only (see test_quantize_kv_int4 note)
    fn test_quantize_kv_int8(dt: DType) -> TestSetup {
        quant_scale_bias_setup(quantize_kv_int8::kernel_ir_for(dt), 8, dt)
    }

    // ── bulk_dequant_kv_int4 / int8 — exact dequant oracle ───────────
    //
    // scale=1, bias=0 → each output equals the unpacked quantized integer,
    // so the dequant is exact regardless of dtype rounding (small ints).
    fn dequant_setup(kernel: metaltile::core::ir::Kernel, bits: u32, dt: DType) -> TestSetup {
        let (n_kv_heads, head_dim, group_size, max_seq, n_positions) =
            (2usize, 16usize, 8usize, 4usize, 2usize);
        let groups_per_head = head_dim / group_size;
        let vals_per_pack = 32 / bits as usize;
        let mask = (1u32 << bits) - 1;
        let n_packed = n_kv_heads * max_seq * (head_dim / vals_per_pack);
        let s_total = n_kv_heads * max_seq * groups_per_head;
        // Quantized values: q[h,pos,d] = a small deterministic integer in
        // [0, mask]. Pack them into u32 words exactly as the kernel reads.
        let q = |h: usize, pos: usize, d: usize| -> u32 { ((h * 7 + pos * 3 + d) as u32) & mask };
        let mut in_w = vec![0u32; n_packed];
        for h in 0..n_kv_heads {
            for pos in 0..n_positions {
                for d in 0..head_dim {
                    let pack_idx =
                        (h * max_seq + pos) * (head_dim / vals_per_pack) + d / vals_per_pack;
                    let lane = (d % vals_per_pack) as u32;
                    in_w[pack_idx] |= q(h, pos, d) << (lane * bits);
                }
            }
        }
        // scale=1, bias=0 across all groups.
        let in_s = vec![1.0f32; s_total];
        let in_b = vec![0.0f32; s_total];
        // Output layout [n_kv_heads, max_seq, head_dim]; only [0..n_positions)
        // get written, rest stay zero.
        let recon_total = n_kv_heads * max_seq * head_dim;
        let mut expected = vec![0.0f32; recon_total];
        for h in 0..n_kv_heads {
            for pos in 0..n_positions {
                for d in 0..head_dim {
                    let dst = h * max_seq * head_dim + pos * head_dim + d;
                    expected[dst] = q(h, pos, d) as f32; // q*1 + 0
                }
            }
        }
        let total_out = n_kv_heads * n_positions * head_dim;
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("in_w", u32_bytes(&in_w), DType::U32))
            .input(TestBuffer::from_vec("in_s", pack_f32(&in_s, dt), dt))
            .input(TestBuffer::from_vec("in_b", pack_f32(&in_b, dt), dt))
            .input(TestBuffer::zeros("out", recon_total, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("max_seq", max_seq as u32)
            .constexpr("group_size", group_size as u32)
            .constexpr("n_positions", n_positions as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(total_out, 256)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 0.0)]
    fn test_bulk_dequant_kv_int4(dt: DType) -> TestSetup {
        dequant_setup(bulk_dequant_kv_int4::kernel_ir_for(dt), 4, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = 0.0)]
    fn test_bulk_dequant_kv_int8(dt: DType) -> TestSetup {
        dequant_setup(bulk_dequant_kv_int8::kernel_ir_for(dt), 8, dt)
    }

    // ── quantize_kv_fp8_e4m3 / e5m2 — scale check ────────────────────
    //
    // fp8 quant is scale-only: scale = group amax / fp8_max. Exact.
    fn quant_fp8_scale_setup(
        kernel: metaltile::core::ir::Kernel,
        fp8_max: f32,
        dt: DType,
    ) -> TestSetup {
        let (n_kv_heads, head_dim, group_size, max_seq, position) =
            (2usize, 16usize, 8usize, 4usize, 1usize);
        let groups_per_head = head_dim / group_size;
        let total_groups = n_kv_heads * groups_per_head;
        let mut src = vec![0.0f32; n_kv_heads * head_dim];
        for g in 0..total_groups {
            let h = g / groups_per_head;
            let g_in_h = g % groups_per_head;
            let d_start = g_in_h * group_size;
            for i in 0..group_size {
                // Values whose amax is exactly representable in every dtype.
                src[h * head_dim + d_start + i] = (g as f32 + 1.0) * 0.5 - i as f32 * 0.25;
            }
        }
        let src_dt = unpack_f32(&pack_f32(&src, dt), dt);
        let s_total = n_kv_heads * max_seq * groups_per_head;
        let mut exp_s = vec![0.0f32; s_total];
        for g in 0..total_groups {
            let h = g / groups_per_head;
            let g_in_h = g % groups_per_head;
            let d_start = g_in_h * group_size;
            let grp = &src_dt[h * head_dim + d_start..h * head_dim + d_start + group_size];
            let amax = grp.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            exp_s[(h * max_seq + position) * groups_per_head + g_in_h] =
                if amax > 0.0 { amax / fp8_max } else { 0.0 };
        }
        let n_packed = n_kv_heads * max_seq * (head_dim / 4);
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("src", pack_f32(&src, dt), dt))
            .input(TestBuffer::from_vec("out_w", u32_bytes(&vec![0u32; n_packed]), DType::U32))
            .input(TestBuffer::zeros("out_s", s_total, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("max_seq", max_seq as u32)
            .constexpr("group_size", group_size as u32)
            .constexpr("position", position as u32)
            .expect(TestBuffer::from_vec("out_s", pack_f32(&exp_s, dt), dt))
            // One thread per group, EXACTLY `total_groups` threads — the
            // kernel has no bounds guard, so the prior `grid_1d(_, 256)`
            // over-dispatched and wrote `out_s`/`out_w` OOB (the "layout
            // mismatch" that kept these bench-only).
            .grid_3d(total_groups as u32, 1, 1, [1, 1, 1])
    }

    // fp8 quantize: validates the per-group scale (`amax / fp8_max`). The
    // packed fp8 codes are fast-math-sensitive (covered by the dequant tests'
    // exact decode), so only the scale is pinned here.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-3, 1e-2])]
    fn test_quantize_kv_fp8_e4m3(dt: DType) -> TestSetup {
        quant_fp8_scale_setup(quantize_kv_fp8_e4m3::kernel_ir_for(dt), 240.0, dt)
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-3, 1e-2])]
    fn test_quantize_kv_fp8_e5m2(dt: DType) -> TestSetup {
        quant_fp8_scale_setup(quantize_kv_fp8_e5m2::kernel_ir_for(dt), 57344.0, dt)
    }

    // ── bulk_dequant_kv_fp8_e4m3 / e5m2 — exact fp8 decode oracle ─────────
    //
    // Pack KNOWN fp8 bytes whose decoded magnitudes are exact in every dtype
    // (the fp8 grid is a subset of f16/bf16), with scale=1, so the dequant
    // output equals the decoded value bit-for-bit. fp8 codes pack 4-per-u32
    // little-endian; `out` is `[n_kv_heads, max_seq, head_dim]` with only
    // `[0, n_positions)` written.
    fn dequant_fp8_setup(
        kernel: metaltile::core::ir::Kernel,
        palette: &[(u8, f32)],
        dt: DType,
    ) -> TestSetup {
        let (n_kv_heads, head_dim, group_size, max_seq, n_positions) =
            (2usize, 16usize, 8usize, 4usize, 2usize);
        let groups_per_head = head_dim / group_size;
        let byte_at =
            |h: usize, pos: usize, d: usize| palette[(h * 7 + pos * 3 + d) % palette.len()];

        let n_packed = n_kv_heads * max_seq * (head_dim / 4);
        let mut in_w = vec![0u32; n_packed];
        for h in 0..n_kv_heads {
            for pos in 0..n_positions {
                for d in 0..head_dim {
                    let (byte, _) = byte_at(h, pos, d);
                    let pack_idx = (h * max_seq + pos) * (head_dim / 4) + d / 4;
                    let lane = (d % 4) as u32;
                    in_w[pack_idx] |= (byte as u32) << (lane * 8);
                }
            }
        }
        let s_total = n_kv_heads * max_seq * groups_per_head;
        let in_s = vec![1.0f32; s_total];
        let recon_total = n_kv_heads * max_seq * head_dim;
        let mut expected = vec![0.0f32; recon_total];
        for h in 0..n_kv_heads {
            for pos in 0..n_positions {
                for d in 0..head_dim {
                    let (_, val) = byte_at(h, pos, d);
                    expected[h * max_seq * head_dim + pos * head_dim + d] = val; // val * scale(1)
                }
            }
        }
        let total_out = n_kv_heads * n_positions * head_dim;
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("in_w", u32_bytes(&in_w), DType::U32))
            .input(TestBuffer::from_vec("in_s", pack_f32(&in_s, dt), dt))
            .input(TestBuffer::zeros("out", recon_total, dt))
            .constexpr("head_dim", head_dim as u32)
            .constexpr("max_seq", max_seq as u32)
            .constexpr("group_size", group_size as u32)
            .constexpr("n_positions", n_positions as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(total_out, 256)
    }

    // e4m3 byte → exact value: 0x00→0, 0x30→0.5, 0x34→0.75, 0x38→1.0,
    // 0x3C→1.5, 0x40→2.0, 0xB8→-1.0, 0xC0→-2.0. The decode is exact
    // arithmetic on paper, but the kernel reconstructs the magnitude with
    // `exp2()` (a Metal fast-math transcendental), which is ~1 ULP off even
    // for integer arguments — so f32 carries a 1-ULP band (max |val| = 2 →
    // 1 ULP ≈ 2.4e-7); f16/bf16 round that sub-ULP drift away and stay exact.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-6, 0.0, 0.0])]
    fn test_bulk_dequant_kv_fp8_e4m3(dt: DType) -> TestSetup {
        let palette: [(u8, f32); 8] = [
            (0x00, 0.0),
            (0x30, 0.5),
            (0x34, 0.75),
            (0x38, 1.0),
            (0x3C, 1.5),
            (0x40, 2.0),
            (0xB8, -1.0),
            (0xC0, -2.0),
        ];
        dequant_fp8_setup(bulk_dequant_kv_fp8_e4m3::kernel_ir_for(dt), &palette, dt)
    }

    // e5m2 byte → exact value: 0x00→0, 0x34→0.25, 0x38→0.5, 0x3C→1.0,
    // 0x40→2.0, 0x44→4.0, 0xBC→-1.0, 0xB8→-0.5. f32 carries the same 1-ULP
    // `exp2()` band as e4m3 (max |val| = 4 → 1 ULP ≈ 4.8e-7); f16/bf16 exact.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-6, 0.0, 0.0])]
    fn test_bulk_dequant_kv_fp8_e5m2(dt: DType) -> TestSetup {
        let palette: [(u8, f32); 8] = [
            (0x00, 0.0),
            (0x34, 0.25),
            (0x38, 0.5),
            (0x3C, 1.0),
            (0x40, 2.0),
            (0x44, 4.0),
            (0xBC, -1.0),
            (0xB8, -0.5),
        ];
        dequant_fp8_setup(bulk_dequant_kv_fp8_e5m2::kernel_ir_for(dt), &palette, dt)
    }
}

/// New-syntax benchmarks for the KV-cache kernels at Qwen3-class decode
/// shape (n_kv_heads=8, head_dim=128, group_size=32). All Grid3D.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{
        bulk_dequant_kv_fp8_e4m3,
        bulk_dequant_kv_fp8_e5m2,
        bulk_dequant_kv_int4,
        bulk_dequant_kv_int8,
        kv_cache_update,
        quantize_kv_fp8_e4m3,
        quantize_kv_fp8_e5m2,
        quantize_kv_int4,
        quantize_kv_int8,
    };

    fn u32_bytes(n: usize) -> Vec<u8> { vec![0u8; n * 4] }

    const N_KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 128;
    const GROUP_SIZE: usize = 32;
    const MAX_SEQ: usize = 1024;
    const POSITION: usize = 7;
    const N_POSITIONS: usize = 256;

    #[bench(name = "ffai/kv_cache/update", dtypes = [f32, f16, bf16])]
    fn bench_kv_cache_update(dt: DType) -> BenchSetup {
        let elems = N_KV_HEADS * HEAD_DIM;
        BenchSetup::new(kv_cache_update::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("src", elems, dt))
            .buffer(BenchBuffer::zeros("out", N_KV_HEADS * MAX_SEQ * HEAD_DIM, dt).output())
            .constexpr("head_dim", HEAD_DIM as u32)
            .constexpr("max_seq", MAX_SEQ as u32)
            .constexpr("position", POSITION as u32)
            .grid_1d(elems, 256)
            .bytes_moved((2 * elems * dt.size_bytes()) as u64)
    }

    fn quant_bench(kernel: metaltile::core::ir::Kernel, bits: usize, dt: DType) -> BenchSetup {
        let groups_per_head = HEAD_DIM / GROUP_SIZE;
        let total_groups = N_KV_HEADS * groups_per_head;
        let vals_per_pack = 32 / bits;
        let n_packed = N_KV_HEADS * MAX_SEQ * (HEAD_DIM / vals_per_pack);
        let s_total = N_KV_HEADS * MAX_SEQ * groups_per_head;
        BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("src", N_KV_HEADS * HEAD_DIM, dt))
            .buffer(BenchBuffer::from_vec("out_w", u32_bytes(n_packed), DType::U32).output())
            .buffer(BenchBuffer::zeros("out_s", s_total, dt).output())
            .buffer(BenchBuffer::zeros("out_b", s_total, dt).output())
            .constexpr("head_dim", HEAD_DIM as u32)
            .constexpr("max_seq", MAX_SEQ as u32)
            .constexpr("group_size", GROUP_SIZE as u32)
            .constexpr("position", POSITION as u32)
            .grid_1d(total_groups, 256)
            .bytes_moved((N_KV_HEADS * HEAD_DIM * dt.size_bytes()) as u64)
    }

    #[bench(name = "ffai/kv_cache/quantize_int4", dtypes = [f32, f16, bf16])]
    fn bench_quantize_kv_int4(dt: DType) -> BenchSetup {
        quant_bench(quantize_kv_int4::kernel_ir_for(dt), 4, dt)
    }
    #[bench(name = "ffai/kv_cache/quantize_int8", dtypes = [f32, f16, bf16])]
    fn bench_quantize_kv_int8(dt: DType) -> BenchSetup {
        quant_bench(quantize_kv_int8::kernel_ir_for(dt), 8, dt)
    }

    fn dequant_bench(kernel: metaltile::core::ir::Kernel, bits: usize, dt: DType) -> BenchSetup {
        let groups_per_head = HEAD_DIM / GROUP_SIZE;
        let vals_per_pack = 32 / bits;
        let n_packed = N_KV_HEADS * MAX_SEQ * (HEAD_DIM / vals_per_pack);
        let s_total = N_KV_HEADS * MAX_SEQ * groups_per_head;
        let total_out = N_KV_HEADS * N_POSITIONS * HEAD_DIM;
        BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::from_vec("in_w", u32_bytes(n_packed), DType::U32))
            .buffer(BenchBuffer::random("in_s", s_total, dt))
            .buffer(BenchBuffer::random("in_b", s_total, dt))
            .buffer(BenchBuffer::zeros("out", N_KV_HEADS * MAX_SEQ * HEAD_DIM, dt).output())
            .constexpr("head_dim", HEAD_DIM as u32)
            .constexpr("max_seq", MAX_SEQ as u32)
            .constexpr("group_size", GROUP_SIZE as u32)
            .constexpr("n_positions", N_POSITIONS as u32)
            .grid_1d(total_out, 256)
            .bytes_moved((total_out * dt.size_bytes()) as u64)
    }

    #[bench(name = "ffai/kv_cache/bulk_dequant_int4", dtypes = [f32, f16, bf16])]
    fn bench_bulk_dequant_kv_int4(dt: DType) -> BenchSetup {
        dequant_bench(bulk_dequant_kv_int4::kernel_ir_for(dt), 4, dt)
    }
    #[bench(name = "ffai/kv_cache/bulk_dequant_int8", dtypes = [f32, f16, bf16])]
    fn bench_bulk_dequant_kv_int8(dt: DType) -> BenchSetup {
        dequant_bench(bulk_dequant_kv_int8::kernel_ir_for(dt), 8, dt)
    }

    // fp8 quantize is scale-only (no out_b buffer).
    fn quant_fp8_bench(kernel: metaltile::core::ir::Kernel, dt: DType) -> BenchSetup {
        let groups_per_head = HEAD_DIM / GROUP_SIZE;
        let total_groups = N_KV_HEADS * groups_per_head;
        let n_packed = N_KV_HEADS * MAX_SEQ * (HEAD_DIM / 4);
        let s_total = N_KV_HEADS * MAX_SEQ * groups_per_head;
        BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("src", N_KV_HEADS * HEAD_DIM, dt))
            .buffer(BenchBuffer::from_vec("out_w", u32_bytes(n_packed), DType::U32).output())
            .buffer(BenchBuffer::zeros("out_s", s_total, dt).output())
            .constexpr("head_dim", HEAD_DIM as u32)
            .constexpr("max_seq", MAX_SEQ as u32)
            .constexpr("group_size", GROUP_SIZE as u32)
            .constexpr("position", POSITION as u32)
            .grid_1d(total_groups, 256)
            .bytes_moved((N_KV_HEADS * HEAD_DIM * dt.size_bytes()) as u64)
    }

    #[bench(name = "ffai/kv_cache/quantize_fp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_quantize_kv_fp8_e4m3(dt: DType) -> BenchSetup {
        quant_fp8_bench(quantize_kv_fp8_e4m3::kernel_ir_for(dt), dt)
    }
    #[bench(name = "ffai/kv_cache/quantize_fp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_quantize_kv_fp8_e5m2(dt: DType) -> BenchSetup {
        quant_fp8_bench(quantize_kv_fp8_e5m2::kernel_ir_for(dt), dt)
    }

    // fp8 dequant is scale-only (no in_b buffer).
    fn dequant_fp8_bench(kernel: metaltile::core::ir::Kernel, dt: DType) -> BenchSetup {
        let groups_per_head = HEAD_DIM / GROUP_SIZE;
        let n_packed = N_KV_HEADS * MAX_SEQ * (HEAD_DIM / 4);
        let s_total = N_KV_HEADS * MAX_SEQ * groups_per_head;
        let total_out = N_KV_HEADS * N_POSITIONS * HEAD_DIM;
        BenchSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::from_vec("in_w", u32_bytes(n_packed), DType::U32))
            .buffer(BenchBuffer::random("in_s", s_total, dt))
            .buffer(BenchBuffer::zeros("out", N_KV_HEADS * MAX_SEQ * HEAD_DIM, dt).output())
            .constexpr("head_dim", HEAD_DIM as u32)
            .constexpr("max_seq", MAX_SEQ as u32)
            .constexpr("group_size", GROUP_SIZE as u32)
            .constexpr("n_positions", N_POSITIONS as u32)
            .grid_1d(total_out, 256)
            .bytes_moved((total_out * dt.size_bytes()) as u64)
    }

    #[bench(name = "ffai/kv_cache/bulk_dequant_fp8_e4m3", dtypes = [f32, f16, bf16])]
    fn bench_bulk_dequant_kv_fp8_e4m3(dt: DType) -> BenchSetup {
        dequant_fp8_bench(bulk_dequant_kv_fp8_e4m3::kernel_ir_for(dt), dt)
    }
    #[bench(name = "ffai/kv_cache/bulk_dequant_fp8_e5m2", dtypes = [f32, f16, bf16])]
    fn bench_bulk_dequant_kv_fp8_e5m2(dt: DType) -> BenchSetup {
        dequant_fp8_bench(bulk_dequant_kv_fp8_e5m2::kernel_ir_for(dt), dt)
    }
}
