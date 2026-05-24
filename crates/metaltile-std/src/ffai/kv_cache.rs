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
//! `#[kernel] pub fn …` + `inventory::submit!` blocks at module scope,
//! parameterised by bit-width.  This shape is required: the `#[kernel]`
//! proc-macro doesn't expand inner declarative macros, so embedding the
//! shared body inside an *inner* `macro_rules!` call (the previous file
//! shape) silently produced empty kernels.
//!
//! Codegen-only. End-to-end correctness validated in FFAI integration
//! tests against real model decoding.

use metaltile::{bench_kernel, kernel};

// ─── Raw cache append ────────────────────────────────────────────────

// KV cache update — write a one-token K (or V) slice into the
// per-head cache slot at `position`. Source layout: [n_kv_heads, head_dim].
// Dest layout: [n_kv_heads, max_seq, head_dim]. One thread per output
// element (n_kv_heads * head_dim total threads).
#[bench_kernel(
    op="kv_cache",
    subop="update",
    class=GenericEmpty,
    tol=0.0,
    kernel_mode=Grid3D,
)]
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
        #[bench_kernel(op="kv_cache", subop=$subop, class=GenericEmpty, tol=0.0, kernel_mode=Grid3D,)]
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
        #[bench_kernel(op="kv_cache", subop=$subop, class=GenericEmpty, tol=0.0, kernel_mode=Grid3D,)]
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
        #[bench_kernel(op="kv_cache", subop=$subop, class=GenericEmpty, tol=0.0, kernel_mode=Grid3D,)]
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
        #[bench_kernel(op="kv_cache", subop=$subop, class=GenericEmpty, tol=0.0, kernel_mode=Grid3D,)]
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
