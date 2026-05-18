//! MSL preamble emission: BF16 compatibility struct and activation helpers.
//!
//! These are emitted once at the top of the generated MSL, before the kernel
//! function, based on the `KernelFeatures` analysis.

use std::fmt::Write;

use super::features::KernelFeatures;
use crate::wl;

impl super::MslGenerator {
    /// Emit the BF16 compatibility struct for pre-Metal-3.1 targets.
    pub(super) fn emit_bf16_preamble(&self, out: &mut String) {
        wl!(out);
        wl!(out, "// BF16 compatibility struct for pre-Metal-3.1 targets");
        wl!(out, "struct bfloat16_t {{");
        wl!(out, "    uint16_t bits;");
        wl!(out, "    bfloat16_t() = default;");
        wl!(out, "    bfloat16_t(float v) {{");
        wl!(out, "        uint32_t x = as_type<uint32_t>(v);");
        wl!(out, "        bits = uint16_t((x + 0x7FFFu + ((x >> 16) & 1u)) >> 16);");
        wl!(out, "    }}");
        wl!(out, "    operator float() const {{");
        wl!(out, "        return as_type<float>(uint32_t(bits) << 16);");
        wl!(out, "    }}");
        wl!(out, "    operator float() const device {{");
        wl!(out, "        return as_type<float>(uint32_t(bits) << 16);");
        wl!(out, "    }}");
        wl!(out, "    operator float() const threadgroup {{");
        wl!(out, "        return as_type<float>(uint32_t(bits) << 16);");
        wl!(out, "    }}");
        wl!(out, "}};");
    }

    /// Emit activation helper template functions.
    pub(super) fn emit_activation_helpers(&self, feat: &KernelFeatures, out: &mut String) {
        if feat.needs_silu {
            wl!(out);
            wl!(out, "template<typename T>");
            wl!(out, "inline T mt_silu(T x) {{ return x / (T(1) + exp(-x)); }}");
        }
        if feat.needs_gelu {
            wl!(out);
            wl!(out, "template<typename T>");
            wl!(out, "inline T mt_gelu(T x) {{");
            wl!(out, "    const T k = T(0.7978845608f);");
            wl!(out, "    return T(0.5f) * x * (T(1) + tanh(k * (x + T(0.044715f) * x*x*x)));");
            wl!(out, "}}");
        }
        if feat.needs_relu {
            wl!(out);
            wl!(out, "template<typename T>");
            wl!(out, "inline T mt_relu(T x) {{ return max(T(0), x); }}");
        }
        if feat.needs_sigmoid {
            wl!(out);
            wl!(out, "template<typename T>");
            wl!(out, "inline T mt_sigmoid(T x) {{ return T(1) / (T(1) + exp(-x)); }}");
        }
        if feat.needs_erf {
            wl!(out);
            // Polynomial approximation matching MLX erf.h (max error < 1 ulp)
            wl!(out, "inline float mt_erf_impl(float a) {{");
            wl!(out, "    float r, s, t, u;");
            wl!(out, "    t = metal::abs(a);");
            wl!(out, "    s = a * a;");
            wl!(out, "    if (t > 0.927734375f) {{");
            wl!(out, "        r = metal::fma(-1.72853470e-5f, t, 3.83197126e-4f);");
            wl!(out, "        u = metal::fma(-3.88396438e-3f, t, 2.42546219e-2f);");
            wl!(out, "        r = metal::fma(r, s, u);");
            wl!(out, "        r = metal::fma(r, t, -1.06777877e-1f);");
            wl!(out, "        r = metal::fma(r, t, -6.34846687e-1f);");
            wl!(out, "        r = metal::fma(r, t, -1.28717512e-1f);");
            wl!(out, "        r = metal::fma(r, t, -t);");
            wl!(out, "        r = -(exp(r) - 1.0f);");
            wl!(out, "        r = metal::copysign(r, a);");
            wl!(out, "    }} else {{");
            wl!(out, "        r = -5.96761703e-4f;");
            wl!(out, "        r = metal::fma(r, s,  4.99119423e-3f);");
            wl!(out, "        r = metal::fma(r, s, -2.67681349e-2f);");
            wl!(out, "        r = metal::fma(r, s,  1.12819925e-1f);");
            wl!(out, "        r = metal::fma(r, s, -3.76125336e-1f);");
            wl!(out, "        r = metal::fma(r, s,  1.28379166e-1f);");
            wl!(out, "        r = metal::fma(r, a, a);");
            wl!(out, "    }}");
            wl!(out, "    return r;");
            wl!(out, "}}");
            wl!(out, "template<typename T>");
            wl!(out, "inline T mt_erf_impl(T x) {{ return T(mt_erf_impl(float(x))); }}");
        }
        if feat.needs_erfinv {
            wl!(out);
            // Inverse error function, ported from MLX erf.h (max error < 2.4 ulp)
            wl!(out, "inline float mt_erfinv_impl(float a) {{");
            wl!(out, "    auto t = metal::fma(a, -a, 1.0f);");
            wl!(out, "    t = metal::log(t);");
            wl!(out, "    float p;");
            wl!(out, "    if (metal::abs(t) > 6.125f) {{");
            wl!(out, "        p =  3.03697567e-10f;"); // 0x1.4deb44p-32
            wl!(out, "        p = metal::fma(p, t,  2.93243101e-8f);");
            wl!(out, "        p = metal::fma(p, t,  1.22150334e-6f);");
            wl!(out, "        p = metal::fma(p, t,  2.84108955e-5f);");
            wl!(out, "        p = metal::fma(p, t,  3.93552968e-4f);");
            wl!(out, "        p = metal::fma(p, t,  3.02698812e-3f);");
            wl!(out, "        p = metal::fma(p, t,  4.83185798e-3f);");
            wl!(out, "        p = metal::fma(p, t, -2.64646143e-1f);");
            wl!(out, "        p = metal::fma(p, t,  8.40016484e-1f);");
            wl!(out, "    }} else {{");
            wl!(out, "        p =  5.43877832e-9f;"); // 0x1.75c000p-28
            wl!(out, "        p = metal::fma(p, t,  1.43285448e-7f);");
            wl!(out, "        p = metal::fma(p, t,  1.22774793e-6f);");
            wl!(out, "        p = metal::fma(p, t,  1.12963626e-7f);");
            wl!(out, "        p = metal::fma(p, t, -5.61530760e-5f);");
            wl!(out, "        p = metal::fma(p, t, -1.47697632e-4f);");
            wl!(out, "        p = metal::fma(p, t,  2.31468678e-3f);");
            wl!(out, "        p = metal::fma(p, t,  1.15392581e-2f);");
            wl!(out, "        p = metal::fma(p, t, -2.32015476e-1f);");
            wl!(out, "        p = metal::fma(p, t,  8.86226892e-1f);");
            wl!(out, "    }}");
            wl!(out, "    return a * p;");
            wl!(out, "}}");
            wl!(out, "template<typename T>");
            wl!(out, "inline T mt_erfinv_impl(T x) {{ return T(mt_erfinv_impl(float(x))); }}");
        }
        if feat.needs_expm1 {
            wl!(out);
            // Metal stdlib lacks expm1(); implement via Taylor for |x| < 1e-4
            // (avoids catastrophic cancellation) and exp(x)-1 elsewhere.
            wl!(out, "// Metal lacks expm1(); accurate for small x via Taylor series.");
            wl!(out, "inline float mt_expm1_impl(float x) {{");
            wl!(out, "    if (fabs(x) < 1.0e-4f) return x + 0.5f * x * x;");
            wl!(out, "    return exp(x) - 1.0f;");
            wl!(out, "}}");
            wl!(out, "template<typename T>");
            wl!(out, "inline T mt_expm1_impl(T x) {{ return T(mt_expm1_impl(float(x))); }}");
        }
        if feat.needs_simd_product {
            wl!(out);
            // simd_size is only accessible as a kernel attribute, not in free functions.
            // Apple Silicon always has SIMD width 32, so unroll the butterfly statically.
            wl!(out, "// Manual SIMD-group product reduction (Metal has no simd_product builtin).");
            wl!(out, "// Unrolled butterfly for Apple Silicon's fixed SIMD width of 32.");
            wl!(out, "inline float __mt_simd_product(float v) {{");
            wl!(out, "    v *= simd_shuffle_down(v, 16u);");
            wl!(out, "    v *= simd_shuffle_down(v, 8u);");
            wl!(out, "    v *= simd_shuffle_down(v, 4u);");
            wl!(out, "    v *= simd_shuffle_down(v, 2u);");
            wl!(out, "    v *= simd_shuffle_down(v, 1u);");
            wl!(out, "    return v;");
            wl!(out, "}}");
        }
    }
}
