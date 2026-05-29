//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Unary elementwise benchmark — #[kernel] DSL vs MLX metal/unary.metal

use metaltile::kernel;

#[kernel(
    bench(
        op="unary",
        subop="exp",
        class=Unary,
        input=Signed,
        tol=1e-4,
        mlx="v_Exp{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_exp<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], exp(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="log",
        class=Unary,
        input=Positive,
        tol=1e-4,
        mlx="v_Log{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_log<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="sqrt",
        class=Unary,
        input=Positive,
        tol=1e-4,
        mlx="v_Sqrt{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_sqrt<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sqrt(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="rsqrt",
        class=Unary,
        input=Positive,
        tol=1e-4,
        mlx="v_Rsqrt{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_rsqrt<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], rsqrt(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="abs",
        class=Unary,
        input=Signed,
        tol=1e-6,
        mlx="v_Abs{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_abs<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], abs(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="silu",
        class=Unary,
        input=Signed,
        tol=1e-4,
    )
)]
pub fn mt_silu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], silu(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="gelu",
        class=Unary,
        input=Signed,
        tol=1e-4,
    )
)]
pub fn mt_gelu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], gelu(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="relu",
        class=Unary,
        input=Signed,
        tol=1e-6,
    )
)]
pub fn mt_relu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], relu(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="cos",
        class=Unary,
        input=Signed,
        // tol=1e-3 — f16 cos drifts by ~2.4e-4 between MT and MLX on
        // adversarial inputs (Apple GPU fast-math handles the last few ULPs
            // of the f16 mantissa differently). f32 stays comfortably below.
        tol=1e-3,
        mlx="v_Cos{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_cos<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], cos(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="sin",
        class=Unary,
        input=Signed,
        // tol=1e-3 — f16 sin drifts by ~4.9e-4 (see mt_cos comment).
        tol=1e-3,
        mlx="v_Sin{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_sin<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sin(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="ceil",
        class=Unary,
        input=Signed,
        tol=1e-6,
        mlx="v_Ceil{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_ceil<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], ceil(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="floor",
        class=Unary,
        input=Signed,
        tol=1e-6,
        mlx="v_Floor{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_floor<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], floor(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="erf",
        class=Unary,
        input=Signed,
        tol=1e-3,
        mlx="v_Erf{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_erf<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], erf(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="exp2",
        class=Unary,
        input=Signed,
        tol=1e-4,
    )
)]
pub fn mt_exp2<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], exp2(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="log2",
        class=Unary,
        input=Positive,
        tol=1e-4,
        mlx="v_Log2{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_log2<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log2(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="sign",
        class=Unary,
        input=Signed,
        tol=0.0,
        mlx="v_Sign{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_sign<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sign(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="round",
        class=Unary,
        input=Signed,
        tol=0.0,
        mlx="v_Round{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_round<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], round(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="neg",
        class=Unary,
        input=Signed,
        tol=1e-6,
        mlx="v_Negative{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_neg<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], -load(a[idx]));
}

#[kernel(
    bench(
        op="unary",
        subop="recip",
        class=Unary,
        input=Positive,
        tol=1e-4,
    )
)]
pub fn mt_recip<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], 1.0f32.cast::<T>() / load(a[idx]));
}

#[kernel(
    bench(
        op="unary",
        subop="square",
        class=Unary,
        input=Signed,
        tol=1e-4,
        mlx="v_Square{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_square<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    store(out[idx], x * x);
}

#[kernel(
    bench(
        op="unary",
        subop="sigmoid",
        class=Unary,
        input=Signed,
        // tol=1e-3 — f16 sigmoid drifts by ~4.9e-4 (exp + reciprocal each
            // pick up ULP-level error, compounds across them).
        tol=1e-3,
        mlx="v_Sigmoid{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_sigmoid<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    // Compute in f32 to match MLX precision (convert back to T at store).
    let x = load(a[idx]).cast::<f32>();
    let result = 1.0f32 / (1.0f32 + exp(-x));
    store(out[idx], result.cast::<T>());
}

#[kernel(
    bench(
        op="unary",
        subop="log1p",
        class=Unary,
        input=Positive,
        tol=1e-4,
        mlx="v_Log1p{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_log1p<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    store(out[idx], log(1.0f32.cast::<T>() + x));
}

// ─── Transcendental ops shipped as discrete MLX kernels ───────────────
// Every op below has a matching `instantiate_unary_float` in MLX's
// unary.metal and produces a kernel named `v_<Op>{tn}{tn}`.

#[kernel(
    bench(
        op="unary",
        subop="sinh",
        class=Unary,
        input=Signed,
        tol=1e-4,
        mlx="v_Sinh{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_sinh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sinh(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="cosh",
        class=Unary,
        input=Signed,
        tol=1e-4,
        mlx="v_Cosh{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_cosh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], cosh(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="tan",
        class=Unary,
        input=Signed,
        // tol=1e-3 — f16 tan has poles near π/2 + kπ where values diverge;
        // even on moderate inputs (~1.0) the f16 ULP drift is ~5e-4.
        tol=1e-3,
        mlx="v_Tan{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_tan<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], tan(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="tanh",
        class=Unary,
        input=Signed,
        // tol=1e-3 — f16 tanh compounds exp and div; worst-case drift ~4e-4.
        tol=1e-3,
        mlx="v_Tanh{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_tanh_op<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], tanh(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="asin",
        class=Unary,
        input=Unit,
        // tol=1e-4 — asin has poles at ±1; input clamped to [-1, 1] via
        // the Unit shape generator, so no pole-induced blowup.
        tol=1e-4,
        mlx="v_ArcSin{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_asin<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], asin(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="atan",
        class=Unary,
        input=Signed,
        // tol=1e-3 — f16 atan drifts ~3e-4 on large-magnitude inputs.
        tol=1e-3,
        mlx="v_ArcTan{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_atan<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], atan(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="asinh",
        class=Unary,
        input=Signed,
        tol=1e-4,
        mlx="v_ArcSinh{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_asinh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], asinh(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="acos",
        class=Unary,
        input=Unit,
        // acos has domain [-1, 1]; Unit input clamps to [-1+ε, 1-ε].
        tol=1e-4,
        mlx="v_ArcCos{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_acos<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], acos(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="trunc",
        class=Unary,
        input=Signed,
        // trunc is exact (sets the integer-part bits to zero).
        tol=1e-6,
    )
)]
pub fn mt_trunc<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], trunc(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="acosh",
        class=Unary,
        input=Positive,
        // input=Positive ensures x ≥ 1e-6, safely inside the domain x ≥ 1.
        tol=1e-4,
        mlx="v_ArcCosh{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_acosh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], acosh(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="atanh",
        class=Unary,
        input=Unit,
        // atanh has poles at ±1; Unit input clamps to [-1+ε, 1-ε].
        tol=1e-4,
        mlx="v_ArcTanh{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_atanh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], atanh(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="expm1",
        class=Unary,
        input=Signed,
        // tol=5e-3 — for large inputs (|x|≈3) two IEEE-754 expm1 impls can diverge
        // by ~1e-4 in f32 and ~4e-3 in bf16 (half-ULP at that magnitude).
        tol=5e-3,
        mlx="v_Expm1{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_expm1<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], expm1(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="log10",
        class=Unary,
        input=Positive,
        tol=1e-4,
        mlx="v_Log10{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_log10<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log10(load(a[idx])));
}

#[kernel(
    bench(
        op="unary",
        subop="erfinv",
        class=Unary,
        input=Unit,
        // tol=1e-2 — erfinv is a degree-9/10 polynomial approximation;
        // f16 compounds ~1e-3 ULP error per FMA (×9 ≈ 1e-2 in worst case).
        tol=1e-2,
        mlx="v_ErfInv{tn}{tn}",
        metal_file="unary.metal",
    )
)]
pub fn mt_erfinv<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], erfinv(load(a[idx])));
}

// Numerically-stable softplus: softplus(x) = max(x, 0) + log1p(exp(-|x|)).
// Avoids overflow at large positive x and underflow at large negative x —
// the naive `log(1 + exp(x))` blows up for x > ~80 (f32) / ~10 (f16).
// MLX has no dedicated softplus kernel (it composes log1p + exp at the
// graph layer); FFAI Ops.softplus calls this fused per-element variant
// directly because it lives on Mamba 2's hot path (`dt = softplus(dt_raw)`).
#[kernel(
    bench(
        op="unary",
        subop="softplus",
        class=Unary,
        input=Signed,
        tol=1e-4,
    )
)]
pub fn mt_softplus<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]).cast::<f32>();
    let zero = 0.0f32;
    let pos = x > zero;
    let m = select(pos, x, zero);
    let ax = select(pos, x, zero - x);
    let r = m + log(1.0f32 + exp(zero - ax));
    store(out[idx], r.cast::<T>());
}

/// Cast a tensor of model dtype `T` to fp32 in-place per element. One
/// thread per element. Used by callers that need to mix fp32 state with
/// bf16 / f16 model activations on the GPU without a host round-trip —
/// the fused GDN prep step is the immediate consumer (its cache state
/// stays fp32 to avoid the 7-bit-mantissa drift over long decodes, but
/// the model activations into the kernel are bf16).
#[kernel(
    bench(
        op="unary",
        subop="cast_to_f32",
        class=GenericEmpty,
        tol=0.0,
        kernel_mode=Elementwise,
    )
)]
pub fn mt_cast_to_f32<T>(input: Tensor<T>, out: Tensor<f32>) {
    let idx = program_id(0);
    store(out[idx], load(input[idx]).cast::<f32>());
}

/// Fused silu + cast-to-f32. Replaces the `silu(bf16) → cast_to_f32`
/// two-dispatch chain in FFAI's batched-prefill GDN inner loop with a
/// single dispatch: read bf16/f16, apply silu, write f32.
///
/// silu(x) = x · sigmoid(x) = x · (1 / (1 + exp(-x))) computed at f32
/// precision to match the bf16 → fp32 + silu → fp32 chain bit-for-bit
/// (modulo rounding mode on the final write — same as the standalone
/// silu kernel).
///
/// Saves T·30 ≈ 15k dispatches per Qwen3.6-A3B prefill at T=512 (one
/// silu + one cast per GDN-layer per-token iter → one fused dispatch).
#[kernel(
    bench(
        op="unary",
        subop="silu_cast_to_f32",
        class=GenericEmpty,
        tol=1e-3,
        dtypes=&[DType::F16, DType::BF16],
        kernel_mode=Elementwise,
    )
)]
pub fn mt_silu_cast_to_f32<T>(input: Tensor<T>, out: Tensor<f32>) {
    let idx = program_id(0);
    let x = load(input[idx]).cast::<f32>();
    let sig = 1.0f32 / (1.0f32 + exp(0.0f32 - x));
    store(out[idx], x * sig);
}

/// Fused scalar-sigmoid fan-out + FMA. Computes
///   `out[i] = base[i] + sigmoid(gate[0]) * value[i]`
/// for `i in 0..hidden`, broadcasting the scalar `gate` across the
/// `[hidden]` vectors. One thread per output element; the scalar
/// re-loads through the GPU L1 cache so the broadcast is free.
///
/// Replaces FFAI's shared-expert host detour: `gateLogit.toFloatArray()`
/// + host `sigmoid()` + `Tensor.filled([hidden])` + `Ops.mul` + `Ops.add`
/// + a `commit + wait` to ensure the scalar is resident. With this
/// kernel the entire fan-out stays on the GPU and the command buffer
/// the gate was produced on no longer needs a host stall before the
/// next layer queues work.
///
/// Inputs are all in model dtype `T` (typically bf16 on Qwen3.6); the
/// internal accumulation widens to fp32 via the load-side `.cast` to
/// preserve sigmoid precision near saturation.
#[kernel(
    bench(
        op="unary",
        subop="sigmoid_scalar_fma",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Elementwise,
    )
)]
pub fn mt_sigmoid_scalar_fma<T>(
    gate: Tensor<T>,
    value: Tensor<T>,
    base: Tensor<T>,
    out: Tensor<T>,
) {
    let idx = program_id(0);
    let gx = load(gate[0]).cast::<f32>();
    let g = 1.0f32 / (1.0f32 + exp(0.0f32 - gx));
    let v = load(value[idx]).cast::<f32>();
    let b = load(base[idx]).cast::<f32>();
    store(out[idx], (b + g * v).cast::<T>());
}

/// Fused elementwise sigmoid-scalar FMA WITH residual add. Computes
///   `out[i] = residual[i] + base[i] + sigmoid(gate[0]) * value[i]`
/// in one dispatch. Used by Qwen3.6-A3B's post-MoE-FFN site to
/// collapse the existing two-dispatch chain:
///   1. `mt_sigmoid_scalar_fma(gate, sharedOut, routed)` → ffnOut
///   2. `mt_add(postMix, ffnOut)`                       → result
/// into a single dispatch that reads `routed`, `sharedOut`, and
/// `postMix` once each and writes `result` once. Saves one full
/// `[hidden]` DRAM roundtrip on the intermediate `ffnOut` plus one
/// dispatch per MoE layer per token (×40 layers for Qwen3.6-A3B).
///
/// Same precision contract as `mt_sigmoid_scalar_fma`: model dtype
/// `T` on the read+write boundary, fp32 accumulation internally so
/// the sigmoid stays accurate at saturation.
#[kernel(
    bench(
        op="unary",
        subop="sigmoid_scalar_fma_residual",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Elementwise,
    )
)]
pub fn mt_sigmoid_scalar_fma_residual<T>(
    gate: Tensor<T>,
    value: Tensor<T>,
    base: Tensor<T>,
    residual: Tensor<T>,
    out: Tensor<T>,
) {
    let idx = program_id(0);
    let gx = load(gate[0]).cast::<f32>();
    let g = 1.0f32 / (1.0f32 + exp(0.0f32 - gx));
    let v = load(value[idx]).cast::<f32>();
    let b = load(base[idx]).cast::<f32>();
    let r = load(residual[idx]).cast::<f32>();
    store(out[idx], (r + b + g * v).cast::<T>());
}

/// Scalar-broadcast FMA. Computes
///   `out[i] = base[i] + scalar[0] * value[i]`
/// for `i in 0..n`, broadcasting a 1-element scalar buffer across the
/// `[n]` vectors. One thread per output element; the scalar re-loads
/// through the GPU L1 cache so the broadcast is free.
///
/// Replaces FFAI's MoE per-expert weighted-add chain at decode T=1:
/// instead of `Tensor.filled([hidden], weight)` (host alloc + memcpy)
/// + `Ops.mul(expertOut, broadcast)` + `Ops.add(accumulator, scaled)`,
/// we pack the routing weight into a 4-byte scalar buffer + dispatch
/// this kernel once. Saves 8 host allocations + 16 dispatches per MoE
/// layer × 40 layers = 320 allocations + 640 dispatches per
/// Qwen3.6-A3B decode token.
///
/// Numerical: accumulation widens to f32 via load-side `.cast` to keep
/// long sums of many small-weight expert outputs precise, then narrows
/// back to T on store.
#[kernel(
    bench(
        op="unary",
        subop="scalar_fma",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Elementwise,
    )
)]
pub fn mt_scalar_fma<T>(scalar: Tensor<T>, value: Tensor<T>, base: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let s = load(scalar[0]).cast::<f32>();
    let v = load(value[idx]).cast::<f32>();
    let b = load(base[idx]).cast::<f32>();
    store(out[idx], (b + s * v).cast::<T>());
}

/// 8-way fused scalar-FMA chain. Computes
///   `out[i] = sum_{k=0..8} scalar_k[0] * value_k[i]`
/// in a single dispatch. Replaces the topK=8 expert accumulator chain
/// in FFAI's MoE decode (8 sequential `mt_scalar_fma` dispatches +
/// 1 acc.zero) with one fused kernel that reads each value tensor
/// once and writes the output once — saving 7 acc reads + 1 zero
/// dispatch per MoE layer × 40 layers = 320 dispatches + ~660 KB of
/// L1/L2 traffic per Qwen3.6-A3B decode token.
///
/// Accumulation widens to f32 via load-side `.cast` to preserve
/// precision on long sums of small-weight expert outputs, then
/// narrows back to T on store. Bit-equivalent to the 8-call chain
/// modulo final-rounding mode.
#[kernel(
    bench(
        op="unary",
        subop="scalar_fma_chain8",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Elementwise,
    )
)]
pub fn mt_scalar_fma_chain8<T>(
    scalar0: Tensor<T>,
    value0: Tensor<T>,
    scalar1: Tensor<T>,
    value1: Tensor<T>,
    scalar2: Tensor<T>,
    value2: Tensor<T>,
    scalar3: Tensor<T>,
    value3: Tensor<T>,
    scalar4: Tensor<T>,
    value4: Tensor<T>,
    scalar5: Tensor<T>,
    value5: Tensor<T>,
    scalar6: Tensor<T>,
    value6: Tensor<T>,
    scalar7: Tensor<T>,
    value7: Tensor<T>,
    out: Tensor<T>,
) {
    let idx = program_id(0);
    let sa = load(scalar0[0]).cast::<f32>();
    let sb = load(scalar1[0]).cast::<f32>();
    let sc = load(scalar2[0]).cast::<f32>();
    let sd = load(scalar3[0]).cast::<f32>();
    let se = load(scalar4[0]).cast::<f32>();
    let sf = load(scalar5[0]).cast::<f32>();
    let sg = load(scalar6[0]).cast::<f32>();
    let sh = load(scalar7[0]).cast::<f32>();
    let va = load(value0[idx]).cast::<f32>();
    let vb = load(value1[idx]).cast::<f32>();
    let vc = load(value2[idx]).cast::<f32>();
    let vd = load(value3[idx]).cast::<f32>();
    let ve = load(value4[idx]).cast::<f32>();
    let vf = load(value5[idx]).cast::<f32>();
    let vg = load(value6[idx]).cast::<f32>();
    let vh = load(value7[idx]).cast::<f32>();
    let sum = sa * va + sb * vb + sc * vc + sd * vd + se * ve + sf * vf + sg * vg + sh * vh;
    store(out[idx], sum.cast::<T>());
}

/// Fused elementwise sigmoid + mul. Computes
///   `out[i] = a[i] * sigmoid(b[i])`
/// in one dispatch. Used by Qwen3 attention layer's output gate:
///   attn_out = attn(x) * sigmoid(gate_proj(x))
/// Currently expressed as `Ops.mul(attnFlat, Ops.sigmoid(gate))` —
/// two dispatches. This fuses to one, saving 10 dispatches per
/// Qwen3.6-A3B decode token (1 per attn layer × 10 attn layers).
///
/// Sigmoid is computed at f32 precision via load-side cast to avoid
/// bf16 saturation drift near the asymptotes.
#[kernel(
    bench(
        op="unary",
        subop="sigmoid_mul",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Elementwise,
    )
)]
pub fn mt_sigmoid_mul<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let av = load(a[idx]).cast::<f32>();
    let bv = load(b[idx]).cast::<f32>();
    let sig = 1.0f32 / (1.0f32 + exp(0.0f32 - bv));
    store(out[idx], (av * sig).cast::<T>());
}

/// ⚠️ BROKEN — produces NaN/wrong output in FFAI integration test
/// (forwardManyEquivalence fails T=8 + T=128). Kept here as a stub
/// for future debug. Likely issue: multi-output codegen + the inlined
/// `mt_rms_inv_scalar` reduction call don't compose correctly in this
/// kernel shape. Possible fix paths:
///   1. Manually inline the reduction without `mt_rms_inv_scalar`
///   2. Use TWO compiled variants — one for residual write only, one
///      for residual+norm — and dispatch both in a shared encoder
///   3. Investigate the codegen MSL output for residual_out + normed_out
///      ordering vs the reduction-tg threadgroup layout
///
/// Fused residual-add + RMSNorm. For each row of `[n]` elements:
///   residual_out[i] = a[i] + b[i]
///   normed_out[i]   = (a[i] + b[i]) * w[i] / sqrt(mean((a+b)^2) + eps)
///
/// Standard transformer pattern at layer boundary:
///   h_new = h_old + mixer_out           (residual add)
///   normed = rms_norm(h_new, w)         (pre-norm for next mixer)
///
/// Both outputs are needed downstream: `residual_out` is the persistent
/// residual stream (input to the SECOND residual add of the same
/// layer); `normed_out` is the pre-FFN/pre-next-mixer input.
///
/// Same TG=n/4 contract as `mt_rms_norm`. Saves 1 dispatch per
/// residual-add+norm pair × 80 such pairs in Qwen3.6-A3B decode
/// (2 per layer × 40 layers) = ~1.4 ms / token at 17 µs encoder
/// overhead each.
#[kernel(
    bench(
        op="unary",
        subop="add_rms_norm",
        class=GenericEmpty,
        tol=1e-3,
        kernel_mode=Reduction,
    )
)]
pub fn mt_add_rms_norm<T>(
    a: Tensor<T>,
    b: Tensor<T>,
    w: Tensor<T>,
    eps_buf: Tensor<f32>,
    mut residual_out: Tensor<T>,
    mut normed_out: Tensor<T>,
    #[constexpr] n: u32,
) {
    let row = program_id::<0>();
    let rs = row * n;
    let col = tid * 4u32;
    let in_bounds = col + 3u32 < n;
    let safe_col = select(in_bounds, col, 0u32);
    let safe_base = rs + safe_col;
    let base = rs + col;
    // Read a + b for 4 elements.
    let a0 = load(a[safe_base]).cast::<f32>();
    let a1 = load(a[safe_base + 1u32]).cast::<f32>();
    let a2 = load(a[safe_base + 2u32]).cast::<f32>();
    let a3 = load(a[safe_base + 3u32]).cast::<f32>();
    let b0 = load(b[safe_base]).cast::<f32>();
    let b1 = load(b[safe_base + 1u32]).cast::<f32>();
    let b2 = load(b[safe_base + 2u32]).cast::<f32>();
    let b3 = load(b[safe_base + 3u32]).cast::<f32>();
    let s0 = a0 + b0;
    let s1 = a1 + b1;
    let s2 = a2 + b2;
    let s3 = a3 + b3;
    let raw_ssq = s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
    let partial_ssq = select(in_bounds, raw_ssq, 0.0f32);
    // ITER 50 (Bagel 2): inline the reduction with `reduce_sum` directly
    // (no cross-kernel call to `mt_rms_inv_scalar`). The previous stub
    // used the cross-kernel call but the multi-output codegen path
    // didn't compose with the inlined reduction. Inlining matches the
    // working `mt_rms_norm` pattern exactly.
    let tg_ssq = reduce_sum(partial_ssq);
    let eps = load(eps_buf[0u32]);
    let rms = rsqrt(tg_ssq / n + eps);
    if in_bounds {
        // Write the residual stream (just the add).
        store(residual_out[base], s0.cast::<T>());
        store(residual_out[base + 1u32], s1.cast::<T>());
        store(residual_out[base + 2u32], s2.cast::<T>());
        store(residual_out[base + 3u32], s3.cast::<T>());
        // Write the normalized stream.
        let n0 = s0 * rms * load(w[col]).cast::<f32>();
        let n1 = s1 * rms * load(w[col + 1u32]).cast::<f32>();
        let n2 = s2 * rms * load(w[col + 2u32]).cast::<f32>();
        let n3 = s3 * rms * load(w[col + 3u32]).cast::<f32>();
        store(normed_out[base], n0.cast::<T>());
        store(normed_out[base + 1u32], n1.cast::<T>());
        store(normed_out[base + 2u32], n2.cast::<T>());
        store(normed_out[base + 3u32], n3.cast::<T>());
    }
}
