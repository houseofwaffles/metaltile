//! Unary elementwise benchmark — #[kernel] DSL vs MLX metal/unary.metal

use metaltile::{bench_kernel, kernel};
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

#[bench_kernel(
    op="unary",
    subop="exp",
    class=Unary,
    input=Signed,
    tol=1e-4,
    mlx="v_Exp{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_exp<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], exp(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="log",
    class=Unary,
    input=Positive,
    tol=1e-4,
    mlx="v_Log{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_log<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="sqrt",
    class=Unary,
    input=Positive,
    tol=1e-4,
    mlx="v_Sqrt{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_sqrt<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sqrt(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="rsqrt",
    class=Unary,
    input=Positive,
    tol=1e-4,
    mlx="v_Rsqrt{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_rsqrt<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], rsqrt(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="abs",
    class=Unary,
    input=Signed,
    tol=1e-6,
    mlx="v_Abs{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_abs<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], abs(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="silu",
    class=Unary,
    input=Signed,
    tol=1e-4,
)]
#[kernel]
pub fn mt_silu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], silu(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="gelu",
    class=Unary,
    input=Signed,
    tol=1e-4,
)]
#[kernel]
pub fn mt_gelu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], gelu(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="relu",
    class=Unary,
    input=Signed,
    tol=1e-6,
)]
#[kernel]
pub fn mt_relu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], relu(load(a[idx])));
}

#[bench_kernel(
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
)]
#[kernel]
pub fn mt_cos<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], cos(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="sin",
    class=Unary,
    input=Signed,
    // tol=1e-3 — f16 sin drifts by ~4.9e-4 (see mt_cos comment).
    tol=1e-3,
    mlx="v_Sin{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_sin<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sin(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="ceil",
    class=Unary,
    input=Signed,
    tol=1e-6,
    mlx="v_Ceil{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_ceil<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], ceil(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="floor",
    class=Unary,
    input=Signed,
    tol=1e-6,
    mlx="v_Floor{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_floor<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], floor(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="erf",
    class=Unary,
    input=Signed,
    tol=1e-3,
    mlx="v_Erf{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_erf<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], erf(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="exp2",
    class=Unary,
    input=Signed,
    tol=1e-4,
)]
#[kernel]
pub fn mt_exp2<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], exp2(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="log2",
    class=Unary,
    input=Positive,
    tol=1e-4,
    mlx="v_Log2{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_log2<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log2(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="sign",
    class=Unary,
    input=Signed,
    tol=0.0,
    mlx="v_Sign{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_sign<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sign(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="round",
    class=Unary,
    input=Signed,
    tol=0.0,
    mlx="v_Round{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_round<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], round(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="neg",
    class=Unary,
    input=Signed,
    tol=1e-6,
    mlx="v_Negative{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_neg<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], -load(a[idx]));
}

#[bench_kernel(
    op="unary",
    subop="recip",
    class=Unary,
    input=Positive,
    tol=1e-4,
)]
#[kernel]
pub fn mt_recip<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], 1.0f32.cast::<T>() / load(a[idx]));
}

#[bench_kernel(
    op="unary",
    subop="square",
    class=Unary,
    input=Signed,
    tol=1e-4,
    mlx="v_Square{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_square<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    store(out[idx], x * x);
}

#[bench_kernel(
    op="unary",
    subop="sigmoid",
    class=Unary,
    input=Signed,
    // tol=1e-3 — f16 sigmoid drifts by ~4.9e-4 (exp + reciprocal each
    // pick up ULP-level error, compounds across them).
    tol=1e-3,
    mlx="v_Sigmoid{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_sigmoid<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    // Compute in f32 to match MLX precision (convert back to T at store).
    let x = load(a[idx]).cast::<f32>();
    let result = 1.0f32 / (1.0f32 + exp(-x));
    store(out[idx], result.cast::<T>());
}

#[bench_kernel(
    op="unary",
    subop="log1p",
    class=Unary,
    input=Positive,
    tol=1e-4,
    mlx="v_Log1p{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_log1p<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    store(out[idx], log(1.0f32.cast::<T>() + x));
}

// ─── Transcendental ops shipped as discrete MLX kernels ───────────────
// Every op below has a matching `instantiate_unary_float` in MLX's
// unary.metal and produces a kernel named `v_<Op>{tn}{tn}`.

#[bench_kernel(
    op="unary",
    subop="sinh",
    class=Unary,
    input=Signed,
    tol=1e-4,
    mlx="v_Sinh{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_sinh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sinh(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="cosh",
    class=Unary,
    input=Signed,
    tol=1e-4,
    mlx="v_Cosh{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_cosh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], cosh(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="tan",
    class=Unary,
    input=Signed,
    // tol=1e-3 — f16 tan has poles near π/2 + kπ where values diverge;
    // even on moderate inputs (~1.0) the f16 ULP drift is ~5e-4.
    tol=1e-3,
    mlx="v_Tan{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_tan<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], tan(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="tanh",
    class=Unary,
    input=Signed,
    // tol=1e-3 — f16 tanh compounds exp and div; worst-case drift ~4e-4.
    tol=1e-3,
    mlx="v_Tanh{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_tanh_op<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], tanh(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="asin",
    class=Unary,
    input=Unit,
    // tol=1e-4 — asin has poles at ±1; input clamped to [-1, 1] via
    // the Unit shape generator, so no pole-induced blowup.
    tol=1e-4,
    mlx="v_ArcSin{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_asin<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], asin(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="atan",
    class=Unary,
    input=Signed,
    // tol=1e-3 — f16 atan drifts ~3e-4 on large-magnitude inputs.
    tol=1e-3,
    mlx="v_ArcTan{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_atan<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], atan(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="asinh",
    class=Unary,
    input=Signed,
    tol=1e-4,
    mlx="v_ArcSinh{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_asinh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], asinh(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="acos",
    class=Unary,
    input=Unit,
    // acos has domain [-1, 1]; Unit input clamps to [-1+ε, 1-ε].
    tol=1e-4,
    mlx="v_ArcCos{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_acos<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], acos(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="trunc",
    class=Unary,
    input=Signed,
    // trunc is exact (sets the integer-part bits to zero).
    tol=1e-6,
)]
#[kernel]
pub fn mt_trunc<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], trunc(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="acosh",
    class=Unary,
    input=Positive,
    // input=Positive ensures x ≥ 1e-6, safely inside the domain x ≥ 1.
    tol=1e-4,
    mlx="v_ArcCosh{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_acosh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], acosh(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="atanh",
    class=Unary,
    input=Unit,
    // atanh has poles at ±1; Unit input clamps to [-1+ε, 1-ε].
    tol=1e-4,
    mlx="v_ArcTanh{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_atanh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], atanh(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="expm1",
    class=Unary,
    input=Signed,
    // tol=5e-3 — for large inputs (|x|≈3) two IEEE-754 expm1 impls can diverge
    // by ~1e-4 in f32 and ~4e-3 in bf16 (half-ULP at that magnitude).
    tol=5e-3,
    mlx="v_Expm1{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_expm1<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], expm1(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="log10",
    class=Unary,
    input=Positive,
    tol=1e-4,
    mlx="v_Log10{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
pub fn mt_log10<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log10(load(a[idx])));
}

#[bench_kernel(
    op="unary",
    subop="erfinv",
    class=Unary,
    input=Unit,
    // tol=1e-2 — erfinv is a degree-9/10 polynomial approximation;
    // f16 compounds ~1e-3 ULP error per FMA (×9 ≈ 1e-2 in worst case).
    tol=1e-2,
    mlx="v_ErfInv{tn}{tn}",
    metal_file="unary.metal",
)]
#[kernel]
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
#[bench_kernel(
    op="unary",
    subop="softplus",
    class=Unary,
    input=Signed,
    tol=1e-4,
)]
#[kernel]
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
#[kernel]
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
#[kernel]
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
#[kernel]
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

inventory::submit! {
    BenchSpec {
        op: "unary",
        subop: "sigmoid_scalar_fma",
        kernel_name: "mt_sigmoid_scalar_fma",
        kernel_ir: mt_sigmoid_scalar_fma::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 1e-3,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Elementwise),
    }
}

inventory::submit! {
    BenchSpec {
        op: "unary",
        subop: "cast_to_f32",
        kernel_name: "mt_cast_to_f32",
        kernel_ir: mt_cast_to_f32::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Elementwise),
    }
}

inventory::submit! {
    BenchSpec {
        op: "unary",
        subop: "silu_cast_to_f32",
        kernel_name: "mt_silu_cast_to_f32",
        kernel_ir: mt_silu_cast_to_f32::kernel_ir_for,
        dtypes: &[DType::F16, DType::BF16],
        tol: 1e-3,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Elementwise),
    }
}
