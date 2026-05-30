//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Unary elementwise benchmark — #[kernel] DSL vs MLX metal/unary.metal

use metaltile::kernel;

#[kernel]
pub fn mt_exp<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], exp(load(a[idx])));
}

#[kernel]
pub fn mt_log<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log(load(a[idx])));
}

#[kernel]
pub fn mt_sqrt<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sqrt(load(a[idx])));
}

#[kernel]
pub fn mt_rsqrt<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], rsqrt(load(a[idx])));
}

#[kernel]
pub fn mt_abs<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], abs(load(a[idx])));
}

#[kernel]
pub fn mt_silu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], silu(load(a[idx])));
}

#[kernel]
pub fn mt_gelu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], gelu(load(a[idx])));
}

#[kernel]
pub fn mt_relu<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], relu(load(a[idx])));
}

#[kernel]
pub fn mt_cos<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], cos(load(a[idx])));
}

#[kernel]
pub fn mt_sin<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sin(load(a[idx])));
}

#[kernel]
pub fn mt_ceil<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], ceil(load(a[idx])));
}

#[kernel]
pub fn mt_floor<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], floor(load(a[idx])));
}

#[kernel]
pub fn mt_erf<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], erf(load(a[idx])));
}

#[kernel]
pub fn mt_exp2<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], exp2(load(a[idx])));
}

#[kernel]
pub fn mt_log2<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log2(load(a[idx])));
}

#[kernel]
pub fn mt_sign<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sign(load(a[idx])));
}

#[kernel]
pub fn mt_round<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], round(load(a[idx])));
}

#[kernel]
pub fn mt_neg<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], -load(a[idx]));
}

#[kernel]
pub fn mt_recip<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], 1.0f32.cast::<T>() / load(a[idx]));
}

#[kernel]
pub fn mt_square<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    store(out[idx], x * x);
}

#[kernel]
pub fn mt_sigmoid<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    // Compute in f32 to match MLX precision (convert back to T at store).
    let x = load(a[idx]).cast::<f32>();
    let result = 1.0f32 / (1.0f32 + exp(-x));
    store(out[idx], result.cast::<T>());
}

#[kernel]
pub fn mt_log1p<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    let x = load(a[idx]);
    store(out[idx], log(1.0f32.cast::<T>() + x));
}

// ─── Transcendental ops shipped as discrete MLX kernels ───────────────
// Every op below has a matching `instantiate_unary_float` in MLX's
// unary.metal and produces a kernel named `v_<Op>{tn}{tn}`.

#[kernel]
pub fn mt_sinh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], sinh(load(a[idx])));
}

#[kernel]
pub fn mt_cosh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], cosh(load(a[idx])));
}

#[kernel]
pub fn mt_tan<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], tan(load(a[idx])));
}

#[kernel]
pub fn mt_tanh_op<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], tanh(load(a[idx])));
}

#[kernel]
pub fn mt_asin<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], asin(load(a[idx])));
}

#[kernel]
pub fn mt_atan<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], atan(load(a[idx])));
}

#[kernel]
pub fn mt_asinh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], asinh(load(a[idx])));
}

#[kernel]
pub fn mt_acos<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], acos(load(a[idx])));
}

#[kernel]
pub fn mt_trunc<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], trunc(load(a[idx])));
}

#[kernel]
pub fn mt_acosh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], acosh(load(a[idx])));
}

#[kernel]
pub fn mt_atanh<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], atanh(load(a[idx])));
}

#[kernel]
pub fn mt_expm1<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], expm1(load(a[idx])));
}

#[kernel]
pub fn mt_log10<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], log10(load(a[idx])));
}

#[kernel]
pub fn mt_erfinv<T>(a: Tensor<T>, out: Tensor<T>) {
    let idx = program_id(0);
    store(out[idx], erfinv(load(a[idx])));
}

/// New-syntax correctness for the simple unary elementwise ops.
///
/// Each test rounds its input to `dt` (so the oracle sees what the GPU loads),
/// computes the f32 reference, and compares per-dtype. Exact ops (abs, relu,
/// neg, sign, ceil, floor, round, trunc) hold at 1e-6; transcendentals use a
/// generous-but-bounded per-dtype band that still catches an empty-body or
/// wrong-formula kernel. `erf`/`gelu`/`erfinv` are bench-only — there's no
/// std f32 oracle for them (the legacy test didn't cover them either).
pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::*;
    use crate::utils::{pack_f32, unpack_f32};

    const N: usize = 512;

    fn signed() -> Vec<f32> { (0..N).map(|i| (i % 17) as f32 * 0.35 - 3.0).collect() }
    // Small range that stays clear of the tan poles at ±π/2.
    fn small() -> Vec<f32> { (0..N).map(|i| (i % 17) as f32 * 0.12 - 1.0).collect() }
    fn positive() -> Vec<f32> { (0..N).map(|i| (i % 17) as f32 * 0.1 + 0.1).collect() }
    fn unit() -> Vec<f32> { (0..N).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect() }
    fn ge_one() -> Vec<f32> { (0..N).map(|i| (i % 17) as f32 * 0.12 + 1.0).collect() }
    // Offset to dodge .5 boundaries where round-half-even vs -away disagree.
    fn round_safe() -> Vec<f32> { (0..N).map(|i| (i % 23) as f32 * 0.137 - 1.5).collect() }

    fn un<F: Fn(f32) -> f32>(kernel: Kernel, a: &[f32], op: F, dt: DType) -> TestSetup {
        let a_dt = unpack_f32(&pack_f32(a, dt), dt);
        let expected: Vec<f32> = a_dt.iter().map(|&x| op(x)).collect();
        TestSetup::new(kernel)
            .input(TestBuffer::from_vec("a", pack_f32(a, dt), dt))
            .input(TestBuffer::zeros("out", a.len(), dt))
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_1d(a.len(), 256)
    }

    // ── Exact ops (bit-exact in every dtype) ──────────────────────────────
    macro_rules! exact_test {
        ($name:ident, $kernel:ident, $input:ident, $op:expr) => {
            #[test_kernel(dtypes = [f32, f16, bf16], tol = 1e-6)]
            fn $name(dt: DType) -> TestSetup { un($kernel::kernel_ir_for(dt), &$input(), $op, dt) }
        };
    }
    exact_test!(test_unary_abs, mt_abs, signed, f32::abs);
    exact_test!(test_unary_relu, mt_relu, signed, |x| x.max(0.0));
    exact_test!(test_unary_neg, mt_neg, signed, |x| -x);
    exact_test!(test_unary_ceil, mt_ceil, round_safe, f32::ceil);
    exact_test!(test_unary_floor, mt_floor, round_safe, f32::floor);
    exact_test!(test_unary_round, mt_round, round_safe, f32::round);
    exact_test!(test_unary_trunc, mt_trunc, round_safe, f32::trunc);
    exact_test!(test_unary_sign, mt_sign, signed, |x| if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    });

    // ── Transcendental ops (per-dtype band) ───────────────────────────────
    macro_rules! trans_test {
        ($name:ident, $kernel:ident, $input:ident, $op:expr) => {
            #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 2e-2, 1e-1])]
            fn $name(dt: DType) -> TestSetup { un($kernel::kernel_ir_for(dt), &$input(), $op, dt) }
        };
    }
    trans_test!(test_unary_exp, mt_exp, signed, f32::exp);
    trans_test!(test_unary_exp2, mt_exp2, signed, f32::exp2);
    trans_test!(test_unary_expm1, mt_expm1, signed, f32::exp_m1);
    trans_test!(test_unary_log, mt_log, positive, f32::ln);
    trans_test!(test_unary_log2, mt_log2, positive, f32::log2);
    trans_test!(test_unary_log10, mt_log10, positive, f32::log10);
    trans_test!(test_unary_log1p, mt_log1p, positive, f32::ln_1p);
    trans_test!(test_unary_sqrt, mt_sqrt, positive, f32::sqrt);
    trans_test!(test_unary_rsqrt, mt_rsqrt, positive, |x| 1.0 / x.sqrt());
    trans_test!(test_unary_recip, mt_recip, positive, |x| 1.0 / x);
    trans_test!(test_unary_square, mt_square, signed, |x| x * x);
    trans_test!(test_unary_sin, mt_sin, signed, f32::sin);
    trans_test!(test_unary_cos, mt_cos, signed, f32::cos);
    trans_test!(test_unary_tan, mt_tan, small, f32::tan);
    trans_test!(test_unary_asin, mt_asin, unit, f32::asin);
    trans_test!(test_unary_acos, mt_acos, unit, f32::acos);
    trans_test!(test_unary_atan, mt_atan, signed, f32::atan);
    trans_test!(test_unary_sinh, mt_sinh, signed, f32::sinh);
    trans_test!(test_unary_cosh, mt_cosh, signed, f32::cosh);
    trans_test!(test_unary_tanh, mt_tanh_op, signed, f32::tanh);
    trans_test!(test_unary_asinh, mt_asinh, signed, f32::asinh);
    trans_test!(test_unary_acosh, mt_acosh, ge_one, f32::acosh);
    trans_test!(test_unary_atanh, mt_atanh, unit, f32::atanh);
    trans_test!(test_unary_silu, mt_silu, signed, |x| x / (1.0 + (-x).exp()));
    trans_test!(test_unary_sigmoid, mt_sigmoid, signed, |x| 1.0 / (1.0 + (-x).exp()));
    trans_test!(test_unary_softplus, mt_softplus, signed, |x| x.max(0.0)
        + (1.0 + (-x.abs()).exp()).ln());
}

/// New-syntax benchmarks for the simple unary elementwise ops (vs MLX
/// `metal/unary.metal`). All read `a` once and write `out` once.
pub mod kernel_benches {
    use metaltile::{bench, core::ir::Kernel, test::*};

    use super::*;

    fn ub(kernel: Kernel, dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;
        BenchSetup::new(kernel)
            .buffer(BenchBuffer::random("a", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .grid_1d(n, 256)
            .bytes_moved((2 * n * dt.size_bytes()) as u64)
    }

    macro_rules! ubench {
        ($name:ident, $full:literal, $kernel:ident) => {
            #[bench(name = $full, dtypes = [f32, f16, bf16])]
            fn $name(dt: DType) -> BenchSetup { ub($kernel::kernel_ir_for(dt), dt) }
        };
    }
    ubench!(bench_exp, "mlx/unary/exp", mt_exp);
    ubench!(bench_exp2, "mlx/unary/exp2", mt_exp2);
    ubench!(bench_expm1, "mlx/unary/expm1", mt_expm1);
    ubench!(bench_log, "mlx/unary/log", mt_log);
    ubench!(bench_log2, "mlx/unary/log2", mt_log2);
    ubench!(bench_log10, "mlx/unary/log10", mt_log10);
    ubench!(bench_log1p, "mlx/unary/log1p", mt_log1p);
    ubench!(bench_sqrt, "mlx/unary/sqrt", mt_sqrt);
    ubench!(bench_rsqrt, "mlx/unary/rsqrt", mt_rsqrt);
    ubench!(bench_recip, "mlx/unary/recip", mt_recip);
    ubench!(bench_square, "mlx/unary/square", mt_square);
    ubench!(bench_abs, "mlx/unary/abs", mt_abs);
    ubench!(bench_neg, "mlx/unary/neg", mt_neg);
    ubench!(bench_sign, "mlx/unary/sign", mt_sign);
    ubench!(bench_relu, "mlx/unary/relu", mt_relu);
    ubench!(bench_ceil, "mlx/unary/ceil", mt_ceil);
    ubench!(bench_floor, "mlx/unary/floor", mt_floor);
    ubench!(bench_round, "mlx/unary/round", mt_round);
    ubench!(bench_trunc, "mlx/unary/trunc", mt_trunc);
    ubench!(bench_sin, "mlx/unary/sin", mt_sin);
    ubench!(bench_cos, "mlx/unary/cos", mt_cos);
    ubench!(bench_tan, "mlx/unary/tan", mt_tan);
    ubench!(bench_asin, "mlx/unary/asin", mt_asin);
    ubench!(bench_acos, "mlx/unary/acos", mt_acos);
    ubench!(bench_atan, "mlx/unary/atan", mt_atan);
    ubench!(bench_sinh, "mlx/unary/sinh", mt_sinh);
    ubench!(bench_cosh, "mlx/unary/cosh", mt_cosh);
    ubench!(bench_tanh, "mlx/unary/tanh", mt_tanh_op);
    ubench!(bench_asinh, "mlx/unary/asinh", mt_asinh);
    ubench!(bench_acosh, "mlx/unary/acosh", mt_acosh);
    ubench!(bench_atanh, "mlx/unary/atanh", mt_atanh);
    ubench!(bench_silu, "mlx/unary/silu", mt_silu);
    ubench!(bench_gelu, "mlx/unary/gelu", mt_gelu);
    ubench!(bench_sigmoid, "mlx/unary/sigmoid", mt_sigmoid);
    ubench!(bench_softplus, "mlx/unary/softplus", mt_softplus);
    ubench!(bench_erf, "mlx/unary/erf", mt_erf);
    ubench!(bench_erfinv, "mlx/unary/erfinv", mt_erfinv);

    // ── Fused / cast variants ─────────────────────────────────────────────
    // These have signatures the `ub` helper can't cover (f32 output, scalar
    // broadcast buffers, multi-input FMA chains, or a reduction). One thread
    // per output element (Elementwise) unless noted.

    // cast / silu-cast: read model dtype `T`, write f32.
    fn cast_b(kernel: Kernel, dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;
        BenchSetup::new(kernel)
            .buffer(BenchBuffer::random("input", n, dt))
            .buffer(BenchBuffer::zeros("out", n, DType::F32).output())
            .grid_1d(n, 256)
            .bytes_moved((n * (dt.size_bytes() + DType::F32.size_bytes())) as u64)
    }

    #[bench(name = "mlx/unary/cast_to_f32", dtypes = [f32, f16, bf16])]
    fn bench_cast_to_f32(dt: DType) -> BenchSetup { cast_b(mt_cast_to_f32::kernel_ir_for(dt), dt) }
    #[bench(name = "mlx/unary/silu_cast_to_f32", dtypes = [f16, bf16])]
    fn bench_silu_cast_to_f32(dt: DType) -> BenchSetup {
        cast_b(mt_silu_cast_to_f32::kernel_ir_for(dt), dt)
    }

    // sigmoid_mul: out[i] = a[i] * sigmoid(b[i]). Two full-length inputs.
    #[bench(name = "mlx/unary/sigmoid_mul", dtypes = [f32, f16, bf16])]
    fn bench_sigmoid_mul(dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;
        BenchSetup::new(mt_sigmoid_mul::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("a", n, dt))
            .buffer(BenchBuffer::random("b", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .grid_1d(n, 256)
            .bytes_moved((3 * n * dt.size_bytes()) as u64)
    }

    // scalar_fma: out[i] = base[i] + scalar[0] * value[i]. `scalar` is a
    // 1-element broadcast buffer; `value` + `base` are full-length.
    #[bench(name = "mlx/unary/scalar_fma", dtypes = [f32, f16, bf16])]
    fn bench_scalar_fma(dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;
        BenchSetup::new(mt_scalar_fma::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("scalar", 1, dt))
            .buffer(BenchBuffer::random("value", n, dt))
            .buffer(BenchBuffer::random("base", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .grid_1d(n, 256)
            .bytes_moved((3 * n * dt.size_bytes()) as u64)
    }

    // sigmoid_scalar_fma: out[i] = base[i] + sigmoid(gate[0]) * value[i].
    #[bench(name = "mlx/unary/sigmoid_scalar_fma", dtypes = [f32, f16, bf16])]
    fn bench_sigmoid_scalar_fma(dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;
        BenchSetup::new(mt_sigmoid_scalar_fma::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("gate", 1, dt))
            .buffer(BenchBuffer::random("value", n, dt))
            .buffer(BenchBuffer::random("base", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .grid_1d(n, 256)
            .bytes_moved((3 * n * dt.size_bytes()) as u64)
    }

    // sigmoid_scalar_fma_residual: out = residual + base + sigmoid(gate[0])·value.
    #[bench(name = "mlx/unary/sigmoid_scalar_fma_residual", dtypes = [f32, f16, bf16])]
    fn bench_sigmoid_scalar_fma_residual(dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;
        BenchSetup::new(mt_sigmoid_scalar_fma_residual::kernel_ir_for(dt))
            .buffer(BenchBuffer::random("gate", 1, dt))
            .buffer(BenchBuffer::random("value", n, dt))
            .buffer(BenchBuffer::random("base", n, dt))
            .buffer(BenchBuffer::random("residual", n, dt))
            .buffer(BenchBuffer::zeros("out", n, dt).output())
            .grid_1d(n, 256)
            .bytes_moved((4 * n * dt.size_bytes()) as u64)
    }

    // scalar_fma_chain8: out[i] = sum_{k=0..8} scalar_k[0] * value_k[i].
    // Eight 1-element scalar buffers + eight full-length value buffers.
    #[bench(name = "mlx/unary/scalar_fma_chain8", dtypes = [f32, f16, bf16])]
    fn bench_scalar_fma_chain8(dt: DType) -> BenchSetup {
        let n = 16 * 1024 * 1024usize;
        let mut s = BenchSetup::new(mt_scalar_fma_chain8::kernel_ir_for(dt));
        for k in 0..8 {
            s = s
                .buffer(BenchBuffer::random(&format!("scalar{k}"), 1, dt))
                .buffer(BenchBuffer::random(&format!("value{k}"), n, dt));
        }
        s.buffer(BenchBuffer::zeros("out", n, dt).output())
            .grid_1d(n, 256)
            .bytes_moved((9 * n * dt.size_bytes()) as u64)
    }

    // add_rms_norm: Reduction, one threadgroup per row, N = tpg*4. Fused
    // residual-add (a+b) + RMSNorm with two outputs (residual + normed).
    #[bench(name = "mlx/unary/add_rms_norm", dtypes = [f32, f16, bf16])]
    fn bench_add_rms_norm(dt: DType) -> BenchSetup {
        let (rows, n) = (4096usize, 4096usize);
        BenchSetup::new(mt_add_rms_norm::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("a", rows * n, dt))
            .buffer(BenchBuffer::random("b", rows * n, dt))
            .buffer(BenchBuffer::random("w", n, dt))
            .buffer(BenchBuffer::from_vec("eps_buf", 1e-5f32.to_le_bytes().to_vec(), DType::F32))
            .buffer(BenchBuffer::zeros("residual_out", rows * n, dt).output())
            .buffer(BenchBuffer::zeros("normed_out", rows * n, dt).output())
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [(n / 4) as u32, 1, 1])
            .bytes_moved((4 * rows * n * dt.size_bytes()) as u64)
    }
}

// Numerically-stable softplus: softplus(x) = max(x, 0) + log1p(exp(-|x|)).
// Avoids overflow at large positive x and underflow at large negative x —
// the naive `log(1 + exp(x))` blows up for x > ~80 (f32) / ~10 (f16).
// MLX has no dedicated softplus kernel (it composes log1p + exp at the
// graph layer); FFAI Ops.softplus calls this fused per-element variant
// directly because it lives on Mamba 2's hot path (`dt = softplus(dt_raw)`).
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
#[kernel]
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
#[kernel]
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
#[kernel]
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
#[kernel]
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
#[kernel]
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
