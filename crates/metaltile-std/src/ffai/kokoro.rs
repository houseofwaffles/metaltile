//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Kokoro TTS (StyleTTS2-derived) building blocks: an **LSTM cell** for
//! the bidirectional text/prosody encoders and **AdaIN1d** (adaptive
//! instance norm) for the style-conditioned decoder.
//!
//! ## `lstm_cell` — one LSTM timestep
//!
//! An LSTM is sequential over time (each step reads the previous hidden
//! and cell state), so the recurrence stays on the host: the host loops
//! over timesteps and, for a **bidirectional** LSTM, runs this cell
//! forward over `t = 0..L` and backward over `t = L-1..0` with separate
//! weight sets, concatenating the two hidden streams. The kernel itself
//! is direction-agnostic — it computes the four gates and the new
//! `(h, c)` for a single timestep across the whole batch in one dispatch.
//!
//! PyTorch `nn.LSTM` parametrisation (gate order i, f, g, o):
//!
//!   i = σ(W_ii·x + b_ii + W_hi·h + b_hi)
//!   f = σ(W_if·x + b_if + W_hf·h + b_hf)
//!   g = tanh(W_ig·x + b_ig + W_hg·h + b_hg)
//!   o = σ(W_io·x + b_io + W_ho·h + b_ho)
//!   c' = f·c + i·g       h' = o·tanh(c')
//!
//!   x        [batch, input_size]   T
//!   h_prev   [batch, hidden]       T
//!   c_prev   [batch, hidden]       T
//!   weight_ih [4·hidden, input_size]  T   (rows: i | f | g | o)
//!   weight_hh [4·hidden, hidden]      T   (rows: i | f | g | o)
//!   bias_ih  [4·hidden]            T
//!   bias_hh  [4·hidden]            T
//!   h_out    [batch, hidden]       T   (output)
//!   c_out    [batch, hidden]       T   (output)
//!
//! One thread per `(b, j)` output unit — `grid_1d(batch*hidden, 256)`.
//!
//! ## `adain1d` — adaptive instance norm
//!
//! StyleTTS2 conditions the decoder on a style vector via AdaIN: each
//! channel is instance-normalised over the time axis, then scaled and
//! shifted by per-(batch, channel) `gamma` / `beta` derived from the
//! style embedding. `out[b,c,t] = gamma[b,c]·(x[b,c,t] - μ_{b,c})/
//! sqrt(σ²_{b,c} + eps) + beta[b,c]`. (StyleTTS2's `(1 + γ)` convention is
//! the caller's responsibility — pass `gamma' = 1 + γ`.) One threadgroup
//! per `(b, c)` row, strided over the time axis so any `length` works.

use metaltile::kernel;

#[kernel]
pub fn lstm_cell<T>(
    x: Tensor<T>,
    h_prev: Tensor<T>,
    c_prev: Tensor<T>,
    weight_ih: Tensor<T>,
    weight_hh: Tensor<T>,
    bias_ih: Tensor<T>,
    bias_hh: Tensor<T>,
    mut h_out: Tensor<T>,
    mut c_out: Tensor<T>,
    #[constexpr] batch: u32,
    #[constexpr] input_size: u32,
    #[constexpr] hidden: u32,
) {
    // One thread per (b, j) hidden unit.
    let idx = program_id::<0>();
    let j = idx % hidden;
    let b = idx / hidden;
    let x_base = b * input_size;
    let h_base = b * hidden;
    // Gate rows in the i|f|g|o stack, built additively (row_f = row_i +
    // hidden, …) so the offset math is unambiguous.
    let row_i = j;
    let row_f = row_i + hidden;
    let row_g = row_f + hidden;
    let row_o = row_g + hidden;
    // Per-gate base offsets into the weight matrices, hoisted out of the
    // accumulation loops.
    let w_ih_i = row_i * input_size;
    let w_ih_f = row_f * input_size;
    let w_ih_g = row_g * input_size;
    let w_ih_o = row_o * input_size;
    let w_hh_i = row_i * hidden;
    let w_hh_f = row_f * hidden;
    let w_hh_g = row_g * hidden;
    let w_hh_o = row_o * hidden;
    // Pre-activations seeded with both bias terms.
    let mut pi = load(bias_ih[row_i]).cast::<f32>() + load(bias_hh[row_i]).cast::<f32>();
    let mut pf = load(bias_ih[row_f]).cast::<f32>() + load(bias_hh[row_f]).cast::<f32>();
    let mut pg = load(bias_ih[row_g]).cast::<f32>() + load(bias_hh[row_g]).cast::<f32>();
    let mut po = load(bias_ih[row_o]).cast::<f32>() + load(bias_hh[row_o]).cast::<f32>();
    // Input projection: W_ih · x.
    for ii in range(0u32, input_size, 1u32) {
        let xv = load(x[x_base + ii]).cast::<f32>();
        pi = pi + xv * load(weight_ih[w_ih_i + ii]).cast::<f32>();
        pf = pf + xv * load(weight_ih[w_ih_f + ii]).cast::<f32>();
        pg = pg + xv * load(weight_ih[w_ih_g + ii]).cast::<f32>();
        po = po + xv * load(weight_ih[w_ih_o + ii]).cast::<f32>();
    }
    // Recurrent projection: W_hh · h_prev.
    for kk in range(0u32, hidden, 1u32) {
        let hv = load(h_prev[h_base + kk]).cast::<f32>();
        pi = pi + hv * load(weight_hh[w_hh_i + kk]).cast::<f32>();
        pf = pf + hv * load(weight_hh[w_hh_f + kk]).cast::<f32>();
        pg = pg + hv * load(weight_hh[w_hh_g + kk]).cast::<f32>();
        po = po + hv * load(weight_hh[w_hh_o + kk]).cast::<f32>();
    }
    // Gate activations. sigmoid inlined as 1/(1+exp(-x)) (the established
    // idiom in gated_delta_prep — keeps the emitted MSL self-contained).
    let ig = 1.0f32 / (1.0f32 + exp(0.0f32 - pi));
    let fg = 1.0f32 / (1.0f32 + exp(0.0f32 - pf));
    let gg = tanh(pg);
    let og = 1.0f32 / (1.0f32 + exp(0.0f32 - po));
    let c_new = fg * load(c_prev[idx]).cast::<f32>() + ig * gg;
    let h_new = og * tanh(c_new);
    // Bounds guard: `grid_1d` rounds the grid up to whole threadgroups,
    // so threads with `idx >= batch*hidden` are launched but must not
    // store. With TWO output buffers an unguarded OOB `h_out[idx]` write
    // lands in the adjacent `c_out` allocation and corrupts its valid
    // entries (single-output kernels don't hit this — their OOB store
    // runs off the end of the only buffer harmlessly).
    if idx < batch * hidden {
        store(c_out[idx], c_new.cast::<T>());
        store(h_out[idx], h_new.cast::<T>());
    }
}

#[kernel]
pub fn adain1d<T>(
    x: Tensor<T>,
    gamma: Tensor<T>,
    beta: Tensor<T>,
    mut out: Tensor<T>,
    eps_buf: Tensor<f32>,
    #[constexpr] length: u32,
) {
    // One threadgroup per (batch, channel) row; row index doubles as the
    // gamma/beta index since both are [batch, channels] row-major.
    let row = program_id::<0>();
    let rs = row * length;
    let tpg = n_simd * 32u32;
    // Pass 1: strided sum + sum-of-squares over the time axis. A thread
    // whose stride walks past `length` contributes 0 but still reaches
    // the reductions (Apple simdgroup reductions need all lanes active).
    let mut s = 0.0f32;
    let mut sq = 0.0f32;
    for i in range(tid, length, tpg) {
        let xi = load(x[rs + i]).cast::<f32>();
        s = s + xi;
        sq = sq + xi * xi;
    }
    let tot = reduce_sum(s);
    let tot_sq = reduce_sum(sq);
    let mean = tot / length;
    let var = tot_sq / length - mean * mean;
    let eps = load(eps_buf[0]);
    let inv = rsqrt(var + eps);
    let g = load(gamma[row]).cast::<f32>();
    let bta = load(beta[row]).cast::<f32>();
    // Pass 2: strided affine-normalised store.
    for i in range(tid, length, tpg) {
        let xi = load(x[rs + i]).cast::<f32>();
        store(out[rs + i], ((xi - mean) * inv * g + bta).cast::<T>());
    }
}

pub mod kernel_tests {
    use metaltile::{core::ir::Kernel, test::*, test_kernel};

    use super::{adain1d, lstm_cell};
    use crate::utils::{pack_f32, unpack_f32};

    fn ramp(n: usize, period: usize, amp: f32, start: f32) -> Vec<f32> {
        (0..n).map(|i| ((i % period) as f32 / period as f32 - 0.5) * amp + start).collect()
    }

    fn sigmoid(v: f32) -> f32 { 1.0 / (1.0 + (-v).exp()) }

    // ── LSTM cell oracle (gate order i, f, g, o) ──
    #[allow(clippy::too_many_arguments)]
    fn naive_lstm_cell(
        x: &[f32],
        h_prev: &[f32],
        c_prev: &[f32],
        weight_ih: &[f32],
        weight_hh: &[f32],
        bias_ih: &[f32],
        bias_hh: &[f32],
        batch: usize,
        input_size: usize,
        hidden: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut h_out = vec![0.0f32; batch * hidden];
        let mut c_out = vec![0.0f32; batch * hidden];
        for b in 0..batch {
            for j in 0..hidden {
                let mut pre = [0.0f32; 4];
                for (g, p) in pre.iter_mut().enumerate() {
                    let row = g * hidden + j;
                    let mut acc = bias_ih[row] + bias_hh[row];
                    for ii in 0..input_size {
                        acc += x[b * input_size + ii] * weight_ih[row * input_size + ii];
                    }
                    for kk in 0..hidden {
                        acc += h_prev[b * hidden + kk] * weight_hh[row * hidden + kk];
                    }
                    *p = acc;
                }
                let ig = sigmoid(pre[0]);
                let fg = sigmoid(pre[1]);
                let gg = pre[2].tanh();
                let og = sigmoid(pre[3]);
                let c_new = fg * c_prev[b * hidden + j] + ig * gg;
                let h_new = og * c_new.tanh();
                c_out[b * hidden + j] = c_new;
                h_out[b * hidden + j] = h_new;
            }
        }
        (h_out, c_out)
    }

    fn lstm_setup(
        kernel: Kernel,
        batch: usize,
        input_size: usize,
        hidden: usize,
        dt: DType,
    ) -> TestSetup {
        let x_f = ramp(batch * input_size, 13, 1.2, 0.0);
        let h_f = ramp(batch * hidden, 17, 0.8, 0.1);
        let c_f = ramp(batch * hidden, 11, 0.6, -0.1);
        let w_ih = ramp(4 * hidden * input_size, 23, 0.3, 0.0);
        let w_hh = ramp(4 * hidden * hidden, 29, 0.25, 0.0);
        let b_ih = ramp(4 * hidden, 7, 0.2, 0.0);
        let b_hh = ramp(4 * hidden, 5, 0.15, 0.0);
        let r = |v: &[f32]| unpack_f32(&pack_f32(v, dt), dt);
        let (h_exp, c_exp) = naive_lstm_cell(
            &r(&x_f),
            &r(&h_f),
            &r(&c_f),
            &r(&w_ih),
            &r(&w_hh),
            &r(&b_ih),
            &r(&b_hh),
            batch,
            input_size,
            hidden,
        );
        TestSetup::new(kernel)
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("h_prev", pack_f32(&h_f, dt), dt))
            .input(TestBuffer::from_vec("c_prev", pack_f32(&c_f, dt), dt))
            .input(TestBuffer::from_vec("weight_ih", pack_f32(&w_ih, dt), dt))
            .input(TestBuffer::from_vec("weight_hh", pack_f32(&w_hh, dt), dt))
            .input(TestBuffer::from_vec("bias_ih", pack_f32(&b_ih, dt), dt))
            .input(TestBuffer::from_vec("bias_hh", pack_f32(&b_hh, dt), dt))
            .input(TestBuffer::zeros("h_out", batch * hidden, dt))
            .input(TestBuffer::zeros("c_out", batch * hidden, dt))
            .constexpr("batch", batch as u32)
            .constexpr("input_size", input_size as u32)
            .constexpr("hidden", hidden as u32)
            .expect(TestBuffer::from_vec("h_out", pack_f32(&h_exp, dt), dt))
            .expect(TestBuffer::from_vec("c_out", pack_f32(&c_exp, dt), dt))
            .grid_1d(batch * hidden, 256)
    }

    // Kokoro text-encoder BiLSTM cell shape (per direction).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 2e-2, 1e-1])]
    fn test_lstm_cell(dt: DType) -> TestSetup {
        lstm_setup(lstm_cell::kernel_ir_for(dt), 2, 128, 64, dt)
    }

    // Equal input/hidden size (prosody predictor LSTM).
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [2e-3, 2e-2, 1e-1])]
    fn test_lstm_cell_square(dt: DType) -> TestSetup {
        lstm_setup(lstm_cell::kernel_ir_for(dt), 3, 96, 96, dt)
    }

    // ── AdaIN1d oracle: per-(b,c) instance norm over time, then scale/shift ──
    fn naive_adain(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        rows: usize,
        length: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * length];
        for r in 0..rows {
            let row = &x[r * length..(r + 1) * length];
            let mean: f32 = row.iter().sum::<f32>() / length as f32;
            let var: f32 = row.iter().map(|&v| v * v).sum::<f32>() / length as f32 - mean * mean;
            let inv = 1.0 / (var + eps).sqrt();
            for (t, &xi) in row.iter().enumerate() {
                out[r * length + t] = (xi - mean) * inv * gamma[r] + beta[r];
            }
        }
        out
    }

    fn adain_setup(rows: usize, length: usize, dt: DType) -> TestSetup {
        let eps = 1e-5f32;
        let x_f = ramp(rows * length, 19, 3.0, 0.0);
        let gamma_f = ramp(rows, 7, 1.0, 1.0);
        let beta_f = ramp(rows, 5, 0.5, 0.0);
        let x = unpack_f32(&pack_f32(&x_f, dt), dt);
        let gamma = unpack_f32(&pack_f32(&gamma_f, dt), dt);
        let beta = unpack_f32(&pack_f32(&beta_f, dt), dt);
        let expected = naive_adain(&x, &gamma, &beta, rows, length, eps);
        TestSetup::new(adain1d::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("x", pack_f32(&x_f, dt), dt))
            .input(TestBuffer::from_vec("gamma", pack_f32(&gamma_f, dt), dt))
            .input(TestBuffer::from_vec("beta", pack_f32(&beta_f, dt), dt))
            .input(TestBuffer::zeros("out", rows * length, dt))
            .input(TestBuffer::from_vec("eps_buf", eps.to_le_bytes().to_vec(), DType::F32))
            .constexpr("length", length as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [1024, 1, 1])
    }

    // (batch*channels) rows; non-128-aligned length to exercise the strided path.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 1e-2, 5e-2])]
    fn test_adain1d(dt: DType) -> TestSetup { adain_setup(8, 300, dt) }
}

/// New-syntax benches: a prosody-LSTM cell and an AdaIN decoder block.
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{adain1d, lstm_cell};

    #[bench(name = "ffai/kokoro/lstm_cell", dtypes = [f32, f16, bf16])]
    fn bench_lstm_cell(dt: DType) -> BenchSetup {
        // Kokoro-class BiLSTM: input 512, hidden 256, batch 8.
        let (batch, input_size, hidden) = (8usize, 512usize, 256usize);
        let n_out = batch * hidden;
        let bytes = (batch * input_size
            + 2 * batch * hidden
            + 4 * hidden * input_size
            + 4 * hidden * hidden)
            * dt.size_bytes();
        BenchSetup::new(lstm_cell::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("x", batch * input_size, dt))
            .buffer(BenchBuffer::random("h_prev", batch * hidden, dt))
            .buffer(BenchBuffer::random("c_prev", batch * hidden, dt))
            .buffer(BenchBuffer::random("weight_ih", 4 * hidden * input_size, dt))
            .buffer(BenchBuffer::random("weight_hh", 4 * hidden * hidden, dt))
            .buffer(BenchBuffer::random("bias_ih", 4 * hidden, dt))
            .buffer(BenchBuffer::random("bias_hh", 4 * hidden, dt))
            .buffer(BenchBuffer::zeros("h_out", n_out, dt).output())
            .buffer(BenchBuffer::zeros("c_out", n_out, dt).output())
            .constexpr("batch", batch as u32)
            .constexpr("input_size", input_size as u32)
            .constexpr("hidden", hidden as u32)
            .grid_1d(n_out, 256)
            .bytes_moved(bytes as u64)
    }

    #[bench(name = "ffai/kokoro/adain1d", dtypes = [f32, f16, bf16])]
    fn bench_adain1d(dt: DType) -> BenchSetup {
        // Decoder AdaIN: 512 channels, time length 1024, batch 4.
        let (batch, channels, length) = (4usize, 512usize, 1024usize);
        let rows = batch * channels;
        BenchSetup::new(adain1d::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("x", rows * length, dt))
            .buffer(BenchBuffer::random("gamma", rows, dt))
            .buffer(BenchBuffer::random("beta", rows, dt))
            .buffer(BenchBuffer::zeros("out", rows * length, dt).output())
            .buffer(BenchBuffer::from_vec("eps_buf", 1e-5f32.to_le_bytes().to_vec(), DType::F32))
            .constexpr("length", length as u32)
            .grid_3d(rows as u32, 1, 1, [1024, 1, 1])
            .bytes_moved((2 * rows * length * dt.size_bytes()) as u64)
    }
}
