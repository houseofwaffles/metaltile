//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! LogSumExp benchmark — #[kernel] DSL vs MLX metal/logsumexp.metal

use metaltile::kernel;

#[kernel]
pub fn mt_logsumexp<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let row = program_id::<0>();
    let rs = row * n;
    let re = rs + n;
    let nf = n / (lsize * 4u32);
    let mut lm = neg_infinity();
    let mut nz = 0.0f32;
    for _r in range(0u32, nf, 1u32) {
        let base = rs + (_r * lsize + tid) * 4u32;
        let v0 = load(inp[base]).cast::<f32>();
        let v1 = load(inp[base + 1u32]).cast::<f32>();
        let v2 = load(inp[base + 2u32]).cast::<f32>();
        let v3 = load(inp[base + 3u32]).cast::<f32>();
        let cm = max(max(v0, v1), max(v2, v3));
        let pm = lm;
        let nm = max(pm, cm);
        nz = nz * exp(pm - nm) + exp(v0 - nm) + exp(v1 - nm) + exp(v2 - nm) + exp(v3 - nm);
        lm = nm;
    }
    for _i in range(rs + nf * lsize * 4u32 + tid, re, lsize) {
        let xi = load(inp[_i]).cast::<f32>();
        let pm = lm;
        let nm = max(pm, xi);
        nz = nz * exp(pm - nm) + exp(xi - nm);
        lm = nm;
    }
    let gm = reduce_max(lm);
    let rscl = nz * exp(lm - gm);
    let gs = reduce_sum(rscl);
    if tid == 0 {
        store(out[row], (gm + log(gs)).cast::<T>());
    }
}

/// New-syntax correctness for `mt_logsumexp` (Reduction, one threadgroup per
/// row, tpg=256). Per-row oracle: `max + log(sum(exp(x - max)))`; one output
/// element per row.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_logsumexp;
    use crate::utils::{pack_f32, unpack_f32};

    fn setup(rows: usize, n: usize, dt: DType) -> TestSetup {
        let mut inp = Vec::with_capacity(rows * n);
        let mut expected = Vec::with_capacity(rows);
        for r in 0..rows {
            let row: Vec<f32> =
                (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.1 + r as f32 * 0.05).collect();
            let rd = unpack_f32(&pack_f32(&row, dt), dt);
            let m = rd.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let s: f32 = rd.iter().map(|&x| (x - m).exp()).sum();
            expected.push(m + s.ln());
            inp.extend_from_slice(&row);
        }
        TestSetup::new(mt_logsumexp::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("inp", pack_f32(&inp, dt), dt))
            .input(TestBuffer::zeros("out", rows, dt))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&expected, dt), dt))
            .grid_3d(rows as u32, 1, 1, [256, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 2e-2, 1e-1])]
    fn test_mt_logsumexp(dt: DType) -> TestSetup { setup(4, 1024, dt) }
}

/// New-syntax benchmark for `mt_logsumexp` (vs MLX `metal/logsumexp.metal`).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_logsumexp;

    #[bench(name = "mlx/logsumexp", dtypes = [f32, f16, bf16])]
    fn bench_logsumexp(dt: DType) -> BenchSetup {
        let (rows, n) = (4096usize, 1024usize);
        BenchSetup::new(mt_logsumexp::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("inp", rows * n, dt))
            .buffer(BenchBuffer::zeros("out", rows, dt).output())
            .constexpr("n", n as u32)
            .grid_3d(rows as u32, 1, 1, [256, 1, 1])
            .bytes_moved((rows * n * dt.size_bytes()) as u64)
    }
}
