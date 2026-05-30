//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Random benchmark — #[kernel] DSL vs MLX metal/random.metal

use metaltile::kernel;

#[kernel]
pub fn mt_random_hash(out: Tensor<u32>, #[constexpr] n: u32) {
    let gid = program_id::<0>();
    let mut s = gid + 1u32;
    s = s ^ (s << 13u32);
    s = s ^ (s >> 17u32);
    s = s ^ (s << 5u32);
    store(out[gid], s);
}

/// New-syntax correctness for `mt_random_hash` — a deterministic xorshift of
/// `gid + 1`, so the oracle replays the exact bit-twiddle. The u32 output is
/// packed raw (not via `pack_f32`'s value-cast) so full-range values compare
/// exactly: both sides round through f32 identically. Non-generic kernel — the
/// `dt` argument is ignored.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::mt_random_hash;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[test_kernel(dtypes = [f32], tol = 0.5)]
    fn test_mt_random_hash(_dt: DType) -> TestSetup {
        let n = 1024usize;
        let expected: Vec<u32> = (0..n)
            .map(|gid| {
                let mut s = (gid as u32).wrapping_add(1);
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                s
            })
            .collect();
        TestSetup::new(mt_random_hash::kernel_ir_for())
            .input(TestBuffer::zeros("out", n, DType::U32))
            .constexpr("n", n as u32)
            .expect(TestBuffer::from_vec("out", u32_bytes(&expected), DType::U32))
            .grid_1d(n, 256)
    }
}

/// New-syntax benchmark for `mt_random_hash` (vs MLX `metal/random.metal`).
pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_random_hash;

    #[bench(name = "mlx/random/random_hash", dtypes = [f32])]
    fn bench_random_hash(_dt: DType) -> BenchSetup {
        let n = 64 * 1024 * 1024usize;
        BenchSetup::new(mt_random_hash::kernel_ir_for())
            .buffer(BenchBuffer::zeros("out", n, DType::U32).output())
            .constexpr("n", n as u32)
            .grid_1d(n, 256)
            .bytes_moved((n * 4) as u64)
    }
}
