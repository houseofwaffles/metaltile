//! Random benchmark — #[kernel] DSL vs MLX metal/random.metal

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="random",
    subop="random_hash",
    class=Random,
    n=1048576,
    tpg=1024,
    tol=0.0,
    mlx="rbitsc",
    metal_file="random.metal",
    dtypes=crate::spec::F32_ONLY,
)]
#[kernel]
pub fn mt_random_hash(out: Tensor<u32>, #[constexpr] n: u32) {
    let gid = program_id::<0>();
    let mut s = gid + 1u32;
    s = s ^ (s << 13u32);
    s = s ^ (s >> 17u32);
    s = s ^ (s << 5u32);
    store(out[gid], s);
}
