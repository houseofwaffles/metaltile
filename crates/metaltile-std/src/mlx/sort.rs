//! Sort benchmark — #[kernel] DSL vs MLX metal/sort.metal

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="sort",
    subop="sort",
    class=Sort,
    b=1024,
    n=1024,
    tpg=256,
    tol=0.0,
    mlx="c_block_sort_{tn}_{tn}_bn256_tn4",
    metal_file="sort.metal",
)]
#[kernel]
pub fn mt_sort<T>(inp: Tensor<T>, out: Tensor<T>, #[constexpr] n: u32) {
    let block_id = program_id::<0>();
    let t = tid;
    threadgroup_alloc("shared", 1024, T);
    let base = block_id * n;
    threadgroup_store("shared", t * 4u32, load(inp[base + t * 4u32]));
    threadgroup_store("shared", t * 4u32 + 1u32, load(inp[base + t * 4u32 + 1u32]));
    threadgroup_store("shared", t * 4u32 + 2u32, load(inp[base + t * 4u32 + 2u32]));
    threadgroup_store("shared", t * 4u32 + 3u32, load(inp[base + t * 4u32 + 3u32]));
    threadgroup_barrier();
    for _k in range(1u32, 11u32, 1u32) {
        for _jb in range(0u32, _k, 1u32) {
            let flip = _k - _jb - 1u32;
            if flip >= 7u32 {
                threadgroup_barrier();
            }
            for _e in range(0u32, 4u32, 1u32) {
                let gi = t * 4u32 + _e;
                let partner = gi ^ (1u32 << flip);
                if gi < partner {
                    let a = threadgroup_load("shared", gi);
                    let b = threadgroup_load("shared", partner);
                    let dir = (gi >> _k) & 1u32;
                    let want_swap = select(dir == 0u32, a > b, a < b);
                    threadgroup_store("shared", gi, select(want_swap, b, a));
                    threadgroup_store("shared", partner, select(want_swap, a, b));
                }
            }
        }
    }
    threadgroup_barrier();
    store(out[base + t * 4u32], threadgroup_load("shared", t * 4u32));
    store(out[base + t * 4u32 + 1u32], threadgroup_load("shared", t * 4u32 + 1u32));
    store(out[base + t * 4u32 + 2u32], threadgroup_load("shared", t * 4u32 + 2u32));
    store(out[base + t * 4u32 + 3u32], threadgroup_load("shared", t * 4u32 + 3u32));
}
