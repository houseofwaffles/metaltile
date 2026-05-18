//! Scan benchmark — #[kernel] DSL vs MLX metal/scan.metal

use metaltile::{bench_kernel, kernel};
static SCAN_SHAPES: &[(usize, usize)] = &[(1_024, 4_096)];

#[bench_kernel(
    op="scan",
    subop="scan",
    class=Scan,
    shapes=&SCAN_SHAPES,
    tpg=256,
    tol=1e-3,
    mlx="contig_scan_inclusive_sum_float32_float32",
    metal_file="scan.metal",
    dtypes=crate::spec::F32_ONLY,
)]
#[kernel]
pub fn mt_scan_f32(inp: Tensor<f32>, out: Tensor<f32>, #[constexpr] n: u32) {
    let row = program_id::<1>();
    let lid = tid;
    let lane = simd_lane;
    let sg = simd_id;
    let ns = n_simd;
    let row_off = row * n;
    threadgroup_alloc("sgs", 9);
    if lid == 0 {
        threadgroup_store("sgs", ns, 0);
    }
    threadgroup_barrier();
    let zero_f = threadgroup_load("sgs", ns);
    let chunk = lsize * 4u32;
    let n_iters = (n + chunk - 1u32) / chunk;
    for _r in range(0, n_iters, 1) {
        let base = _r * chunk + lid * 4u32;
        let v0 = select(base < n, load(inp[row_off + base]), zero_f);
        let v1 = select(base + 1u32 < n, load(inp[row_off + base + 1u32]), zero_f);
        let v2 = select(base + 2u32 < n, load(inp[row_off + base + 2u32]), zero_f);
        let v3 = select(base + 3u32 < n, load(inp[row_off + base + 3u32]), zero_f);
        let s1 = v0 + v1;
        let s2 = s1 + v2;
        let s3 = s2 + v3;
        let thread_excl = simd_scan_exclusive(s3);
        if lane == 31 {
            threadgroup_store("sgs", sg, thread_excl + s3);
        }
        threadgroup_barrier();
        if sg == 0 {
            let wt = select(lane < ns, threadgroup_load("sgs", lane), zero_f);
            let wt_excl = simd_scan_exclusive(wt);
            if lane < ns {
                threadgroup_store("sgs", lane, wt_excl);
            }
        }
        threadgroup_barrier();
        let cur_prefix = threadgroup_load("sgs", ns);
        let warp_excl = threadgroup_load("sgs", sg);
        let base_prefix = cur_prefix + warp_excl + thread_excl;
        if base < n {
            store(out[row_off + base], base_prefix + v0);
        }
        if base + 1u32 < n {
            store(out[row_off + base + 1u32], base_prefix + s1);
        }
        if base + 2u32 < n {
            store(out[row_off + base + 2u32], base_prefix + s2);
        }
        if base + 3u32 < n {
            store(out[row_off + base + 3u32], base_prefix + s3);
        }
        threadgroup_barrier();
        if lid == lsize - 1 {
            threadgroup_store("sgs", ns, base_prefix + s3);
        }
        threadgroup_barrier();
    }
}
