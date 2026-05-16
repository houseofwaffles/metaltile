//! ArgReduce benchmark — #[kernel] DSL vs MLX metal/arg_reduce.metal

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="arg_reduce",
    subop="argmax",
    class=ArgReduce,
    n=1048576,
    check_n=4096,
    tpg=256,
    tol=0.5,
    mlx="argmax_float32",
    metal_file="arg_reduce.metal",
    dtypes=crate::spec::F32_ONLY,
)]
#[kernel]
pub fn mt_argmax_f32(inp: Tensor<f32>, out: Tensor<f32>, #[constexpr] n: u32) {
    let lid = tid;
    let mut best_val = neg_infinity();
    let mut best_idx = lid - lid;
    threadgroup_alloc("tg_vals", 256);
    threadgroup_alloc("tg_idxs", 256);
    let chunk = lsize * 4u32;
    let n_iters = (n + chunk - 1u32) / chunk;
    for _r in range(0, n_iters, 1) {
        let base = _r * chunk + lid;
        let p0 = base;
        let p1 = base + lsize;
        let p2 = base + lsize * 2u32;
        let p3 = base + lsize * 3u32;
        let v0 = select(p0 < n, load(inp[p0]), neg_infinity());
        let v1 = select(p1 < n, load(inp[p1]), neg_infinity());
        let v2 = select(p2 < n, load(inp[p2]), neg_infinity());
        let v3 = select(p3 < n, load(inp[p3]), neg_infinity());
        let b0 = v0 > best_val;
        best_val = select(b0, v0, best_val);
        best_idx = select(b0, p0, best_idx);
        let b1 = v1 > best_val;
        best_val = select(b1, v1, best_val);
        best_idx = select(b1, p1, best_idx);
        let b2 = v2 > best_val;
        best_val = select(b2, v2, best_val);
        best_idx = select(b2, p2, best_idx);
        let b3 = v3 > best_val;
        best_val = select(b3, v3, best_val);
        best_idx = select(b3, p3, best_idx);
    }
    threadgroup_store("tg_vals", lid, best_val);
    threadgroup_store("tg_idxs", lid, best_idx);
    threadgroup_barrier();
    {
        let ov = threadgroup_load("tg_vals", lid + 128);
        let oi = threadgroup_load("tg_idxs", lid + 128);
        let tv = threadgroup_load("tg_vals", lid);
        let ti = threadgroup_load("tg_idxs", lid);
        let bgt = ov > tv;
        let beq = ov == tv;
        let blt = oi < ti;
        let b23 = beq & blt;
        let bet = bgt | b23;
        let nv = select(bet, ov, tv);
        let ni = select(bet, oi, ti);
        if lid < 128 {
            threadgroup_store("tg_vals", lid, nv);
            threadgroup_store("tg_idxs", lid, ni);
        }
    }
    threadgroup_barrier();
    {
        let ov = threadgroup_load("tg_vals", lid + 64);
        let oi = threadgroup_load("tg_idxs", lid + 64);
        let tv = threadgroup_load("tg_vals", lid);
        let ti = threadgroup_load("tg_idxs", lid);
        let bgt = ov > tv;
        let beq = ov == tv;
        let blt = oi < ti;
        let b23 = beq & blt;
        let bet = bgt | b23;
        let nv = select(bet, ov, tv);
        let ni = select(bet, oi, ti);
        if lid < 64 {
            threadgroup_store("tg_vals", lid, nv);
            threadgroup_store("tg_idxs", lid, ni);
        }
    }
    threadgroup_barrier();
    {
        let ov = threadgroup_load("tg_vals", lid + 32);
        let oi = threadgroup_load("tg_idxs", lid + 32);
        let tv = threadgroup_load("tg_vals", lid);
        let ti = threadgroup_load("tg_idxs", lid);
        let bgt = ov > tv;
        let beq = ov == tv;
        let blt = oi < ti;
        let b23 = beq & blt;
        let bet = bgt | b23;
        let nv = select(bet, ov, tv);
        let ni = select(bet, oi, ti);
        if lid < 32 {
            threadgroup_store("tg_vals", lid, nv);
            threadgroup_store("tg_idxs", lid, ni);
        }
    }
    threadgroup_barrier();
    {
        let ov = threadgroup_load("tg_vals", lid + 16);
        let oi = threadgroup_load("tg_idxs", lid + 16);
        let tv = threadgroup_load("tg_vals", lid);
        let ti = threadgroup_load("tg_idxs", lid);
        let bgt = ov > tv;
        let beq = ov == tv;
        let blt = oi < ti;
        let b23 = beq & blt;
        let bet = bgt | b23;
        let nv = select(bet, ov, tv);
        let ni = select(bet, oi, ti);
        if lid < 16 {
            threadgroup_store("tg_vals", lid, nv);
            threadgroup_store("tg_idxs", lid, ni);
        }
    }
    threadgroup_barrier();
    {
        let ov = threadgroup_load("tg_vals", lid + 8);
        let oi = threadgroup_load("tg_idxs", lid + 8);
        let tv = threadgroup_load("tg_vals", lid);
        let ti = threadgroup_load("tg_idxs", lid);
        let bgt = ov > tv;
        let beq = ov == tv;
        let blt = oi < ti;
        let b23 = beq & blt;
        let bet = bgt | b23;
        let nv = select(bet, ov, tv);
        let ni = select(bet, oi, ti);
        if lid < 8 {
            threadgroup_store("tg_vals", lid, nv);
            threadgroup_store("tg_idxs", lid, ni);
        }
    }
    threadgroup_barrier();
    {
        let ov = threadgroup_load("tg_vals", lid + 4);
        let oi = threadgroup_load("tg_idxs", lid + 4);
        let tv = threadgroup_load("tg_vals", lid);
        let ti = threadgroup_load("tg_idxs", lid);
        let bgt = ov > tv;
        let beq = ov == tv;
        let blt = oi < ti;
        let b23 = beq & blt;
        let bet = bgt | b23;
        let nv = select(bet, ov, tv);
        let ni = select(bet, oi, ti);
        if lid < 4 {
            threadgroup_store("tg_vals", lid, nv);
            threadgroup_store("tg_idxs", lid, ni);
        }
    }
    threadgroup_barrier();
    {
        let ov = threadgroup_load("tg_vals", lid + 2);
        let oi = threadgroup_load("tg_idxs", lid + 2);
        let tv = threadgroup_load("tg_vals", lid);
        let ti = threadgroup_load("tg_idxs", lid);
        let bgt = ov > tv;
        let beq = ov == tv;
        let blt = oi < ti;
        let b23 = beq & blt;
        let bet = bgt | b23;
        let nv = select(bet, ov, tv);
        let ni = select(bet, oi, ti);
        if lid < 2 {
            threadgroup_store("tg_vals", lid, nv);
            threadgroup_store("tg_idxs", lid, ni);
        }
    }
    threadgroup_barrier();
    {
        let ov = threadgroup_load("tg_vals", lid + 1);
        let oi = threadgroup_load("tg_idxs", lid + 1);
        let tv = threadgroup_load("tg_vals", lid);
        let ti = threadgroup_load("tg_idxs", lid);
        let bgt = ov > tv;
        let beq = ov == tv;
        let blt = oi < ti;
        let b23 = beq & blt;
        let bet = bgt | b23;
        let nv = select(bet, ov, tv);
        let ni = select(bet, oi, ti);
        if lid < 1 {
            threadgroup_store("tg_vals", lid, nv);
            threadgroup_store("tg_idxs", lid, ni);
        }
    }
    threadgroup_barrier();
    if lid == 0 {
        store(out[0], threadgroup_load("tg_idxs", 0));
    }
}
