//! Steel tiled GEMM — #[kernel] DSL vs MLX steel/gemm/kernels/steel_gemm_fused.metal
//!
//! Block shape: 64×64×16 / 2×2. Each SIMD group covers a 32×32 sub-tile
//! via 4×4 M/N fragments of 8×8 and BK/8=2 K-fragments, accumulating
//! across K steps.

use metaltile::kernel;

#[kernel]
pub fn mt_steel_gemm_64x64x16_2x2<T>(a: Tensor<T>, b: Tensor<T>, out: Tensor<T>, #[constexpr] m: u32, #[constexpr] n: u32, #[constexpr] k: u32) {
    let tg_col = program_id::<0>();
    let tg_row = program_id::<1>();
    let sg_id = simd_group_id();
    let sg_m = sg_id / 2;
    let sg_n = sg_id % 2;
    let lane = simd_lane_id();

    let qid = lane / 4;
    let fm = (qid & 4) + ((lane / 2) % 4);
    let fn0 = (qid & 2) * 2 + (lane % 2) * 2;
    let fn1 = fn0 + 1;
    let sub_m0 = sg_m * 32;
    let sub_n0 = sg_n * 32;

    let n_fm = 4;
    let n_fn = 4;
    let n_kf = 2;

    for _fm_i in range(0, n_fm, 1) {
        for _fn_i in range(0, n_fn, 1) {
            let acc = simdgroup_alloc::<f32, 8, 8>();
            simdgroup_elem_store(acc, 0, 0);
            simdgroup_elem_store(acc, 1, 0);
            let m_row = sub_m0 + _fm_i * 8;
            let n_col = sub_n0 + _fn_i * 8;

            for _kf in range(0, n_kf, 1) {
                let kf = _kf * 8;
                let sub_a = simdgroup_alloc::<f16, 8, 8>();
                let sub_b = simdgroup_alloc::<f16, 8, 8>();
                // Direct load from device memory (no threadgroup staging)
                simdgroup_elem_store(sub_a, 0, load(a[(tg_row * 64 + m_row + fm) * k + kf + fn0]));
                simdgroup_elem_store(sub_a, 1, load(a[(tg_row * 64 + m_row + fm) * k + kf + fn1]));
                simdgroup_elem_store(sub_b, 0, load(b[(kf + fn0) * n + (tg_col * 64 + n_col + fm)]));
                simdgroup_elem_store(sub_b, 1, load(b[(kf + fn1) * n + (tg_col * 64 + n_col + fm)]));
                simdgroup_matmul(sub_a, sub_b, acc);
            }

            let r0 = simdgroup_elem_load(acc, 0);
            let r1 = simdgroup_elem_load(acc, 1);
            store(out[(tg_row * 64 + m_row + fm) * n + (tg_col * 64 + n_col + fn0)], r0.cast::<T>());
            store(out[(tg_row * 64 + m_row + fm) * n + (tg_col * 64 + n_col + fn1)], r1.cast::<T>());
        }
    }
}

inventory::submit! { crate::spec::BenchSpec {
    op: "steel_gemm_fused", subop: "bm64_bn64_bk16_wm2_wn2",
    kernel_name: "mt_steel_gemm_64x64x16_2x2",
    kernel_ir: mt_steel_gemm_64x64x16_2x2::kernel_ir_for,
    dtypes: crate::bench_types::FLOAT_DTYPES, tol: 1e-2f32,
    mlx_src: Some(include_str!(concat!(env!("OUT_DIR"), "/metal/steel/gemm/steel_gemm_fused.metal"))),
    mlx_pattern: Some("steel_gemm_fused_nn_{tn}_{tn}_bm64_bn64_bk16_wm2_wn2"),
    shapes: &[],
    dispatch: crate::spec::BenchDispatch::SteelGemm {
        m: 4096, n: 4096, k: 4096, check_m: 64, check_n: 64, check_k: 16, bm: 64, bn: 64, tpg: 128,
    },
    kernel_mode: Some(metaltile_core::ir::KernelMode::SimdGroup2D),
}}
