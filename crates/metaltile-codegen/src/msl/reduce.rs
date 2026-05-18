//! Threadgroup and SIMD-group reduction emission.
//!
//! Handles `Op::Reduce` lowering: two-level reduction for threadgroup-scope
//! (Reduction/Tile2D modes) and SIMD-group reduction for Elementwise/Grid3D.

use std::fmt::Write;

use metaltile_core::ir::{Kernel, ReduceKind};

use crate::wl;

impl super::MslGenerator {
    /// Emit a reduction: SIMD-group scope or two-level threadgroup scope.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_reduce(
        &self,
        out: &mut String,
        pad: &str,
        result_var: &str,
        input_var: &str,
        axis: u32,
        kind: ReduceKind,
        hoists: &mut Vec<String>,
        kernel: &Kernel,
    ) {
        // Threadgroup-scope reduction for Reduction/Tile2D modes:
        // Two-level simd_sum: intra-warp via simd_sum, inter-warp via 8-slot threadgroup mem.
        // axis=0 reduces across rows (threadgroup dimension), axis=1 across columns (SIMD group).
        let use_threadgroup = (kernel.mode == metaltile_core::ir::KernelMode::Reduction
            || kernel.mode == metaltile_core::ir::KernelMode::Tile2D)
            && axis == 0;
        if use_threadgroup {
            let tg_name = format!("{result_var}_sg");
            // 1024 threads / 32 per SIMD = 32 SIMD groups max.
            hoists.push(format!("threadgroup float {tg_name}[32];"));

            let (simd_fn, pad_val) = match kind {
                ReduceKind::Sum | ReduceKind::Mean => ("simd_sum", "0.0f"),
                ReduceKind::Max => ("simd_max", "-INFINITY"),
                ReduceKind::Min => ("simd_min", "INFINITY"),
                ReduceKind::Product => ("__mt_simd_product", "1.0f"),
            };

            // Phase 1: intra-warp reduction via simd_sum/max/min.
            wl!(out, "{pad}float {result_var};");
            wl!(out, "{pad}{{");
            wl!(out, "{pad}    float _sv = {simd_fn}(float({input_var}));");
            // Phase 2: lane 0 of each SIMD group writes its total.
            wl!(out, "{pad}    if (simd_lane == 0) {tg_name}[simd_group] = _sv;");
            wl!(out, "{pad}    threadgroup_barrier(mem_flags::mem_threadgroup);");
            // Phase 3: first SIMD group reduces warp totals and broadcasts via [0].
            wl!(out, "{pad}    if (simd_group == 0) {{");
            wl!(
                out,
                "{pad}        float _wv = simd_lane < n_simd ? {tg_name}[simd_lane] : {pad_val};"
            );
            wl!(out, "{pad}        {tg_name}[0] = {simd_fn}(_wv);");
            wl!(out, "{pad}    }}");
            wl!(out, "{pad}    threadgroup_barrier(mem_flags::mem_threadgroup);");
            if kind == ReduceKind::Mean {
                wl!(out, "{pad}    {result_var} = {tg_name}[0] / float(lsize);");
            } else {
                wl!(out, "{pad}    {result_var} = {tg_name}[0];");
            }
            wl!(out, "{pad}}}");
            return;
        }

        // Default: SIMD-group scope reduction (Elementwise / Grid3D modes).
        match kind {
            ReduceKind::Sum => wl!(out, "{pad}float {result_var} = simd_sum(float({input_var}));"),
            ReduceKind::Max => wl!(out, "{pad}float {result_var} = simd_max(float({input_var}));"),
            ReduceKind::Min => wl!(out, "{pad}float {result_var} = simd_min(float({input_var}));"),
            ReduceKind::Product =>
                wl!(out, "{pad}float {result_var} = __mt_simd_product(float({input_var}));"),
            ReduceKind::Mean => {
                wl!(
                    out,
                    "{pad}float {result_var} = simd_sum(float({input_var})) / float(simd_size);"
                );
            },
        }
    }
}
