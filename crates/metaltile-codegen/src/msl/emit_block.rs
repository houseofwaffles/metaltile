//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Op-by-op block emission.
//!
//! Walks a `Block`'s op list and emits MSL statements for each IR op.

use std::{collections::BTreeMap, fmt::Write};

use metaltile_core::{
    dtype::DType,
    ir::{BinOpKind, Block, BlockId, CoopTileScope, Kernel, KernelMode, Op, ReduceKind, ValueId},
};
use rustc_hash::FxHashMap;

use super::{
    MslGenerator,
    matmul::{dim_to_msl_str, fmt_float},
};
use crate::{error::Result, passes::type_check::TypeEnv, wl};

impl MslGenerator {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_block(
        &self,
        block: &Block,
        all_blocks: &FxHashMap<BlockId, Block>,
        out: &mut String,
        indent: usize,
        kernel: &Kernel,
        type_env: &TypeEnv,
        extra_names: &BTreeMap<ValueId, String>,
        hoists: &mut Vec<String>,
    ) -> Result<()> {
        let has_tile = matches!(kernel.mode, KernelMode::Tile2D);
        let pad = "    ".repeat(indent);

        // Deduplicate variable names within this block.
        let mut dedup_extra = extra_names.clone();
        {
            use std::collections::BTreeMap as Map;
            let mut name_counts: Map<String, u32> = Map::new();
            for (vid, raw_name) in &block.names {
                let base = format!("v_{raw_name}");
                let count = name_counts.entry(base.clone()).or_insert(0);
                *count += 1;
                if *count > 1 {
                    dedup_extra.insert(*vid, format!("{base}_{count}"));
                }
            }
        }
        let extra_names = &dedup_extra;

        for (i, op) in block.ops.iter().enumerate() {
            let vid = block.results.get(i).and_then(|x| *x);

            match op {
                // ---- indexing ------------------------------------------
                Op::ProgramId { axis } => {
                    let v = self.vname(vid, block, extra_names);
                    match kernel.mode {
                        KernelMode::Elementwise => match axis {
                            0 => wl!(out, "{pad}uint {v} = tid;"),
                            _ => wl!(out, "{pad}uint {v} = 0;"),
                        },
                        KernelMode::Reduction => match axis {
                            0 => wl!(out, "{pad}uint {v} = tgid_x;"),
                            1 => wl!(out, "{pad}uint {v} = tgid_y;"),
                            2 => wl!(out, "{pad}uint {v} = tgid_z;"),
                            _ => wl!(out, "{pad}uint {v} = 0;"),
                        },
                        KernelMode::Grid3D => match axis {
                            0 => wl!(out, "{pad}uint {v} = gid.x;"),
                            1 => wl!(out, "{pad}uint {v} = gid.y;"),
                            2 => wl!(out, "{pad}uint {v} = gid.z;"),
                            _ => wl!(out, "{pad}uint {v} = 0;"),
                        },
                        KernelMode::Tile2D => match axis {
                            0 => wl!(out, "{pad}uint {v} = tgid.x;"),
                            1 => wl!(out, "{pad}uint {v} = tgid.y;"),
                            _ => wl!(out, "{pad}uint {v} = 0;"),
                        },
                        KernelMode::SimdGroup2D => match axis {
                            0 => wl!(out, "{pad}uint {v} = tid.x;"),
                            1 => wl!(out, "{pad}uint {v} = tid.y;"),
                            2 => wl!(out, "{pad}uint {v} = tid.z;"),
                            _ => wl!(out, "{pad}uint {v} = 0;"),
                        },
                    }
                },

                Op::Const { value } => {
                    let v = self.vname(vid, block, extra_names);
                    // Non-negative constants are emitted as `uint` to avoid
                    // -Wsign-compare warnings when used alongside uint operands
                    // (e.g. ProgramId, Arange, loop counters). Negative
                    // constants remain `int` since they require a signed type.
                    if *value >= 0 {
                        wl!(out, "{pad}uint {v} = {value}u;");
                    } else {
                        wl!(out, "{pad}int {v} = {value};");
                    }
                },

                Op::Arange { start, step, len } => {
                    let v = self.vname(vid, block, extra_names);
                    let s0 = start.unwrap_or(0.0);
                    let ds = step.unwrap_or(1.0);
                    let raw_idx = if has_tile {
                        format!("(tid.y * {len} + tid.x) % {len}")
                    } else if kernel.mode == KernelMode::Grid3D {
                        format!("gid.x % {len}")
                    } else {
                        format!("tid % {len}")
                    };
                    if s0 == 0.0 && ds == 1.0 {
                        wl!(out, "{pad}uint {v} = {raw_idx};");
                    } else {
                        wl!(out, "{pad}float {v} = {s0}f + float({raw_idx}) * {ds}f;");
                    }
                },

                // ---- memory --------------------------------------------
                Op::Load { src, indices, .. } => {
                    let v = self.vname(vid, block, extra_names);
                    let src_name = match (src.as_str(), kernel.mode) {
                        // DSL aliases for Metal builtins whose emitted
                        // parameter names differ by kernel mode.
                        ("simd_id", _) => "simd_group",
                        ("tgid_x", KernelMode::SimdGroup2D) => "tid.x",
                        ("tgid_y", KernelMode::SimdGroup2D) => "tid.y",
                        ("tgid_z", KernelMode::SimdGroup2D) => "tid.z",
                        ("tid_x", KernelMode::SimdGroup2D) => "lid.x",
                        ("tid_y", KernelMode::SimdGroup2D) => "lid.y",
                        ("tid_z", KernelMode::SimdGroup2D) => "lid.z",
                        _ => src.as_str(),
                    };
                    let src_dtype =
                        kernel.params.iter().find(|p| p.name == src_name).map(|p| p.dtype);
                    let promote_bf16 = self.config.native_bfloat
                        && src_dtype == Some(DType::BF16)
                        && matches!(kernel.mode, KernelMode::Elementwise | KernelMode::SimdGroup2D);
                    if indices.is_empty() {
                        if promote_bf16 {
                            wl!(out, "{pad}float {v} = float({src_name});");
                        } else {
                            wl!(out, "{pad}auto {v} = {src_name};");
                        }
                    } else {
                        let idx = self.emit_idx(indices, block, extra_names, kernel, src_name);
                        if promote_bf16 {
                            wl!(out, "{pad}float {v} = float({src_name}[{idx}]);");
                        } else {
                            wl!(out, "{pad}auto {v} = {src_name}[{idx}];");
                        }
                    }
                },

                Op::Store { dst, indices, value, .. } => {
                    let idx = self.emit_idx(indices, block, extra_names, kernel, dst);
                    let val = self.vname(Some(*value), block, extra_names);
                    let dst_dtype = kernel.params.iter().find(|p| p.name == *dst).map(|p| p.dtype);
                    if self.config.native_bfloat && dst_dtype == Some(DType::BF16) {
                        wl!(out, "{pad}{dst}[{idx}] = bfloat({val});");
                    } else {
                        wl!(out, "{pad}{dst}[{idx}] = {val};");
                    }
                },

                Op::Zeros { dtype, shape } => {
                    let v = self.vname(vid, block, extra_names);
                    let (decl, init_lines) = self.emit_tile_alloc(dtype, shape, &v, 0.0);
                    wl!(out, "{pad}{decl};");
                    for line in &init_lines {
                        wl!(out, "{pad}{line}");
                    }
                },

                Op::Splat { value: sv, dtype, shape } => {
                    let v = self.vname(vid, block, extra_names);
                    let (decl, _) = self.emit_tile_alloc(dtype, shape, &v, *sv);
                    let lit = fmt_float(*sv, dtype);
                    wl!(out, "{pad}{decl};");
                    match shape.num_elements() {
                        Some(1) | None if shape.rank() == 0 => {
                            wl!(out, "{pad}{v} = {lit};");
                        },
                        _ => {
                            let n = self.shape_nelems_str(shape);
                            wl!(out, "{pad}for (uint _i = 0; _i < {n}; _i++) {v}[_i] = {lit};");
                        },
                    }
                },

                Op::Slice { value, ranges } => {
                    let v = self.vname(vid, block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    let offset_str = type_env
                        .get(value)
                        .and_then(|tv| {
                            let shape = &tv.shape;
                            let rank = shape.rank();
                            if rank <= 1 {
                                None
                            } else {
                                let mut parts: Vec<String> = Vec::new();
                                for &(axis, start, _len) in ranges {
                                    if start == 0 {
                                        continue;
                                    }
                                    let stride_dims: Vec<String> = (axis as usize + 1..rank)
                                        .filter_map(|i| shape.dim(i))
                                        .map(dim_to_msl_str)
                                        .collect();
                                    if stride_dims.is_empty() {
                                        parts.push(start.to_string());
                                    } else {
                                        parts.push(format!(
                                            "{} * {}",
                                            start,
                                            stride_dims.join(" * ")
                                        ));
                                    }
                                }
                                if parts.is_empty() { None } else { Some(parts.join(" + ")) }
                            }
                        })
                        .unwrap_or_else(|| {
                            ranges.iter().map(|(_, start, _)| start).sum::<i64>().to_string()
                        });
                    if offset_str == "0" {
                        wl!(out, "{pad}auto {v} = {rv};");
                    } else {
                        wl!(out, "{pad}auto {v} = {rv} + {offset_str};");
                    }
                },

                Op::Transpose { value } => {
                    let v = self.vname(vid, block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    if let Some(tv) = type_env.get(value) {
                        let rank = tv.shape.rank();
                        if rank == 2 {
                            let m = tv.shape.dim(0).map(dim_to_msl_str).unwrap_or("M".into());
                            let n = tv.shape.dim(1).map(dim_to_msl_str).unwrap_or("N".into());
                            let t = self.msl_type_name(tv.dtype);
                            wl!(out, "{pad}{t} {v}[{m} * {n}];");
                            wl!(out, "{pad}for (uint r = 0; r < {m}; r++) {{");
                            wl!(out, "{pad}    for (uint c = 0; c < {n}; c++) {{");
                            wl!(out, "{pad}        {v}[c * {m} + r] = {rv}[r * {n} + c];");
                            wl!(out, "{pad}    }}");
                            wl!(out, "{pad}}}");
                        } else {
                            wl!(out, "{pad}// Transpose: unsupported rank {rank}");
                            wl!(out, "{pad}auto {v} = {rv};");
                        }
                    } else {
                        wl!(out, "{pad}// Transpose: caller uses transposed index strides");
                        wl!(out, "{pad}auto {v} = {rv};");
                    }
                },

                // ---- compute -------------------------------------------
                // FMA fusion is now an IR pass (`FmaFusionPass`) — when
                // it fires, the IR has a real `Op::Fma { a, b, c }`
                // here.  The emit lowers it to a single `fma(a, b, c)`
                // line; the upstream `Op::Mul` becomes dead and is
                // swept by DCE.  Pre-#209/3 this was a textual
                // peephole on `BinOp::Add` / `BinOp::Sub` that left
                // the standalone Mul behind in MSL as a dead variable.
                Op::Fma { a, b, c } => {
                    let v = self.vname(vid, block, extra_names);
                    let av = self.vname(Some(*a), block, extra_names);
                    let bv = self.vname(Some(*b), block, extra_names);
                    let cv = self.vname(Some(*c), block, extra_names);
                    wl!(out, "{pad}auto {v} = fma({av}, {bv}, {cv});");
                },

                Op::BinOp { op, lhs, rhs } => {
                    let v = self.vname(vid, block, extra_names);
                    let l = self.vname(Some(*lhs), block, extra_names);
                    let r = self.vname(Some(*rhs), block, extra_names);
                    match op {
                        BinOpKind::Max
                        | BinOpKind::Min
                        | BinOpKind::Pow
                        | BinOpKind::ATan2
                        | BinOpKind::Rem => {
                            wl!(out, "{pad}auto {v} = {}({l}, {r});", op.msl_symbol())
                        },
                        BinOpKind::And => wl!(out, "{pad}auto {v} = ({l} && {r});"),
                        BinOpKind::Or => wl!(out, "{pad}auto {v} = ({l} || {r});"),
                        BinOpKind::Xor => wl!(out, "{pad}auto {v} = ((bool){l} != (bool){r});"),
                        BinOpKind::BitAnd => wl!(out, "{pad}auto {v} = ({l} & {r});"),
                        BinOpKind::BitOr => wl!(out, "{pad}auto {v} = ({l} | {r});"),
                        BinOpKind::BitXor => wl!(out, "{pad}auto {v} = ({l} ^ {r});"),
                        BinOpKind::Shl => wl!(out, "{pad}auto {v} = ({l} << {r});"),
                        BinOpKind::Shr => wl!(out, "{pad}auto {v} = ({l} >> {r});"),
                        BinOpKind::Mod => wl!(out, "{pad}auto {v} = ({l} % {r});"),
                        BinOpKind::Add => wl!(out, "{pad}auto {v} = {l} + {r};"),
                        BinOpKind::Sub => wl!(out, "{pad}auto {v} = {l} - {r};"),
                        _ => wl!(out, "{pad}auto {v} = {l} {} {r};", op.msl_symbol()),
                    }
                },

                Op::UnaryOp { op, value } => {
                    let v = self.vname(vid, block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    let expr = op.msl_emit(&rv);
                    wl!(out, "{pad}auto {v} = {expr};");
                },

                Op::Activation { kind, value } => {
                    let v = self.vname(vid, block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    wl!(out, "{pad}auto {v} = {}({rv});", kind.msl_fn());
                },

                Op::Select { cond, on_true, on_false } => {
                    let v = self.vname(vid, block, extra_names);
                    let vc = self.vname(Some(*cond), block, extra_names);
                    let vt = self.vname(Some(*on_true), block, extra_names);
                    let vf = self.vname(Some(*on_false), block, extra_names);
                    wl!(out, "{pad}auto {v} = bool({vc}) ? {vt} : {vf};");
                },

                Op::Broadcast { value, shape } => {
                    let v = self.vname(vid, block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    let n = self.shape_nelems_str(shape);
                    let dt = type_env
                        .get(value)
                        .map(|tv| self.msl_type_name(tv.dtype))
                        .unwrap_or("float");
                    wl!(out, "{pad}{dt} {v}_data[{n}];");
                    wl!(out, "{pad}for (uint _i = 0; _i < {n}; _i++) {v}_data[_i] = {dt}({rv});");
                    wl!(out, "{pad}{dt}* {v} = {v}_data;");
                },

                Op::Dot { .. } => {
                    self.emit_tiled(out, &pad, kernel, vid)?;
                },

                Op::StrideReduce {
                    src,
                    offset,
                    stride,
                    end,
                    op: rk,
                    transform,
                    secondary_src,
                    secondary_base,
                    ..
                } => {
                    let v = self.vname(vid, block, extra_names);
                    let off = self.vname(Some(*offset), block, extra_names);
                    let st = self.vname(Some(*stride), block, extra_names);
                    let en = self.vname(Some(*end), block, extra_names);
                    let init = match rk {
                        ReduceKind::Sum | ReduceKind::Mean => "0.0f",
                        ReduceKind::Max => "-INFINITY",
                        ReduceKind::Min => "INFINITY",
                        ReduceKind::Product => "1.0f",
                    };
                    let base_elem = if let Some(sec_src) = secondary_src {
                        let base_v = self.vname(*secondary_base, block, extra_names);
                        format!("float({src}[_i]) * float({sec_src}[_i - {base_v}])")
                    } else {
                        format!("float({src}[_i])")
                    };
                    let elem_expr = match transform.as_ref().map(|v| v.as_slice()) {
                        None | Some(&[]) => base_elem,
                        Some(ops) => {
                            let mut expr = base_elem;
                            for sub_op in ops {
                                expr = match sub_op {
                                    Op::UnaryOp { op, .. } => op.msl_emit(&expr),
                                    Op::Activation { kind, .. } => {
                                        format!("{}({})", kind.msl_fn(), expr)
                                    },
                                    Op::Cast { dtype, .. } => self.emit_cast_expr(*dtype, &expr),
                                    Op::BinOp { op, rhs, .. } => {
                                        let rv = self.vname(Some(*rhs), block, extra_names);
                                        match op {
                                            BinOpKind::Mul => format!("({} * float({rv}))", expr),
                                            BinOpKind::Add => format!("({} + float({rv}))", expr),
                                            BinOpKind::Sub => format!("({} - float({rv}))", expr),
                                            BinOpKind::Div => format!("({} / float({rv}))", expr),
                                            _ => expr,
                                        }
                                    },
                                    _ => expr,
                                };
                            }
                            expr
                        },
                    };
                    let has_tid = matches!(kernel.mode, KernelMode::Reduction | KernelMode::Tile2D);
                    if has_tid {
                        let update = match rk {
                            ReduceKind::Sum | ReduceKind::Mean => format!("{v} += {elem_expr};"),
                            ReduceKind::Max => format!("{v} = max({v}, {elem_expr});"),
                            ReduceKind::Min => format!("{v} = min({v}, {elem_expr});"),
                            ReduceKind::Product => format!("{v} *= {elem_expr};"),
                        };
                        wl!(out, "{pad}float {v} = {init};");
                        wl!(out, "{pad}{{");
                        wl!(out, "{pad}    uint _sz   = {en} - {off};");
                        wl!(out, "{pad}    uint _full = _sz / (lsize * 4u);");
                        wl!(
                            out,
                            "{pad}    int  _xtra = (int)_sz - (int)(_full * lsize * 4u) - (int)(tid * 4u);"
                        );
                        wl!(out, "{pad}    if (_xtra >= 4) {{ _full++; _xtra = 0; }}");
                        wl!(out, "{pad}    uint _base = {off} + tid * 4u;");
                        wl!(
                            out,
                            "{pad}    for (uint _b = 0; _b < _full; _b++, _base += lsize * 4u) {{"
                        );
                        wl!(out, "{pad}        for (uint _k = 0; _k < 4u; _k++) {{");
                        wl!(out, "{pad}            uint _i = _base + _k;");
                        wl!(out, "{pad}            {update}");
                        wl!(out, "{pad}        }}");
                        wl!(out, "{pad}    }}");
                        wl!(out, "{pad}    for (int _k = 0; _k < _xtra; _k++) {{");
                        wl!(out, "{pad}        uint _i = _base + (uint)_k;");
                        wl!(out, "{pad}        {update}");
                        wl!(out, "{pad}    }}");
                        wl!(out, "{pad}}}");
                    } else {
                        let update = match rk {
                            ReduceKind::Sum | ReduceKind::Mean => format!("{v} += {elem_expr};"),
                            ReduceKind::Max => format!("{v} = max({v}, {elem_expr});"),
                            ReduceKind::Min => format!("{v} = min({v}, {elem_expr});"),
                            ReduceKind::Product => format!("{v} *= {elem_expr};"),
                        };
                        wl!(out, "{pad}float {v} = {init};");
                        wl!(out, "{pad}for (uint _i = {off}; _i < {en}; _i += {st}) {{");
                        wl!(out, "{pad}    {update}");
                        wl!(out, "{pad}}}");
                    }
                },

                Op::Reduce { value: val, axis, op: rk } => {
                    let v = self.vname(vid, block, extra_names);
                    let rv = self.vname(Some(*val), block, extra_names);
                    self.emit_reduce(out, &pad, &v, &rv, *axis, *rk, hoists, kernel);
                },

                Op::Cast { value, dtype } => {
                    let v = self.vname(vid, block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    let type_decl = self.msl_type_name(*dtype);
                    let src_dtype = type_env.get(value).map(|tv| tv.dtype);
                    wl!(
                        out,
                        "{pad}{type_decl} {v} = {};",
                        self.emit_cast_expr_with_src(*dtype, src_dtype, &rv)
                    );
                },

                // ---- control flow --------------------------------------
                Op::Loop { var, start, end, step, body: bid } => {
                    let s = self.vname(Some(*start), block, extra_names);
                    let e = self.vname(Some(*end), block, extra_names);
                    let st = self.vname(Some(*step), block, extra_names);
                    let vn = format!("i_{}", var.as_u32());
                    wl!(out, "{pad}for (uint {vn} = {s}; {vn} < {e}; {vn} += {st}) {{");
                    if let Some(bb) = all_blocks.get(bid) {
                        let loop_var_vid = ValueId::new(0xC000_0000 | var.as_u32());
                        let mut inner_names: BTreeMap<ValueId, String> =
                            block.names.iter().map(|(&k, v)| (k, format!("v_{v}"))).collect();
                        inner_names.extend(extra_names.iter().map(|(&k, v)| (k, v.clone())));
                        inner_names.insert(loop_var_vid, vn.clone());
                        inner_names.insert(ValueId::new(var.as_u32() + 0x4000_0000), vn.clone());
                        self.emit_block(
                            bb,
                            all_blocks,
                            out,
                            indent + 1,
                            kernel,
                            type_env,
                            &inner_names,
                            hoists,
                        )?;
                    }
                    wl!(out, "{pad}}}");
                },

                // ---- cross-kernel call (should be resolved by KernelInlinePass) ----
                Op::KernelCall { callee, .. } => {
                    // If we reach here the KernelInlinePass did not run or
                    // could not resolve this call.  Emit a compile-error
                    // placeholder so the Metal shader fails loudly rather
                    // than silently producing wrong results.
                    wl!(
                        out,
                        "{pad}/* ERROR: unresolved KernelCall to `{callee}` — \
                         KernelInlinePass must run before MSL emit */",
                    );
                },

                // ---- escape hatch --------------------------------------
                Op::InlineMsl { source, inputs, outputs } => {
                    for (oi, slot) in outputs.iter().enumerate() {
                        let out_vid = vid.map(|v| ValueId::new(v.as_u32() + oi as u32));
                        let vn = self.vname(out_vid, block, extra_names);
                        wl!(out, "{pad}{} {vn};", self.msl_type_name(slot.dtype));
                    }
                    for (ii, inp_vid) in inputs.iter().enumerate() {
                        let inp = self.vname(Some(*inp_vid), block, extra_names);
                        wl!(out, "{pad}auto _in{ii} = {inp};");
                    }
                    for line in source.lines() {
                        wl!(out, "{pad}{line}");
                    }
                },

                // ---- CoopTile setup: declare descriptor + operator + ct_a/ct_b/ct_c
                Op::CoopTileSetup {
                    name,
                    m,
                    n,
                    k,
                    ta,
                    tb,
                    tc,
                    acc_mode,
                    exec_scope,
                    act_dtype,
                    acc_dtype,
                    direct_inputs,
                    a_is_tg,
                    a_ei,
                    a_eo,
                    b_is_tg,
                    b_ei,
                    b_eo,
                } => {
                    let scope_s = match exec_scope {
                        CoopTileScope::SimdGroup => "metal::execution_simdgroup",
                        CoopTileScope::Threadgroup => "metal::execution_threadgroup",
                    };
                    // Apple's `matmul2d_descriptor::mode` has two members:
                    // `multiply` (C ← A·B, fresh) and `multiply_accumulate`
                    // (C ← C + A·B). `Overwrite` is the fresh-product case.
                    let acc_mode_s = match acc_mode {
                        metaltile_core::ir::CoopTileAccMode::Overwrite => "mode::multiply",
                        metaltile_core::ir::CoopTileAccMode::MultiplyAccumulate =>
                            "mode::multiply_accumulate",
                    };
                    let ta_s = if *ta { "true" } else { "false" };
                    let tb_s = if *tb { "true" } else { "false" };
                    let tc_s = if *tc { "true" } else { "false" };
                    let act_t = self.msl_type_name(*act_dtype);
                    let acc_t = self.msl_type_name(*acc_dtype);
                    wl!(out, "{pad}// --- CoopTileSetup: {name} ({m}×{n}×{k}) ---");
                    wl!(
                        out,
                        "{pad}constexpr auto {name}_desc = mpp::tensor_ops::matmul2d_descriptor("
                    );
                    wl!(out, "{pad}    /*M=*/{m}, /*N=*/{n}, /*K=*/{k},");
                    wl!(out, "{pad}    /*ta=*/{ta_s}, /*tb=*/{tb_s}, /*tc=*/{tc_s},");
                    wl!(out, "{pad}    mpp::tensor_ops::matmul2d_descriptor::{acc_mode_s});");
                    wl!(out, "{pad}mpp::tensor_ops::matmul2d<{name}_desc, {scope_s}> {name}_op;");
                    if *direct_inputs {
                        // Direct-input mode: emit type aliases + ct_c only (no ct_a/ct_b).
                        let a_as = if *a_is_tg { "threadgroup" } else { "device" };
                        let b_as = if *b_is_tg { "threadgroup" } else { "device" };
                        wl!(
                            out,
                            "{pad}using {name}_tA_t = metal::tensor<{a_as} {act_t}, metal::extents<int, {a_ei}, {a_eo}>, metal::tensor_inline>;"
                        );
                        wl!(
                            out,
                            "{pad}using {name}_tB_t = metal::tensor<{b_as} {act_t}, metal::extents<int, {b_ei}, {b_eo}>, metal::tensor_inline>;"
                        );
                        wl!(
                            out,
                            "{pad}auto {name}_ct_c = {name}_op.template get_destination_cooperative_tensor<{name}_tA_t, {name}_tB_t, {acc_t}>();"
                        );
                    } else {
                        wl!(
                            out,
                            "{pad}auto {name}_ct_a = {name}_op.template get_left_input_cooperative_tensor<{act_t}, {act_t}, {acc_t}>();"
                        );
                        wl!(
                            out,
                            "{pad}auto {name}_ct_b = {name}_op.template get_right_input_cooperative_tensor<{act_t}, {act_t}, {acc_t}>();"
                        );
                        wl!(
                            out,
                            "{pad}auto {name}_ct_c = {name}_op.template get_destination_cooperative_tensor<decltype({name}_ct_a), decltype({name}_ct_b), {acc_t}>();"
                        );
                    }
                },

                // ---- CoopTile zero: zero C accumulator before K-loop
                Op::CoopTileZero { name } => {
                    wl!(
                        out,
                        "{pad}for (uint16_t _i = 0; _i < {name}_ct_c.get_capacity(); ++_i) {name}_ct_c[_i] = {{}};"
                    );
                },

                // ---- CoopTile load A
                Op::CoopTileLoadA { name, ptr_name, ptr_offset, is_tg, dtype, ei, eo, direct } => {
                    let t = self.msl_type_name(*dtype);
                    let as_ = if *is_tg { "threadgroup" } else { "device" };
                    let ptr = if let Some(off) = ptr_offset {
                        let ov = self.vname(Some(*off), block, extra_names);
                        format!("{ptr_name} + {ov}")
                    } else {
                        ptr_name.clone()
                    };
                    let ptr_expr =
                        if *is_tg { ptr.clone() } else { format!("const_cast<{as_} {t}*>({ptr})") };
                    if *direct {
                        // Direct-input mode: instantiate tensor view using the type alias.
                        wl!(
                            out,
                            "{pad}{name}_tA_t {name}_tA({ptr_expr}, metal::extents<int, {ei}, {eo}>{{}});"
                        );
                    } else {
                        wl!(
                            out,
                            "{pad}metal::tensor<{as_} {t}, metal::extents<int, {ei}, {eo}>, metal::tensor_inline> {name}_tA({ptr_expr}, metal::extents<int, {ei}, {eo}>{{}}); {name}_ct_a.load({name}_tA);"
                        );
                    }
                },

                // ---- CoopTile load B
                Op::CoopTileLoadB { name, ptr_name, ptr_offset, is_tg, dtype, ei, eo, direct } => {
                    let t = self.msl_type_name(*dtype);
                    let as_ = if *is_tg { "threadgroup" } else { "device" };
                    let ptr = if let Some(off) = ptr_offset {
                        let ov = self.vname(Some(*off), block, extra_names);
                        format!("{ptr_name} + {ov}")
                    } else {
                        ptr_name.clone()
                    };
                    let ptr_expr =
                        if *is_tg { ptr.clone() } else { format!("const_cast<{as_} {t}*>({ptr})") };
                    if *direct {
                        // Direct-input mode: instantiate tensor view using the type alias.
                        wl!(
                            out,
                            "{pad}{name}_tB_t {name}_tB({ptr_expr}, metal::extents<int, {ei}, {eo}>{{}});"
                        );
                    } else {
                        wl!(
                            out,
                            "{pad}metal::tensor<{as_} {t}, metal::extents<int, {ei}, {eo}>, metal::tensor_inline> {name}_tB({ptr_expr}, metal::extents<int, {ei}, {eo}>{{}}); {name}_ct_b.load({name}_tB);"
                        );
                    }
                },

                // ---- CoopTile run: execute A·B → C
                Op::CoopTileRun { name, direct } =>
                    if *direct {
                        wl!(out, "{pad}{name}_op.run({name}_tA, {name}_tB, {name}_ct_c);");
                    } else {
                        wl!(out, "{pad}{name}_op.run({name}_ct_a, {name}_ct_b, {name}_ct_c);");
                    },

                // ---- CoopTile store C
                Op::CoopTileStoreC { name, ptr_name, ptr_offset, is_tg, dtype, ei, eo } => {
                    let t = self.msl_type_name(*dtype);
                    let as_ = if *is_tg { "threadgroup" } else { "device" };
                    let ptr = if let Some(off) = ptr_offset {
                        let ov = self.vname(Some(*off), block, extra_names);
                        format!("{ptr_name} + {ov}")
                    } else {
                        ptr_name.clone()
                    };
                    wl!(
                        out,
                        "{pad}metal::tensor<{as_} {t}, metal::extents<int, {ei}, {eo}>, metal::tensor_inline> {name}_tC({ptr}, metal::extents<int, {ei}, {eo}>{{}}); {name}_ct_c.store({name}_tC);"
                    );
                },

                // ---- fused elementwise chain ---------------------------
                Op::FusedElementwise { ops: fused_ops } => {
                    let v = self.vname(vid, block, extra_names);
                    if fused_ops.is_empty() {
                        wl!(out, "{pad}auto {v} = {{}};");
                    } else if let Some(resolved_vid) = vid {
                        let last_idx = fused_ops.len() - 1;
                        let expr = self.emit_fused_expr(
                            out,
                            &pad,
                            fused_ops,
                            block,
                            kernel,
                            type_env,
                            extra_names,
                            resolved_vid,
                            last_idx,
                        );
                        let result_is_bf16 = type_env
                            .get(&resolved_vid)
                            .map(|tv| tv.dtype == DType::BF16)
                            .unwrap_or(false);
                        if self.config.native_bfloat && result_is_bf16 {
                            if expr.starts_with("bfloat(") || expr.starts_with("as_type<bfloat2>(")
                            {
                                wl!(out, "{pad}bfloat {v} = {expr};");
                            } else {
                                wl!(out, "{pad}bfloat {v} = bfloat({expr});");
                            }
                        } else if result_is_bf16 {
                            let bf_type = self.msl_type_name(DType::BF16);
                            wl!(out, "{pad}{bf_type} {v} = {expr};");
                        } else {
                            wl!(out, "{pad}auto {v} = {expr};");
                        }
                    }
                },

                Op::VectorExtract { vec, lane } => {
                    let v = self.vname(vid, block, extra_names);
                    let src = self.vname(Some(*vec), block, extra_names);
                    // Use array-index syntax — Metal vec types support
                    // operator[] returning scalar of the element type.
                    wl!(out, "{pad}auto {v} = {src}[{lane}];");
                },

                // ---- vectorised data movement ---------------------------
                Op::VectorLoad { src, byte_offset, len } => {
                    let v = self.vname(vid, block, extra_names);
                    let bo = self.vname(Some(*byte_offset), block, extra_names);
                    let param_dtype = kernel
                        .params
                        .iter()
                        .find(|p| p.name == *src)
                        .map(|p| p.dtype)
                        .unwrap_or(DType::F32);
                    let scalar_t = param_dtype.msl_name();
                    let vec_t: String = match (*len, param_dtype) {
                        (4, DType::F16) => "half4".into(),
                        (4, DType::F32) => "float4".into(),
                        (4, DType::BF16) if self.config.native_bfloat => "bfloat4".into(),
                        (2, DType::BF16) if self.config.native_bfloat => "bfloat2".into(),
                        (2, _) => format!("{}2", scalar_t),
                        (4, _) => format!("{}4", scalar_t),
                        _ => format!("{}4", scalar_t),
                    };
                    wl!(
                        out,
                        "{pad}{vec_t} {v} = *((device {vec_t}*)((device {scalar_t}*){src} + {bo}));"
                    );
                },

                Op::VectorStore { dst, byte_offset, len, value } => {
                    let rv = self.vname(Some(*value), block, extra_names);
                    let bo = self.vname(Some(*byte_offset), block, extra_names);
                    let param_dtype = kernel
                        .params
                        .iter()
                        .find(|p| p.name == *dst)
                        .map(|p| p.dtype)
                        .unwrap_or(DType::F32);
                    let scalar_t = param_dtype.msl_name();
                    let vec_t: String = match (*len, param_dtype) {
                        (4, DType::F16) => "half4".into(),
                        (4, DType::F32) => "float4".into(),
                        (4, DType::BF16) if self.config.native_bfloat => "bfloat4".into(),
                        (2, DType::BF16) if self.config.native_bfloat => "bfloat2".into(),
                        (2, _) => format!("{}2", scalar_t),
                        (4, _) => format!("{}4", scalar_t),
                        _ => format!("{}4", scalar_t),
                    };
                    wl!(
                        out,
                        "{pad}*((device {vec_t}*)((device {scalar_t}*){dst} + {bo})) = {vec_t}({rv});"
                    );
                },

                // ---- vector pack --------------------------------------
                Op::Pack { dtype, elements } => {
                    let v = self.vname(vid, block, extra_names);
                    let args: Vec<String> =
                        elements.iter().map(|e| self.vname(Some(*e), block, extra_names)).collect();
                    let vec_t: String = match (elements.len() as u32, *dtype) {
                        (4, DType::F16) => "half4".into(),
                        (4, DType::F32) => "float4".into(),
                        (4, DType::BF16) if self.config.native_bfloat => "bfloat4".into(),
                        (2, DType::BF16) if self.config.native_bfloat => "bfloat2".into(),
                        (2, _) => format!("{}2", dtype.msl_name()),
                        (4, _) => format!("{}4", dtype.msl_name()),
                        _ => format!("{}4", dtype.msl_name()),
                    };
                    // Metal does not support bfloat4(v0,v1,v2,v3) component
                    // constructors even in Metal 3.1+; emit zero-init +
                    // per-element assignment instead.
                    let is_bfloat = matches!(*dtype, DType::BF16) && self.config.native_bfloat;
                    if is_bfloat {
                        wl!(out, "{pad}{vec_t} {v} = 0;");
                        let lanes = ["x", "y", "z", "w"];
                        for (i, arg) in args.iter().enumerate().take(elements.len()) {
                            wl!(out, "{pad}{v}.{ln} = bfloat({arg});", ln = lanes[i]);
                        }
                    } else {
                        let args_str = args.join(", ");
                        wl!(out, "{pad}{vec_t} {v} = {vec_t}({args_str});");
                    }
                },

                // ---- conditional branch ----------------------------------
                Op::If { cond, then_block: tbid, else_block } => {
                    let cv = self.vname(Some(*cond), block, extra_names);
                    wl!(out, "{pad}if ({cv}) {{");
                    let mut child_names: BTreeMap<ValueId, String> =
                        block.names.iter().map(|(&k, v)| (k, format!("v_{v}"))).collect();
                    child_names.extend(extra_names.iter().map(|(&k, v)| (k, v.clone())));
                    if let Some(tb) = all_blocks.get(tbid) {
                        self.emit_block(
                            tb,
                            all_blocks,
                            out,
                            indent + 1,
                            kernel,
                            type_env,
                            &child_names,
                            hoists,
                        )?;
                    }
                    if let Some(ebid) = else_block {
                        wl!(out, "{pad}}} else {{");
                        if let Some(eb) = all_blocks.get(ebid) {
                            self.emit_block(
                                eb,
                                all_blocks,
                                out,
                                indent + 1,
                                kernel,
                                type_env,
                                &child_names,
                                hoists,
                            )?;
                        }
                    }
                    wl!(out, "{pad}}}");
                },

                // ---- shape manipulation (zero-cost aliases) -------------
                Op::ExpandDims { value, .. } | Op::Reshape { value, .. } => {
                    let v = self.vname(vid, block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    wl!(out, "{pad}// ExpandDims/Reshape: same data, new indexing");
                    wl!(out, "{pad}auto {v} = {rv};");
                },

                Op::Cat { .. } => {
                    let v = self.vname(vid, block, extra_names);
                    wl!(out, "{pad}// Op::Cat: needs lowering pass");
                    wl!(out, "{pad}auto {v} = {{}};");
                },

                // ---- gather / scatter ------------------------------------
                Op::Gather { src, indices, .. } => {
                    let v = self.vname(vid, block, extra_names);
                    let iv = self.vname(Some(*indices), block, extra_names);
                    wl!(out, "{pad}// Op::Gather: indexed load from {src}");
                    wl!(out, "{pad}auto {v} = {src}[{iv}];");
                },

                Op::Scatter { dst, indices, value, .. } => {
                    let iv = self.vname(Some(*indices), block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    wl!(out, "{pad}// Op::Scatter: indexed store to {dst}");
                    wl!(out, "{pad}{dst}[{iv}] = {rv};");
                },

                // ---- atomics ----------------------------------------------
                Op::Atomic { op: ak, scope, dst, index, value } => {
                    let iv = self.vname(Some(*index), block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    match scope {
                        metaltile_core::ir::AtomicScope::Device => {
                            wl!(
                                out,
                                "{pad}{}({dst} + {iv}, {rv}, memory_order_relaxed);",
                                ak.msl_fn(),
                            );
                        },
                        metaltile_core::ir::AtomicScope::Threadgroup => {
                            // `dst` is a `threadgroup_alloc`'d uint array.
                            // Reinterpret the slot as `threadgroup atomic_uint*`
                            // — same form MLX uses (turbo_quant.metal).
                            wl!(
                                out,
                                "{pad}{}((threadgroup atomic_uint*)&{dst}[{iv}], {rv}, memory_order_relaxed);",
                                ak.msl_fn(),
                            );
                        },
                    }
                },

                // ---- scan operations --------------------------------------
                //
                // Maps `simd_scan_{inclusive,exclusive}` to Metal's
                // `simd_prefix_{inclusive,exclusive}_{sum,product}` simdgroup
                // intrinsics. The codegen previously emitted a `value + init`
                // placeholder which silently returned the wrong result (no
                // cross-lane communication at all) — `mt_scan_f32`'s
                // hierarchical-scan pattern depended on the exclusive sum
                // and was producing garbage on GPU dispatch. Min/Max scans
                // aren't shipped by Metal as built-ins; emit a placeholder
                // for them rather than silently miscompile and add a
                // TODO so callers know to lower to a butterfly shuffle if
                // they need them.
                Op::Scan { value, axis: _, op: rk, exclusive } => {
                    let v = self.vname(vid, block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    let fn_name = match (rk, *exclusive) {
                        (ReduceKind::Sum | ReduceKind::Mean, true) =>
                            Some("simd_prefix_exclusive_sum"),
                        (ReduceKind::Sum | ReduceKind::Mean, false) =>
                            Some("simd_prefix_inclusive_sum"),
                        _ => None,
                    };
                    match fn_name {
                        Some(f) => wl!(out, "{pad}float {v} = {f}({rv});"),
                        None => {
                            let init = match rk {
                                ReduceKind::Max => "-INFINITY",
                                ReduceKind::Min => "INFINITY",
                                ReduceKind::Product => "1.0f",
                                _ => "0.0f",
                            };
                            // TODO: lower min/max scans via simd_shuffle_xor
                            // butterfly. No kernel hits this today.
                            wl!(
                                out,
                                "{pad}float {v} = {rv} + {init}; // TODO: simd \
                                 scan min/max not implemented"
                            );
                        },
                    }
                },

                Op::StrideScan { dst, offset, end, .. } => {
                    let _off = self.vname(Some(*offset), block, extra_names);
                    let _en = self.vname(Some(*end), block, extra_names);
                    wl!(out, "{pad}// StrideScan: prefix sum over {dst}");
                    wl!(
                        out,
                        "{pad}auto {} = 0; // placeholder",
                        self.vname(vid, block, extra_names)
                    );
                },

                Op::StrideArgReduce { src, offset, end, op: _rk } => {
                    let v = self.vname(vid, block, extra_names);
                    let off = self.vname(Some(*offset), block, extra_names);
                    let en = self.vname(Some(*end), block, extra_names);
                    wl!(out, "{pad}float best_val = -INFINITY;");
                    wl!(out, "{pad}uint {v} = 0;");
                    wl!(out, "{pad}for (uint _i = {off}; _i < {en}; _i++) {{");
                    wl!(out, "{pad}    if ({src}[_i] > best_val) {{");
                    wl!(out, "{pad}        best_val = {src}[_i];");
                    wl!(out, "{pad}        {v} = _i;");
                    wl!(out, "{pad}    }}");
                    wl!(out, "{pad}}}");
                },

                Op::StrideStore { src, dst, offset, end, scalar, aux_src } => {
                    let off = self.vname(Some(*offset), block, extra_names);
                    let en = self.vname(Some(*end), block, extra_names);
                    let sc = self.vname(Some(*scalar), block, extra_names);
                    let has_tid = matches!(kernel.mode, KernelMode::Reduction | KernelMode::Tile2D);
                    if has_tid {
                        wl!(out, "{pad}{{");
                        wl!(out, "{pad}    uint _sz   = {en} - {off};");
                        wl!(out, "{pad}    uint _full = _sz / (lsize * 4u);");
                        wl!(
                            out,
                            "{pad}    int  _xtra = (int)_sz - (int)(_full * lsize * 4u) - (int)(tid * 4u);"
                        );
                        wl!(out, "{pad}    if (_xtra >= 4) {{ _full++; _xtra = 0; }}");
                        wl!(out, "{pad}    uint _base = {off} + tid * 4u;");
                        wl!(
                            out,
                            "{pad}    for (uint _b = 0; _b < _full; _b++, _base += lsize * 4u) {{"
                        );
                        wl!(out, "{pad}        for (uint _k = 0; _k < 4u; _k++) {{");
                        wl!(out, "{pad}            uint _i = _base + _k;");
                        if let Some(aux) = aux_src {
                            wl!(out, "{pad}            {dst}[_i] = {src}[_i] * {sc} * {aux}[_i];");
                        } else {
                            wl!(out, "{pad}            {dst}[_i] = {src}[_i] * {sc};");
                        }
                        wl!(out, "{pad}        }}");
                        wl!(out, "{pad}    }}");
                        wl!(out, "{pad}    for (int _k = 0; _k < _xtra; _k++) {{");
                        wl!(out, "{pad}        uint _i = _base + (uint)_k;");
                        if let Some(aux) = aux_src {
                            wl!(out, "{pad}        {dst}[_i] = {src}[_i] * {sc} * {aux}[_i];");
                        } else {
                            wl!(out, "{pad}        {dst}[_i] = {src}[_i] * {sc};");
                        }
                        wl!(out, "{pad}    }}");
                        wl!(out, "{pad}}}");
                    } else {
                        if let Some(aux) = aux_src {
                            wl!(
                                out,
                                "{pad}for (uint _i = {off}; _i < {en}; _i++) {dst}[_i] = {src}[_i] * {sc} * {aux}[_i];"
                            );
                        } else {
                            wl!(
                                out,
                                "{pad}for (uint _i = {off}; _i < {en}; _i++) {dst}[_i] = {src}[_i] * {sc};"
                            );
                        }
                    }
                },

                // ---- SIMD/threadgroup primitives ---------------------------
                Op::SimdReduce { value, op: rk } => {
                    let v = self.vname(vid, block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    match rk {
                        ReduceKind::Sum | ReduceKind::Mean => {
                            wl!(out, "{pad}float {v} = simd_sum(float({rv}));")
                        },
                        ReduceKind::Max => wl!(out, "{pad}float {v} = simd_max(float({rv}));"),
                        ReduceKind::Min => wl!(out, "{pad}float {v} = simd_min(float({rv}));"),
                        ReduceKind::Product => {
                            wl!(out, "{pad}float {v} = __mt_simd_product(float({rv}));")
                        },
                    }
                },

                Op::SimdShuffleXor { value, mask } => {
                    let v = self.vname(vid, block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    wl!(out, "{pad}auto {v} = simd_shuffle_xor({rv}, {mask}u);");
                },

                Op::SimdBroadcast { value, lane } => {
                    let v = self.vname(vid, block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    let rl = self.vname(Some(*lane), block, extra_names);
                    wl!(out, "{pad}auto {v} = simd_broadcast({rv}, {rl});");
                },

                Op::ThreadgroupAlloc { dtype, size, name } => {
                    let t = self.msl_type_name(*dtype);
                    hoists.push(format!("threadgroup {t} {name}[{size}];"));
                },

                // ThreadgroupLoad+StackLoad: identical array-indexed load.
                Op::ThreadgroupLoad { name, index } | Op::StackLoad { name, index } => {
                    let v = self.vname(vid, block, extra_names);
                    let iv = self.vname(Some(*index), block, extra_names);
                    wl!(out, "{pad}auto {v} = {name}[{iv}];");
                },

                // ThreadgroupStore+StackStore: identical array-indexed store.
                Op::ThreadgroupStore { name, index, value }
                | Op::StackStore { name, index, value } => {
                    let iv = self.vname(Some(*index), block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    wl!(out, "{pad}{name}[{iv}] = {rv};");
                },

                // ---- per-thread stack alloc / barrier ----
                Op::StackAlloc { dtype, size, name } => {
                    let t = self.msl_type_name(*dtype);
                    // No `threadgroup` qualifier — Metal places small
                    // fixed-size local arrays in per-thread registers
                    // (with spill to thread-local memory if they don't fit).
                    hoists.push(format!("{t} {name}[{size}];"));
                },

                Op::Barrier => {
                    wl!(out, "{pad}threadgroup_barrier(mem_flags::mem_threadgroup);");
                },

                Op::SimdgroupBarrier => {
                    wl!(out, "{pad}simdgroup_barrier(mem_flags::mem_none);");
                },

                // ---- simdgroup matrix ops ---------------------------------
                Op::SimdLaneId => {
                    let v = self.vname(vid, block, extra_names);
                    wl!(out, "{pad}uint {v} = simd_lane;");
                },

                Op::SimdGroupId => {
                    let v = self.vname(vid, block, extra_names);
                    wl!(out, "{pad}uint {v} = simd_group;");
                },

                Op::SimdgroupAlloc { dtype, m, n } => {
                    let t = self.msl_type_name(*dtype);
                    let v = self.vname(vid, block, extra_names);
                    hoists.push(format!("simdgroup_matrix<{t}, {m}, {n}> {v};"));
                },

                Op::SimdgroupElemLoad { value, index } => {
                    let v = self.vname(vid, block, extra_names);
                    let sv = self.vname(Some(*value), block, extra_names);
                    wl!(out, "{pad}auto {v} = {sv}.thread_elements()[{index}];");
                },

                Op::SimdgroupElemStore { value, index, data } => {
                    let sv = self.vname(Some(*value), block, extra_names);
                    let dv = self.vname(Some(*data), block, extra_names);
                    wl!(out, "{pad}{sv}.thread_elements()[{index}] = {dv};");
                },

                Op::SimdgroupLoad { dest, tg, offset, stride, transpose } => {
                    // HW-fused fragment load: one MSL `simdgroup_load`
                    // instruction issues a coalesced fetch across all 32
                    // lanes of the simdgroup, with HW swizzle, bypassing
                    // the per-lane bank-conflict scatter of repeated
                    // `simdgroup_elem_store(frag, idx, threadgroup_load(...))`.
                    // `ulong2(0,0)` origin mirrors MLX/llama.cpp usage
                    // (steel_gemm/qmm_t). `transpose=true` reads a row-major
                    // `[N, K]` tile as `[K, N]` — used for B operand of
                    // `C = A * B` MMA in the qmm_t pattern.
                    let dv = self.vname(Some(*dest), block, extra_names);
                    let off = self.vname(Some(*offset), block, extra_names);
                    let tflag = if *transpose { "true" } else { "false" };
                    wl!(
                        out,
                        "{pad}simdgroup_load({dv}, &{tg}[{off}], {stride}, ulong2(0, 0), {tflag});"
                    );
                },

                Op::SimdgroupMatMul { a, b, c } => {
                    let av = self.vname(Some(*a), block, extra_names);
                    let bv = self.vname(Some(*b), block, extra_names);
                    let cv = self.vname(Some(*c), block, extra_names);
                    // simdgroup_multiply_accumulate(dst, A, B, C) — 4 args:
                    //   dst = result, C = addend (both are the accumulator cv)
                    wl!(out, "{pad}simdgroup_multiply_accumulate({cv}, {av}, {bv}, {cv});");
                },

                // ---- simd prefix scan -------------------------------------
                Op::SimdScan { value, op: rk, exclusive } => {
                    let v = self.vname(vid, block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    let prefix = if *exclusive { "exclusive" } else { "inclusive" };
                    match rk {
                        ReduceKind::Sum | ReduceKind::Mean => {
                            wl!(out, "{pad}float {v} = simd_prefix_{prefix}_sum({rv});")
                        },
                        ReduceKind::Product => {
                            wl!(out, "{pad}float {v} = simd_prefix_{prefix}_product({rv});")
                        },
                        ReduceKind::Max => {
                            wl!(out, "{pad}float {v} = simd_prefix_{prefix}_max({rv});")
                        },
                        ReduceKind::Min => {
                            wl!(out, "{pad}float {v} = simd_prefix_{prefix}_min({rv});")
                        },
                    }
                },

                // ---- mutable local variables ----------------------------
                Op::DeclareLocal { name, value } => {
                    let rv = self.vname(Some(*value), block, extra_names);
                    wl!(out, "{pad}auto __ml_{name} = {rv};");
                },

                Op::SetLocal { name, value } => {
                    let rv = self.vname(Some(*value), block, extra_names);
                    wl!(out, "{pad}__ml_{name} = {rv};");
                },

                // ---- arg reduce -----------------------------------------
                Op::ArgReduce { value, axis, op: rk } => {
                    let v = self.vname(vid, block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    wl!(out, "{pad}// ArgReduce(v{}, axis={axis}, {rk:?}): lowering needed", rv);
                    wl!(out, "{pad}auto {v} = 0;");
                },
            }
        }

        Ok(())
    }
}

// FMA recognition lives in `passes::fma_fusion::FmaFusionPass` (IR-level
// rewrite of `Add(Mul(a, b), c)` → `Op::Fma { a, b, c }`).  The pre-#209/3
// emit-time peephole + per-kernel skip-set lived here; both are deleted.
