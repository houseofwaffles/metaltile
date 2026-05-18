//! Op-by-op block emission.
//!
//! Walks a `Block`'s op list and emits MSL statements for each IR op.

use std::{collections::BTreeMap, fmt::Write};

use metaltile_core::{
    dtype::DType,
    ir::{BinOpKind, Block, BlockId, Kernel, KernelMode, Op, ReduceKind, ValueId},
};

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
        all_blocks: &BTreeMap<BlockId, Block>,
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
                    // `simd_id` is a DSL alias for the simdgroup index; the
                    // Metal kernel parameter is named `simd_group`.
                    let src = if src.as_str() == "simd_id" { "simd_group" } else { src.as_str() };
                    let src_dtype = kernel.params.iter().find(|p| p.name == src).map(|p| p.dtype);
                    let promote_bf16 = self.config.native_bfloat
                        && src_dtype == Some(DType::BF16)
                        && matches!(
                            kernel.mode,
                            KernelMode::Elementwise | KernelMode::SimdGroup2D
                        );
                    if indices.is_empty() {
                        if promote_bf16 {
                            wl!(out, "{pad}float {v} = float({src});");
                        } else {
                            wl!(out, "{pad}auto {v} = {src};");
                        }
                    } else {
                        let idx = self.emit_idx(indices, block, extra_names, kernel, src);
                        if promote_bf16 {
                            wl!(out, "{pad}float {v} = float({src}[{idx}]);");
                        } else {
                            wl!(out, "{pad}auto {v} = {src}[{idx}];");
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
                Op::BinOp { op, lhs, rhs } => {
                    let v = self.vname(vid, block, extra_names);
                    let l = self.vname(Some(*lhs), block, extra_names);
                    let r = self.vname(Some(*rhs), block, extra_names);
                    let result_is_float = vid
                        .and_then(|id| type_env.get(&id))
                        .map(|tv| matches!(tv.dtype, DType::F32 | DType::F16 | DType::BF16))
                        .unwrap_or(false);
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
                        BinOpKind::Add => {
                            // FMA recognition: Mul + Add → fma() (floats only)
                            match (
                                result_is_float,
                                try_get_mul(*lhs, block),
                                try_get_mul(*rhs, block),
                            ) {
                                (true, Some((ml, mr)), None) => wl!(
                                    out,
                                    "{pad}auto {v} = fma({ml}, {mr}, {r});",
                                    ml = self.vname(Some(ml), block, extra_names),
                                    mr = self.vname(Some(mr), block, extra_names),
                                ),
                                (true, None, Some((ml, mr))) => wl!(
                                    out,
                                    "{pad}auto {v} = fma({ml}, {mr}, {l});",
                                    ml = self.vname(Some(ml), block, extra_names),
                                    mr = self.vname(Some(mr), block, extra_names),
                                ),
                                _ => wl!(out, "{pad}auto {v} = {l} + {r};"),
                            }
                        },
                        BinOpKind::Sub => {
                            // FMA recognition: Mul - X → fma() (floats only)
                            match (result_is_float, try_get_mul(*lhs, block)) {
                                (true, Some((ml, mr))) => wl!(
                                    out,
                                    "{pad}auto {v} = fma({ml}, {mr}, -{r});",
                                    ml = self.vname(Some(ml), block, extra_names),
                                    mr = self.vname(Some(mr), block, extra_names),
                                ),
                                _ => wl!(out, "{pad}auto {v} = {l} - {r};"),
                            }
                        },
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
                                    Op::Activation { kind, .. } =>
                                        format!("{}({})", kind.msl_fn(), expr),
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
                    wl!(out, "{pad}{type_decl} {v} = {};", self.emit_cast_expr(*dtype, &rv));
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
                        inner_names.insert(ValueId::new(var.as_u32() + 1000), vn.clone());
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
                            wl!(out, "{pad}bfloat {v} = bfloat({expr});");
                        } else if result_is_bf16 {
                            let bf_type = self.msl_type_name(DType::BF16);
                            wl!(out, "{pad}{bf_type} {v} = {expr};");
                        } else {
                            wl!(out, "{pad}auto {v} = {expr};");
                        }
                    }
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
                        (8, DType::F16) => "half8".into(),
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
                        (8, DType::F16) => "half8".into(),
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
                Op::Atomic { op: ak, dst, index, value } => {
                    let iv = self.vname(Some(*index), block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    wl!(out, "{pad}{}({dst} + {iv}, {rv}, memory_order_relaxed);", ak.msl_fn());
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

                // ---- flash attention / sliding window attention ------------
                Op::FlashAttention { .. } | Op::SlidingWindowAttention { .. } => {
                    wl!(out, "{pad}// Attention: needs lowering pass");
                },

                // ---- high-level norms / mlp blocks -------------------------
                Op::RmsNorm { .. } | Op::GatedMlp { .. } => {
                    wl!(out, "{pad}// High-level op: needs lowering pass");
                },

                // ---- dequantize --------------------------------------------
                Op::Dequantize { .. } => {
                    wl!(out, "{pad}// Dequantize: needs lowering pass");
                },

                // ---- SIMD/threadgroup primitives ---------------------------
                Op::SimdReduce { value, op: rk } => {
                    let v = self.vname(vid, block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    match rk {
                        ReduceKind::Sum | ReduceKind::Mean =>
                            wl!(out, "{pad}float {v} = simd_sum(float({rv}));"),
                        ReduceKind::Max => wl!(out, "{pad}float {v} = simd_max(float({rv}));"),
                        ReduceKind::Min => wl!(out, "{pad}float {v} = simd_min(float({rv}));"),
                        ReduceKind::Product =>
                            wl!(out, "{pad}float {v} = __mt_simd_product(float({rv}));"),
                    }
                },

                Op::ThreadgroupAlloc { dtype, size, name } => {
                    let t = self.msl_type_name(*dtype);
                    hoists.push(format!("threadgroup {t} {name}[{size}];"));
                },

                Op::ThreadgroupLoad { name, index } => {
                    let v = self.vname(vid, block, extra_names);
                    let iv = self.vname(Some(*index), block, extra_names);
                    wl!(out, "{pad}auto {v} = {name}[{iv}];");
                },

                Op::ThreadgroupStore { name, index, value } => {
                    let iv = self.vname(Some(*index), block, extra_names);
                    let rv = self.vname(Some(*value), block, extra_names);
                    wl!(out, "{pad}{name}[{iv}] = {rv};");
                },

                Op::Barrier => {
                    wl!(out, "{pad}threadgroup_barrier(mem_flags::mem_threadgroup);");
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
                        ReduceKind::Sum | ReduceKind::Mean =>
                            wl!(out, "{pad}float {v} = simd_prefix_{prefix}_sum({rv});"),
                        ReduceKind::Product =>
                            wl!(out, "{pad}float {v} = simd_prefix_{prefix}_product({rv});"),
                        ReduceKind::Max =>
                            wl!(out, "{pad}float {v} = simd_prefix_{prefix}_max({rv});"),
                        ReduceKind::Min =>
                            wl!(out, "{pad}float {v} = simd_prefix_{prefix}_min({rv});"),
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

// ---------------------------------------------------------------------------
// FMA recognition helper
// ---------------------------------------------------------------------------

/// If `vid` is defined by a `BinOp(Mul, a, b)` in `block`, return `Some((a, b))`.
fn try_get_mul(vid: ValueId, block: &Block) -> Option<(ValueId, ValueId)> {
    for (i, op) in block.ops.iter().enumerate() {
        if block.results.get(i) == Some(&Some(vid))
            && let Op::BinOp { op: BinOpKind::Mul, lhs, rhs } = op
        {
            return Some((*lhs, *rhs));
        }
    }
    None
}
