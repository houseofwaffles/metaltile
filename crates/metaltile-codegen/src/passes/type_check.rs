//! Type & Shape Checking — validate IR before MSL emission.
//!
//! Validates that the IR is well-formed before code generation:
//! - Dot operands must be 2D with matching K dimension
//! - Load index count must equal the tensor's rank
//! - Reduce axis must be plausible (≤ 3)
//! - Slice lengths must be positive and offsets non-negative
//! - Store target must be an output parameter
//!
//! Also performs forward type inference: produces a [`TypeEnv`] mapping every
//! [`ValueId`] to its `(DType, Shape)` so the MSL emitter can emit correct
//! declarations and casts.
//!
//! This is a verification pass — it catches IR bugs early with clear error
//! messages rather than letting them manifest as invalid MSL or GPU crashes.

use std::collections::BTreeMap;

use metaltile_core::{
    dtype::DType,
    error::{Error, Result},
    ir::{BinOpKind, Block, BlockId, Kernel, Op, Param, ReduceKind, ValueId},
    shape::{Dim, Shape},
};

pub struct TypeCheckPass;

impl super::Pass for TypeCheckPass {
    fn name(&self) -> &str { "type_check" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        // Validate that every ConstExpr dim in params has a matching constexpr decl.
        for p in &kernel.params {
            for dim in p.shape.iter() {
                if let Dim::ConstExpr(ce) = dim {
                    let found = kernel.constexprs.iter().any(|d| d.name == *ce);
                    if !found {
                        return Err(Error::Validation(format!(
                            "param '{}' uses ConstExpr '{}' in shape but no constexpr decl found",
                            p.name,
                            ce.name()
                        )));
                    }
                }
            }
        }

        let type_env = infer_types(kernel)?;
        check_block(&kernel.body, kernel, &type_env)?;

        let block_ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();
        for bid in block_ids {
            if bid == kernel.body.id {
                continue;
            }
            if let Some(block) = kernel.blocks.get(&bid) {
                let block = block.clone();
                check_block(&block, kernel, &type_env)?;
            }
        }
        Ok(())
    }
}

fn check_block(block: &Block, kernel: &Kernel, env: &TypeEnv) -> Result<()> {
    for (op_idx, op) in block.ops.iter().enumerate() {
        let result = block.results.get(op_idx).and_then(|x| *x);
        check_op(op, kernel, env).map_err(|err| add_op_context(err, block, op_idx, op, result))?;
    }
    Ok(())
}

fn check_op(op: &Op, kernel: &Kernel, env: &TypeEnv) -> Result<()> {
    match op {
        Op::Dot { .. } => {
            let tensors: Vec<_> = kernel.params.iter().filter(|p| p.shape.rank() == 2).collect();
            if tensors.len() < 2 {
                return Err(Error::Validation(
                    "Op::Dot requires at least two 2D tensor parameters [M,K] and [K,N]".into(),
                ));
            }
            let k_a = tensors[0].shape.dim(1);
            let k_b = tensors[1].shape.dim(0);
            if k_a != k_b && k_a.is_some() && k_b.is_some() {
                return Err(Error::ShapeMismatch {
                    expected: format!("{:?}", k_a),
                    actual: format!("{:?}", k_b),
                });
            }
        },

        Op::Load { src, indices, .. } => {
            if let Some(param) = kernel.params.iter().find(|p| &p.name == src) {
                let rank = param.shape.rank();
                if rank > 0 && indices.len() != rank {
                    return Err(Error::Validation(format!(
                        "Op::Load '{src}': index count {} != tensor rank {rank}",
                        indices.len()
                    )));
                }
            }
        },

        Op::Store { dst, indices, .. } => {
            let is_output = kernel.params.iter().any(|p| &p.name == dst && p.is_output);
            if !is_output {
                return Err(Error::Validation(format!(
                    "Op::Store target '{dst}' must be an output parameter"
                )));
            }
            if let Some(param) = kernel.params.iter().find(|p| &p.name == dst) {
                let rank = param.shape.rank();
                if rank > 0 && indices.len() != rank {
                    return Err(Error::Validation(format!(
                        "Op::Store '{dst}': index count {} != tensor rank {rank}",
                        indices.len()
                    )));
                }
            }
        },

        Op::Reduce { axis, .. } if *axis > 3 => {
            return Err(Error::Validation(format!("Op::Reduce: axis {axis} > 3 is implausible")));
        },

        Op::StrideReduce { src, offset, stride, end, secondary_src, secondary_base, .. } => {
            require_param(kernel, src)?;
            require_scalar_integer(env, *offset, "offset")?;
            require_scalar_integer(env, *stride, "stride")?;
            require_scalar_integer(env, *end, "end")?;
            if let Some(sec_src) = secondary_src {
                require_param(kernel, sec_src)?;
                if secondary_base.is_none() {
                    return Err(Error::Validation(
                        "Op::StrideReduce: secondary_src requires secondary_base".into(),
                    ));
                }
            } else if secondary_base.is_some() {
                return Err(Error::Validation(
                    "Op::StrideReduce: secondary_base requires secondary_src".into(),
                ));
            }
        },

        Op::Gather { src, indices, axis } => {
            let param = require_param(kernel, src)?;
            validate_axis(*axis, &param.shape, "Op::Gather")?;
            if *axis != 0 {
                return Err(Error::Validation(
                    "Op::Gather: only axis 0 is supported by the current emitter".into(),
                ));
            }
            require_scalar_integer(env, *indices, "indices")?;
        },

        Op::Scatter { dst, indices, value, axis } => {
            let param = require_output_param(kernel, dst)?;
            validate_axis(*axis, &param.shape, "Op::Scatter")?;
            if *axis != 0 {
                return Err(Error::Validation(
                    "Op::Scatter: only axis 0 is supported by the current emitter".into(),
                ));
            }
            require_scalar_integer(env, *indices, "indices")?;
            let value_tv = require_typed_value(env, *value, "value")?;
            if value_tv.shape.rank() != 0 {
                return Err(Error::Validation(format!(
                    "Op::Scatter: value must be scalar for the current emitter, got shape {}",
                    value_tv.shape
                )));
            }
            if value_tv.dtype != param.dtype {
                return Err(Error::Validation(format!(
                    "Op::Scatter: value dtype {} does not match destination dtype {}",
                    value_tv.dtype, param.dtype
                )));
            }
        },

        Op::Scan { value, axis, op, exclusive } => {
            let value_tv = require_typed_value(env, *value, "value")?;
            validate_axis(*axis, &value_tv.shape, "Op::Scan")?;
            if *axis != 0 {
                return Err(Error::Validation(
                    "Op::Scan: only axis 0 is supported by the current emitter".into(),
                ));
            }
            if value_tv.shape.rank() != 0 {
                return Err(Error::Validation(format!(
                    "Op::Scan: value must be scalar for the current emitter, got shape {}",
                    value_tv.shape
                )));
            }
            if matches!(op, ReduceKind::Mean) {
                return Err(Error::Validation(
                    "Op::Scan: mean is not a supported scan operator".into(),
                ));
            }
            if *exclusive && !matches!(op, ReduceKind::Sum) {
                return Err(Error::Validation(
                    "Op::Scan: exclusive scan is currently supported only for sum".into(),
                ));
            }
        },

        Op::Slice { ranges, .. } =>
            for (ax, start, len) in ranges {
                if *len <= 0 {
                    return Err(Error::Validation(format!(
                        "Op::Slice: axis {ax} has non-positive length {len}"
                    )));
                }
                if *start < 0 {
                    return Err(Error::Validation(format!(
                        "Op::Slice: axis {ax} has negative start offset {start}"
                    )));
                }
            },

        Op::UnaryOp { .. }
        | Op::Activation { .. }
        | Op::Select { .. }
        | Op::Broadcast { .. }
        | Op::Splat { .. } => {},

        _ => {},
    }
    Ok(())
}

fn add_op_context(
    err: Error,
    block: &Block,
    op_idx: usize,
    op: &Op,
    result: Option<ValueId>,
) -> Error {
    let mut ctx = format!("block {} op #{} ({})", block.id.as_u32(), op_idx, op_name(op));
    if let Some(vid) = result {
        ctx.push_str(&format!(" -> {vid}"));
    }
    let detail = match err {
        Error::ShapeMismatch { expected, actual } => {
            format!("shape mismatch: expected {expected}, got {actual}")
        },
        Error::UnresolvedConstExpr(msg) => format!("unresolved constexpr: {msg}"),
        Error::UnknownDimension(msg) => format!("dimension is not statically known: {msg}"),
        Error::InvalidRank { expected, actual } => {
            format!("invalid rank: expected {expected}, got {actual}")
        },
        Error::Validation(msg) => msg,
        Error::UnknownValue(msg) => format!("unknown value reference: {msg}"),
        Error::Internal(msg) => format!("internal error: {msg}"),
    };
    Error::Validation(format!("{ctx}: {detail}"))
}

fn op_name(op: &Op) -> &'static str {
    match op {
        Op::ProgramId { .. } => "ProgramId",
        Op::Const { .. } => "Const",
        Op::Arange { .. } => "Arange",
        Op::Load { .. } => "Load",
        Op::Store { .. } => "Store",
        Op::BinOp { .. } => "BinOp",
        Op::Dot { .. } => "Dot",
        Op::Reduce { .. } => "Reduce",
        Op::StrideReduce { .. } => "StrideReduce",
        Op::Cast { .. } => "Cast",
        Op::Loop { .. } => "Loop",
        Op::If { .. } => "If",
        Op::Zeros { .. } => "Zeros",
        Op::Transpose { .. } => "Transpose",
        Op::ExpandDims { .. } => "ExpandDims",
        Op::Reshape { .. } => "Reshape",
        Op::Cat { .. } => "Cat",
        Op::Slice { .. } => "Slice",
        Op::InlineMsl { .. } => "InlineMsl",
        Op::FlashAttention { .. } => "FlashAttention",
        Op::SlidingWindowAttention { .. } => "SlidingWindowAttention",
        Op::RmsNorm { .. } => "RmsNorm",
        Op::GatedMlp { .. } => "GatedMlp",
        Op::UnaryOp { .. } => "UnaryOp",
        Op::Activation { .. } => "Activation",
        Op::Select { .. } => "Select",
        Op::Broadcast { .. } => "Broadcast",
        Op::Splat { .. } => "Splat",
        Op::FusedElementwise { .. } => "FusedElementwise",
        Op::VectorLoad { .. } => "VectorLoad",
        Op::VectorStore { .. } => "VectorStore",
        Op::VectorExtract { .. } => "VectorExtract",
        Op::Gather { .. } => "Gather",
        Op::Scatter { .. } => "Scatter",
        Op::Atomic { .. } => "Atomic",
        Op::Scan { .. } => "Scan",
        Op::StrideStore { .. } => "StrideStore",
        Op::Dequantize { .. } => "Dequantize",
        Op::SimdReduce { .. } => "SimdReduce",
        Op::SimdShuffleXor { .. } => "SimdShuffleXor",
        Op::SimdBroadcast { .. } => "SimdBroadcast",
        Op::ThreadgroupAlloc { .. } => "ThreadgroupAlloc",
        Op::ThreadgroupLoad { .. } => "ThreadgroupLoad",
        Op::ThreadgroupStore { .. } => "ThreadgroupStore",
        Op::Barrier => "Barrier",
        Op::SimdgroupAlloc { .. } => "SimdgroupAlloc",
        Op::SimdgroupElemLoad { .. } => "SimdgroupElemLoad",
        Op::SimdgroupElemStore { .. } => "SimdgroupElemStore",
        Op::SimdgroupMatMul { .. } => "SimdgroupMatMul",
        Op::SimdScan { .. } => "SimdScan",
        Op::SimdLaneId => "SimdLaneId",
        Op::SimdGroupId => "SimdGroupId",
        Op::DeclareLocal { .. } => "DeclareLocal",
        Op::SetLocal { .. } => "SetLocal",
        Op::ArgReduce { .. } => "ArgReduce",
        Op::StrideScan { .. } => "StrideScan",
        Op::StrideArgReduce { .. } => "StrideArgReduce",
    }
}

fn require_param<'a>(kernel: &'a Kernel, name: &str) -> Result<&'a Param> {
    kernel
        .params
        .iter()
        .find(|p| p.name == name)
        .ok_or_else(|| Error::UnknownValue(format!("tensor parameter '{name}'")))
}

fn require_output_param<'a>(kernel: &'a Kernel, name: &str) -> Result<&'a Param> {
    let param = require_param(kernel, name)?;
    if !param.is_output {
        return Err(Error::Validation(format!("parameter '{name}' must be an output tensor")));
    }
    Ok(param)
}

fn require_typed_value<'a>(env: &'a TypeEnv, vid: ValueId, role: &str) -> Result<&'a TypedValue> {
    env.get(&vid).ok_or_else(|| Error::UnknownValue(format!("{role} value {vid}")))
}

fn require_scalar_integer(env: &TypeEnv, vid: ValueId, role: &str) -> Result<()> {
    let tv = require_typed_value(env, vid, role)?;
    if tv.shape.rank() != 0 {
        return Err(Error::Validation(format!("{role} must be scalar, got shape {}", tv.shape)));
    }
    if !tv.dtype.is_integer() {
        return Err(Error::Validation(format!(
            "{role} must use an integer dtype, got {}",
            tv.dtype
        )));
    }
    Ok(())
}

fn validate_axis(axis: u32, shape: &Shape, op_name: &str) -> Result<()> {
    let logical_rank = shape.rank().max(1);
    if axis as usize >= logical_rank {
        return Err(Error::Validation(format!(
            "{op_name}: axis {axis} out of bounds for rank {logical_rank}"
        )));
    }
    Ok(())
}

/// Inferred type and shape for an IR value.
#[derive(Debug, Clone)]
pub struct TypedValue {
    pub dtype: DType,
    pub shape: Shape,
}

/// Type environment: maps every `ValueId` in the kernel to its inferred type.
pub type TypeEnv = BTreeMap<ValueId, TypedValue>;

/// Forward type inference across all blocks in a kernel.
/// Returns a `TypeEnv` that the MSL emitter can query for every `ValueId`.
pub fn infer_types(kernel: &Kernel) -> Result<TypeEnv> {
    let mut env = TypeEnv::new();
    infer_block(&kernel.body, kernel, &kernel.blocks, &mut env)?;
    for block in kernel.blocks.values() {
        // Don't re-infer the body block (bid 0).
        if block.id != kernel.body.id {
            infer_block(block, kernel, &kernel.blocks, &mut env)?;
        }
    }
    Ok(env)
}

/// Determine the result DType of a BinOp given its input DType.
/// Comparison ops always produce Bool regardless of input type.
fn binop_result_dtype(op: BinOpKind, input_dtype: DType) -> DType {
    match op {
        BinOpKind::CmpLt
        | BinOpKind::CmpGt
        | BinOpKind::CmpLe
        | BinOpKind::CmpGe
        | BinOpKind::CmpEq
        | BinOpKind::CmpNe => DType::Bool,
        _ => input_dtype,
    }
}

/// Extract the first input ValueId from a sub-op (for type propagation).
fn first_input_shape(op: &Op) -> Option<ValueId> {
    match op {
        Op::Cast { value, .. }
        | Op::UnaryOp { value, .. }
        | Op::Activation { value, .. }
        | Op::BinOp { lhs: value, .. }
        | Op::Select { on_true: value, .. } => Some(*value),
        _ => None,
    }
}

fn infer_block(
    block: &Block,
    kernel: &Kernel,
    all_blocks: &BTreeMap<BlockId, Block>,
    env: &mut TypeEnv,
) -> Result<()> {
    for (op_idx, op) in block.ops.iter().enumerate() {
        let Some(vid) = block.results.get(op_idx).and_then(|x| *x) else {
            continue;
        };

        match op {
            // ---- indexing ------------------------------------------
            Op::ProgramId { .. } => {
                env.insert(vid, TypedValue { dtype: DType::U32, shape: Shape::scalar() });
            },
            Op::Const { .. } => {
                env.insert(vid, TypedValue { dtype: DType::I32, shape: Shape::scalar() });
            },
            Op::Arange { len, .. } => {
                env.insert(vid, TypedValue {
                    dtype: DType::U32,
                    shape: Shape::new([Dim::ConstExpr(len.clone())]),
                });
            },

            // ---- data movement ------------------------------------
            Op::Load { src, .. } => {
                if let Some(param) = kernel.params.iter().find(|p| &p.name == src) {
                    env.insert(vid, TypedValue { dtype: param.dtype, shape: param.shape.clone() });
                } else if kernel.constexprs.iter().any(|ce| ce.name.name() == src.as_str()) {
                    // Constexpr parameters are `constant uint` scalars.
                    env.insert(vid, TypedValue { dtype: DType::U32, shape: Shape::scalar() });
                }
            },

            Op::Broadcast { value: src_v, shape, .. } => {
                let dtype = env.get(src_v).map(|tv| tv.dtype).unwrap_or(DType::F32);
                env.insert(vid, TypedValue { dtype, shape: shape.clone() });
            },

            Op::Transpose { value } =>
                if let Some(tv) = env.get(value) {
                    let new_shape = if tv.shape.rank() >= 2 {
                        let r = tv.shape.rank();
                        let mut dims: Vec<Dim> = tv.shape.iter().cloned().collect();
                        dims.swap(r - 2, r - 1);
                        Shape::new(dims)
                    } else {
                        tv.shape.clone()
                    };
                    env.insert(vid, TypedValue { dtype: tv.dtype, shape: new_shape });
                },

            Op::Slice { value, ranges } =>
                if let Some(tv) = env.get(value) {
                    let mut dims: Vec<Dim> = tv.shape.iter().cloned().collect();
                    for &(axis, _start, len) in ranges {
                        if (axis as usize) < dims.len() && len > 0 {
                            dims[axis as usize] = Dim::Known(len as usize);
                        }
                    }
                    env.insert(vid, TypedValue { dtype: tv.dtype, shape: Shape::new(dims) });
                },

            // ---- compute -------------------------------------------
            Op::BinOp { lhs, .. } =>
                if let Some(tv) = env.get(lhs) {
                    env.insert(vid, TypedValue { dtype: tv.dtype, shape: tv.shape.clone() });
                },

            Op::UnaryOp { value, .. } | Op::Activation { value, .. } => {
                if let Some(tv) = env.get(value) {
                    env.insert(vid, TypedValue { dtype: tv.dtype, shape: tv.shape.clone() });
                }
            },

            Op::Select { on_true, .. } =>
                if let Some(tv) = env.get(on_true) {
                    env.insert(vid, TypedValue { dtype: tv.dtype, shape: tv.shape.clone() });
                },

            Op::Cast { value, dtype } => {
                let shape = env.get(value).map(|tv| tv.shape.clone()).unwrap_or(Shape::scalar());
                env.insert(vid, TypedValue { dtype: *dtype, shape });
            },

            Op::Reduce { value, axis, .. } =>
                if let Some(tv) = env.get(value) {
                    let rank = tv.shape.rank();
                    let new_shape = if rank <= 1 || *axis as usize >= rank {
                        Shape::scalar()
                    } else {
                        let dims: Vec<Dim> = tv
                            .shape
                            .iter()
                            .enumerate()
                            .filter(|&(d, _)| d != *axis as usize)
                            .map(|(_, dim)| dim.clone())
                            .collect();
                        if dims.is_empty() { Shape::scalar() } else { Shape::new(dims) }
                    };
                    env.insert(vid, TypedValue { dtype: DType::F32, shape: new_shape });
                },

            Op::StrideReduce { dtype, .. } => {
                env.insert(vid, TypedValue { dtype: *dtype, shape: Shape::scalar() });
            },

            Op::Dot { a, b } => {
                let (shape_a, shape_b) =
                    (env.get(a).map(|tv| tv.shape.clone()), env.get(b).map(|tv| tv.shape.clone()));
                let (m, n) = match (shape_a.as_ref(), shape_b.as_ref()) {
                    (Some(sa), Some(sb)) if sa.rank() == 2 && sb.rank() == 2 => (
                        sa.dim(0).cloned().unwrap_or(Dim::Any),
                        sb.dim(1).cloned().unwrap_or(Dim::Any),
                    ),
                    _ => (Dim::Any, Dim::Any),
                };
                let dtype = env.get(a).map(|tv| tv.dtype).unwrap_or(DType::F16);
                env.insert(vid, TypedValue { dtype, shape: Shape::new([m, n]) });
            },

            Op::Zeros { dtype, shape } | Op::Splat { dtype, shape, .. } => {
                env.insert(vid, TypedValue { dtype: *dtype, shape: shape.clone() });
            },

            // ---- control flow --------------------------------------
            Op::Loop { var, body: bid, .. } => {
                // Register the loop variable as uint before processing the loop body
                // so that index arithmetic like `_r * lsize + lid` infers as uint.
                // The body_parser encodes the loop variable as VarId(N)+1000.
                let loop_var_vid = ValueId::new(var.as_u32() + 1000);
                env.insert(loop_var_vid, TypedValue { dtype: DType::U32, shape: Shape::scalar() });
                // Recurse into the loop body.
                if let Some(loop_block) = all_blocks.get(bid) {
                    infer_block(loop_block, kernel, all_blocks, env)?;
                }
            },

            // ---- escape hatch / high-level -------------------------
            Op::InlineMsl { outputs, .. } =>
                for (oi, slot) in outputs.iter().enumerate() {
                    let out_vid = ValueId::new(vid.as_u32() + oi as u32);
                    env.insert(out_vid, TypedValue { dtype: slot.dtype, shape: Shape::scalar() });
                },

            Op::Dequantize { .. } => {
                // Dequantize produces a scalar f16 value (the dequantized weight element).
                env.insert(vid, TypedValue { dtype: DType::F16, shape: Shape::scalar() });
            },

            Op::SimdReduce { value, .. } => {
                // Same type as input
                if let Some(tv) = env.get(value).cloned() {
                    env.insert(vid, tv);
                } else {
                    env.insert(vid, TypedValue { dtype: DType::F32, shape: Shape::scalar() });
                }
            },
            Op::SimdShuffleXor { value, .. } | Op::SimdBroadcast { value, .. } => {
                // Same scalar type as the input value; the cross-lane move
                // doesn't change dtype or shape.
                if let Some(tv) = env.get(value).cloned() {
                    env.insert(vid, tv);
                } else {
                    env.insert(vid, TypedValue { dtype: DType::F32, shape: Shape::scalar() });
                }
            },
            Op::Gather { src, indices, .. } => {
                let dtype = kernel
                    .params
                    .iter()
                    .find(|p| p.name == *src)
                    .map(|p| p.dtype)
                    .unwrap_or(DType::F32);
                let shape = env.get(indices).map(|tv| tv.shape.clone()).unwrap_or(Shape::scalar());
                env.insert(vid, TypedValue { dtype, shape });
            },
            Op::Scan { value, .. } =>
                if let Some(tv) = env.get(value).cloned() {
                    env.insert(vid, tv);
                } else {
                    env.insert(vid, TypedValue { dtype: DType::F32, shape: Shape::scalar() });
                },
            Op::ThreadgroupLoad { .. } => {
                env.insert(vid, TypedValue { dtype: DType::F32, shape: Shape::scalar() });
            },
            Op::ArgReduce { .. } | Op::StrideArgReduce { .. } => {
                env.insert(vid, TypedValue { dtype: DType::U32, shape: Shape::scalar() });
            },
            Op::StrideScan { .. } => {
                // Side-effect only — writes directly to dst buffer, no SSA result.
            },
            Op::FlashAttention { .. }
            | Op::SlidingWindowAttention { .. }
            | Op::RmsNorm { .. }
            | Op::GatedMlp { .. }
            | Op::Store { .. }
            | Op::VectorLoad { .. }
            | Op::VectorStore { .. }
            | Op::VectorExtract { .. }
            | Op::If { .. }
            | Op::ExpandDims { .. }
            | Op::Reshape { .. }
            | Op::Cat { .. }
            | Op::Scatter { .. }
            | Op::Atomic { .. }
            | Op::StrideStore { .. }
            | Op::ThreadgroupAlloc { .. }
            | Op::ThreadgroupStore { .. }
            | Op::Barrier
            | Op::DeclareLocal { .. }
            | Op::SetLocal { .. }
            | Op::SimdgroupMatMul { .. }
            | Op::SimdgroupElemStore { .. } => {
                // No output value to type (or side-effect-only op).
            },
            Op::SimdgroupAlloc { .. } | Op::SimdgroupElemLoad { .. } | Op::SimdScan { .. } => {
                env.insert(vid, TypedValue { dtype: DType::F32, shape: Shape::scalar() });
            },
            Op::SimdLaneId | Op::SimdGroupId => {
                env.insert(vid, TypedValue { dtype: DType::U32, shape: Shape::scalar() });
            },

            Op::FusedElementwise { ops } => {
                // The final op determines the output type.
                // Walk the chain to build local types, then use the last op.
                let mut local_env = env.clone();
                for (si, sub_op) in ops.iter().enumerate() {
                    let sub_vid = ValueId::new(vid.as_u32() + si as u32);
                    match sub_op {
                        Op::Cast { dtype, .. } => {
                            let shape = if si == 0 {
                                first_input_shape(sub_op)
                                    .and_then(|iv| env.get(&iv))
                                    .map(|tv| tv.shape.clone())
                                    .unwrap_or(Shape::scalar())
                            } else {
                                local_env
                                    .get(&ValueId::new(vid.as_u32() + (si as u32 - 1)))
                                    .map(|tv| tv.shape.clone())
                                    .unwrap_or(Shape::scalar())
                            };
                            local_env.insert(sub_vid, TypedValue { dtype: *dtype, shape });
                        },
                        Op::UnaryOp { .. }
                        | Op::Activation { .. }
                        | Op::BinOp { .. }
                        | Op::Select { .. } => {
                            let shape = if si == 0 {
                                first_input_shape(sub_op)
                                    .and_then(|iv| env.get(&iv))
                                    .map(|tv| tv.shape.clone())
                                    .unwrap_or(Shape::scalar())
                            } else {
                                local_env
                                    .get(&ValueId::new(vid.as_u32() + (si as u32 - 1)))
                                    .map(|tv| tv.shape.clone())
                                    .unwrap_or(Shape::scalar())
                            };
                            let input_dtype = if si == 0 {
                                first_input_shape(sub_op)
                                    .and_then(|iv| env.get(&iv))
                                    .map(|tv| tv.dtype)
                                    .unwrap_or(DType::F32)
                            } else {
                                local_env
                                    .get(&ValueId::new(vid.as_u32() + (si as u32 - 1)))
                                    .map(|tv| tv.dtype)
                                    .unwrap_or(DType::F32)
                            };
                            let dtype = if let Op::BinOp { op, .. } = sub_op {
                                binop_result_dtype(*op, input_dtype)
                            } else if let Op::Select { on_true, .. } = sub_op {
                                // Select result type is determined by on_true/on_false,
                                // not the condition (which may be Bool from a prior CmpXX).
                                let raw = on_true.as_u32();
                                if raw & 0x8000_0000 != 0 {
                                    // internal sub-op ref within this chain
                                    let pos = raw & !0x8000_0000;
                                    local_env
                                        .get(&ValueId::new(vid.as_u32() + pos))
                                        .map(|tv| tv.dtype)
                                        .unwrap_or(input_dtype)
                                } else {
                                    env.get(on_true).map(|tv| tv.dtype).unwrap_or(input_dtype)
                                }
                            } else {
                                input_dtype
                            };
                            local_env.insert(sub_vid, TypedValue { dtype, shape });
                        },
                        _ => {},
                    }
                }
                let last_vid = ValueId::new(vid.as_u32() + (ops.len() as u32 - 1));
                if let Some(tv) = local_env.get(&last_vid) {
                    env.insert(vid, tv.clone());
                }
            },
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use metaltile_core::ir::{Kernel, Op, Param, ValueId};

    use super::*;
    use crate::passes::Pass;

    fn tensor_param(name: &str, dtype: DType, is_output: bool) -> Param {
        Param {
            name: name.into(),
            dtype,
            shape: Shape::scalar(),
            is_output,
            kind: Default::default(),
        }
    }

    #[test]
    fn stride_reduce_rejects_non_integer_stride_with_context() {
        let mut k = Kernel::new("stride_reduce_bad_stride");
        k.params.push(tensor_param("src", DType::F32, false));
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::Cast { value: ValueId::new(0), dtype: DType::F32 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 16 }, ValueId::new(2));
        k.body.push_op(
            Op::StrideReduce {
                src: "src".into(),
                offset: ValueId::new(0),
                stride: ValueId::new(1),
                end: ValueId::new(2),
                op: ReduceKind::Sum,
                dtype: DType::F32,
                transform: None,
                secondary_src: None,
                secondary_base: None,
            },
            ValueId::new(3),
        );

        let err = TypeCheckPass.run(&mut k).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("block 0 op #3 (StrideReduce) -> %3"));
        assert!(msg.contains("stride must use an integer dtype, got f32"));
    }

    #[test]
    fn gather_rejects_non_scalar_indices() {
        let mut k = Kernel::new("gather_bad_indices");
        k.params.push(tensor_param("src", DType::F32, false));
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(
            Op::Broadcast { value: ValueId::new(0), shape: Shape::new([Dim::Known(4)]) },
            ValueId::new(1),
        );
        k.body.push_op(
            Op::Gather { src: "src".into(), indices: ValueId::new(1), axis: 0 },
            ValueId::new(2),
        );

        let err = TypeCheckPass.run(&mut k).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("block 0 op #2 (Gather) -> %2"));
        assert!(msg.contains("indices must be scalar"));
    }

    #[test]
    fn scatter_rejects_mismatched_value_dtype() {
        let mut k = Kernel::new("scatter_bad_value");
        k.params.push(tensor_param("dst", DType::F32, true));
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::Cast { value: ValueId::new(0), dtype: DType::F16 }, ValueId::new(1));
        k.body.push_op_no_result(Op::Scatter {
            dst: "dst".into(),
            indices: ValueId::new(0),
            value: ValueId::new(1),
            axis: 0,
        });

        let err = TypeCheckPass.run(&mut k).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("block 0 op #2 (Scatter)"));
        assert!(msg.contains("value dtype f16 does not match destination dtype f32"));
    }

    #[test]
    fn scan_rejects_axis_out_of_bounds() {
        let mut k = Kernel::new("scan_bad_axis");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(
            Op::Broadcast { value: ValueId::new(0), shape: Shape::new([Dim::Known(4)]) },
            ValueId::new(1),
        );
        k.body.push_op(
            Op::Scan { value: ValueId::new(1), axis: 1, op: ReduceKind::Sum, exclusive: false },
            ValueId::new(2),
        );

        let err = TypeCheckPass.run(&mut k).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("block 0 op #2 (Scan) -> %2"));
        assert!(msg.contains("Op::Scan: axis 1 out of bounds for rank 1"));
    }

    #[test]
    fn infer_types_tracks_stride_reduce_gather_and_scan() {
        let mut k = Kernel::new("inference_supported_ops");
        k.params.push(tensor_param("src", DType::F16, false));
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 16 }, ValueId::new(2));
        k.body.push_op(
            Op::StrideReduce {
                src: "src".into(),
                offset: ValueId::new(0),
                stride: ValueId::new(1),
                end: ValueId::new(2),
                op: ReduceKind::Sum,
                dtype: DType::F32,
                transform: None,
                secondary_src: None,
                secondary_base: None,
            },
            ValueId::new(3),
        );
        k.body.push_op(
            Op::Gather { src: "src".into(), indices: ValueId::new(0), axis: 0 },
            ValueId::new(4),
        );
        k.body.push_op(
            Op::Scan { value: ValueId::new(4), axis: 0, op: ReduceKind::Sum, exclusive: false },
            ValueId::new(5),
        );

        let env = infer_types(&k).unwrap();
        let stride_reduce = env.get(&ValueId::new(3)).unwrap();
        assert_eq!(stride_reduce.dtype, DType::F32);
        assert_eq!(stride_reduce.shape, Shape::scalar());

        let gather = env.get(&ValueId::new(4)).unwrap();
        assert_eq!(gather.dtype, DType::F16);
        assert_eq!(gather.shape, Shape::scalar());

        let scan = env.get(&ValueId::new(5)).unwrap();
        assert_eq!(scan.dtype, DType::F16);
        assert_eq!(scan.shape, Shape::scalar());
    }
}
