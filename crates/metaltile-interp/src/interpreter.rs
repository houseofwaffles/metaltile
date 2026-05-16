//! CPU reference interpreter for MetalTile kernels.
//!
//! Walks the kernel IR and executes it on CPU tensors. Produces
//! bitwise results that (within fp tolerance) match GPU output.
//!
//! Uses a simple register-based VM model:
//! - Registers map to SSA ValueIds
//! - Each register holds either a scalar (f64) or a small dense buffer
//! - Operations are executed sequentially in block order

use std::collections::BTreeMap;

use metaltile_core::{
    constexpr::ConstExprValues,
    dtype::DType,
    ir::{
        ActKind,
        BinOpKind,
        Block,
        BlockId,
        IndexExpr,
        Kernel,
        Op,
        ReduceKind,
        UnaryOpKind,
        ValueId,
    },
    shape::{Dim, Shape},
};

use crate::tensor::TensorData;

/// Abramowitz & Stegun 7.1.26 approximation — max error < 1.5×10⁻⁷.
fn erf_f64(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let result = 1.0 - poly * (-x * x).exp();
    result.copysign(x)
}

fn num_elements(shape: &[usize]) -> usize { shape.iter().product() }

fn coords_from_linear(shape: &[usize], mut index: usize) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut coords = vec![0usize; shape.len()];
    for dim in (0..shape.len()).rev() {
        let size = shape[dim];
        coords[dim] = index % size;
        index /= size;
    }
    coords
}

fn scalar_tensor(value: f64, dtype: DType) -> TensorData {
    let mut tensor = TensorData::zeros(&[], dtype);
    tensor.write_scalar(0, value);
    tensor
}

fn reduce_identity(op: ReduceKind) -> f64 {
    match op {
        ReduceKind::Sum | ReduceKind::Mean => 0.0,
        ReduceKind::Max => f64::NEG_INFINITY,
        ReduceKind::Min => f64::INFINITY,
    }
}

fn reduce_combine(op: ReduceKind, acc: f64, value: f64) -> f64 {
    match op {
        ReduceKind::Sum | ReduceKind::Mean => acc + value,
        ReduceKind::Max => acc.max(value),
        ReduceKind::Min => acc.min(value),
    }
}

/// Result of interpreting a kernel.
#[derive(Debug)]
pub struct InterpResult {
    /// Output tensors (by parameter name).
    pub outputs: BTreeMap<String, TensorData>,
}

/// A register value: either a scalar f64 or a tensor.
#[derive(Debug, Clone)]
enum RegisterValue {
    Scalar(f64),
    Tensor(TensorData),
}

/// The CPU interpreter.
pub struct Interpreter {
    /// Input tensors by parameter name.
    inputs: BTreeMap<String, TensorData>,
    /// Constexpr values.
    constexprs: ConstExprValues,
    /// SSA registers: ValueId → value.
    registers: BTreeMap<ValueId, RegisterValue>,
    /// Current program_id per axis (0=x, 1=y, 2=z).
    current_program_id: [usize; 3],
}

impl Interpreter {
    /// Create a new interpreter with the given inputs and constexprs.
    pub fn new(inputs: BTreeMap<String, TensorData>, constexprs: ConstExprValues) -> Self {
        Interpreter { inputs, constexprs, registers: BTreeMap::new(), current_program_id: [0; 3] }
    }

    /// Run a kernel and return output tensors.
    ///
    /// Executes in single-program mode (program_id = 0 on all axes).
    /// For full multi-program correctness, use `run_grid`.
    pub fn run(&mut self, kernel: &Kernel) -> metaltile_core::error::Result<InterpResult> {
        // Execute the body block directly (kernel.blocks[0] is a stale clone made at
        // construction time; kernel.body has all the ops added by the macro/builder).
        let body = kernel.body.clone();
        self.execute_block(&body, &kernel.blocks)?;

        let mut outputs = BTreeMap::new();
        for param in &kernel.params {
            if param.is_output
                && let Some(tensor) = self.inputs.get(&param.name)
            {
                outputs.insert(param.name.clone(), tensor.clone());
            }
        }

        Ok(InterpResult { outputs })
    }

    /// Run the kernel for every program_id from 0..grid_x.
    /// Use this for elementwise kernels where each program handles one element.
    pub fn run_grid(
        &mut self,
        kernel: &Kernel,
        grid_x: usize,
    ) -> metaltile_core::error::Result<InterpResult> {
        let body = kernel.body.clone();
        let blocks = kernel.blocks.clone();
        for pid in 0..grid_x {
            self.current_program_id[0] = pid;
            self.execute_block(&body, &blocks)?;
        }
        self.current_program_id = [0; 3];

        let mut outputs = BTreeMap::new();
        for param in &kernel.params {
            if param.is_output
                && let Some(tensor) = self.inputs.get(&param.name)
            {
                outputs.insert(param.name.clone(), tensor.clone());
            }
        }
        Ok(InterpResult { outputs })
    }

    /// Run the kernel for every (x, y, z) program_id triple.
    /// Use this for Grid3D kernels (rope, etc.).
    pub fn run_grid_3d(
        &mut self,
        kernel: &Kernel,
        grid_x: usize,
        grid_y: usize,
        grid_z: usize,
    ) -> metaltile_core::error::Result<InterpResult> {
        let body = kernel.body.clone();
        let blocks = kernel.blocks.clone();
        for z in 0..grid_z {
            for y in 0..grid_y {
                for x in 0..grid_x {
                    self.current_program_id = [x, y, z];
                    self.execute_block(&body, &blocks)?;
                }
            }
        }
        self.current_program_id = [0; 3];

        let mut outputs = BTreeMap::new();
        for param in &kernel.params {
            if param.is_output
                && let Some(tensor) = self.inputs.get(&param.name)
            {
                outputs.insert(param.name.clone(), tensor.clone());
            }
        }
        Ok(InterpResult { outputs })
    }

    /// Run a Reduction-mode kernel: one program_id per row (axis 0), serial over columns.
    /// The interpreter runs single-threaded per row, so StrideReduce naturally iterates all elements
    /// (stride resolves to 1 since no threadgroup context exists on CPU).
    pub fn run_grid_reduction(
        &mut self,
        kernel: &Kernel,
        num_rows: usize,
    ) -> metaltile_core::error::Result<InterpResult> {
        let body = kernel.body.clone();
        let blocks = kernel.blocks.clone();
        for row in 0..num_rows {
            self.current_program_id = [row, 0, 0];
            self.execute_block(&body, &blocks)?;
        }
        self.current_program_id = [0; 3];

        let mut outputs = BTreeMap::new();
        for param in &kernel.params {
            if param.is_output
                && let Some(tensor) = self.inputs.get(&param.name)
            {
                outputs.insert(param.name.clone(), tensor.clone());
            }
        }
        Ok(InterpResult { outputs })
    }

    fn execute_block(
        &mut self,
        block: &Block,
        all_blocks: &BTreeMap<BlockId, Block>,
    ) -> metaltile_core::error::Result<()> {
        for (op_idx, op) in block.ops.iter().enumerate() {
            let result = block.results.get(op_idx).and_then(|x| *x);
            self.execute_op(op, result, all_blocks)?;
        }
        Ok(())
    }

    fn execute_op(
        &mut self,
        op: &Op,
        result: Option<ValueId>,
        all_blocks: &BTreeMap<BlockId, Block>, // used by Op::Loop
    ) -> metaltile_core::error::Result<()> {
        match op {
            Op::ProgramId { axis } => {
                let pid = self.current_program_id[(*axis as usize).min(2)];
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Scalar(pid as f64));
                }
            },

            Op::Const { value } =>
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Scalar(*value as f64));
                },

            Op::Arange { start, step, len } => {
                let n = self.constexprs.resolve(len);
                let s0 = start.unwrap_or(0.0);
                let ds = step.unwrap_or(1.0);
                if let Some(rid) = result {
                    // arange on CPU returns the full range as a 1D tensor
                    let mut data = TensorData::zeros(&[n], DType::F32);
                    for i in 0..n {
                        data.write_scalar(i, s0 + i as f64 * ds);
                    }
                    self.registers.insert(rid, RegisterValue::Tensor(data));
                }
            },

            Op::Zeros { dtype, shape } => {
                let _num_elems = shape.num_elements().unwrap_or(1);
                let shape_vec: Vec<usize> = shape
                    .iter()
                    .map(|d| match d {
                        metaltile_core::shape::Dim::Known(n) => *n,
                        metaltile_core::shape::Dim::ConstExpr(ce) => self.constexprs.resolve(ce),
                        metaltile_core::shape::Dim::Any => 1,
                    })
                    .collect();
                let data = TensorData::zeros(&shape_vec, *dtype);
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Tensor(data));
                }
            },

            Op::Load { src, indices, .. } => {
                let tensor = self.inputs.get(src).cloned().ok_or_else(|| {
                    metaltile_core::error::Error::UnknownValue(format!("tensor {src}"))
                })?;

                // Resolve indices to linear offset
                let offset = self.resolve_indices(indices, &tensor);
                let scalar = tensor.read_scalar(offset);

                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Scalar(scalar));
                }
            },

            Op::Store { dst, indices, value, .. } => {
                let val = self.get_scalar(*value);
                let mut tensor = self.inputs.get(dst).cloned().ok_or_else(|| {
                    metaltile_core::error::Error::UnknownValue(format!("tensor {dst}"))
                })?;
                let offset = self.resolve_indices(indices, &tensor);
                tensor.write_scalar(offset, val);
                self.inputs.insert(dst.clone(), tensor);
            },

            Op::BinOp { op, lhs, rhs } => {
                let l = self.get_scalar(*lhs);
                let r = self.get_scalar(*rhs);
                let out = match op {
                    BinOpKind::Add => l + r,
                    BinOpKind::Sub => l - r,
                    BinOpKind::Mul => l * r,
                    BinOpKind::Div => l / r,
                    BinOpKind::Max => l.max(r),
                    BinOpKind::Min => l.min(r),
                    BinOpKind::And => ((l != 0.0) && (r != 0.0)) as u8 as f64,
                    BinOpKind::Or => ((l != 0.0) || (r != 0.0)) as u8 as f64,
                    BinOpKind::Xor => ((l != 0.0) != (r != 0.0)) as u8 as f64,
                    BinOpKind::CmpLt => (l < r) as u8 as f64,
                    BinOpKind::CmpGt => (l > r) as u8 as f64,
                    BinOpKind::CmpLe => (l <= r) as u8 as f64,
                    BinOpKind::CmpGe => (l >= r) as u8 as f64,
                    BinOpKind::CmpEq => (l == r) as u8 as f64,
                    BinOpKind::CmpNe => (l != r) as u8 as f64,
                    BinOpKind::Pow => l.powf(r),
                    BinOpKind::Shl => ((l as i64) << (r as i64)) as f64,
                    BinOpKind::Shr => ((l as i64) >> (r as i64)) as f64,
                    BinOpKind::BitAnd => ((l as i64) & (r as i64)) as f64,
                    BinOpKind::BitOr => ((l as i64) | (r as i64)) as f64,
                    BinOpKind::BitXor => ((l as i64) ^ (r as i64)) as f64,
                };
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Scalar(out));
                }
            },

            Op::Dot { a: _a, b: _b } => {
                // Simplified CPU dot: just multiply scalars
                // Real implementation would do tile matmul
                let a_val = self.get_scalar(*_a);
                let b_val = self.get_scalar(*_b);
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Scalar(a_val * b_val));
                }
            },

            Op::Reduce { value, axis, op } => {
                let input = self.register_as_tensor(*value)?;
                let reduced = self.reduce_tensor(&input, *axis as usize, *op)?;
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Tensor(reduced));
                }
            },

            Op::Cast { value, dtype: _dtype } => {
                let v = self.get_scalar(*value);
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Scalar(v));
                }
            },

            Op::Transpose { value } =>
                if let Some(rid) = result {
                    let reg_val = match self.registers.get(value).cloned() {
                        Some(RegisterValue::Tensor(tensor)) =>
                            RegisterValue::Tensor(self.transpose_tensor(&tensor)),
                        Some(other) => other,
                        None => RegisterValue::Scalar(0.0),
                    };
                    self.registers.insert(rid, reg_val);
                },

            Op::Loop { var, start, end, step, body: bid } => {
                let s = self.get_scalar(*start) as usize;
                let e = self.get_scalar(*end) as usize;
                let st = (self.get_scalar(*step) as usize).max(1);
                if let Some(body_block) = all_blocks.get(bid).cloned() {
                    for i in (s..e).step_by(st) {
                        // body_parser stores loop var at ValueId(var_id + 1000) to
                        // avoid collision with regular SSA values.
                        self.registers.insert(
                            ValueId::new(var.as_u32() + 1000),
                            RegisterValue::Scalar(i as f64),
                        );
                        self.execute_block(&body_block, all_blocks)?;
                    }
                }
            },

            Op::If { cond, then_block, else_block } => {
                let selected =
                    if self.get_scalar(*cond) != 0.0 { Some(*then_block) } else { *else_block };
                if let Some(block_id) = selected
                    && let Some(block) = all_blocks.get(&block_id)
                {
                    self.execute_block(block, all_blocks)?;
                }
            },

            Op::ExpandDims { value, axis } => {
                let mut tensor = self.register_as_tensor(*value)?;
                let insert_at = (*axis as usize).min(tensor.shape.len());
                tensor.shape.insert(insert_at, 1);
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Tensor(tensor));
                }
            },

            Op::Reshape { value, shape } => {
                let mut tensor = self.register_as_tensor(*value)?;
                tensor.shape = self.resolve_shape(shape, tensor.num_elements())?;
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Tensor(tensor));
                }
            },

            Op::Cat { values, axis } => {
                let cat = self.cat_tensors(values, *axis as usize)?;
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Tensor(cat));
                }
            },

            Op::Gather { src, indices, axis } => {
                let src_tensor = self.inputs.get(src).cloned().ok_or_else(|| {
                    metaltile_core::error::Error::UnknownValue(format!("tensor {src}"))
                })?;
                let indices_tensor = self.register_as_tensor(*indices)?;
                let gathered = self.gather_tensor(&src_tensor, &indices_tensor, *axis as usize)?;
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Tensor(gathered));
                }
            },

            Op::Scatter { dst, indices, value, axis } => {
                let indices_tensor = self.register_as_tensor(*indices)?;
                let value_reg = self.register_value(*value)?;
                let mut dst_tensor = self.inputs.get(dst).cloned().ok_or_else(|| {
                    metaltile_core::error::Error::UnknownValue(format!("tensor {dst}"))
                })?;
                self.scatter_tensor(&mut dst_tensor, &indices_tensor, &value_reg, *axis as usize)?;
                self.inputs.insert(dst.clone(), dst_tensor);
            },

            Op::Scan { value, axis, op, exclusive } => {
                let input = self.register_as_tensor(*value)?;
                let scanned = self.scan_tensor(&input, *axis as usize, *op, *exclusive)?;
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Tensor(scanned));
                }
            },

            Op::Slice { value, ranges: _ranges } => {
                if let Some(reg_val) = self.registers.get(value).cloned()
                    && let Some(rid) = result
                {
                    self.registers.insert(rid, reg_val);
                }
            },

            Op::UnaryOp { op, value } => {
                let v = self.get_scalar(*value);
                let out = match op {
                    UnaryOpKind::Exp => v.exp(),
                    UnaryOpKind::Log => v.ln(),
                    UnaryOpKind::Sqrt => v.sqrt(),
                    UnaryOpKind::Rsqrt => 1.0 / v.sqrt(),
                    UnaryOpKind::Abs => v.abs(),
                    UnaryOpKind::Neg => -v,
                    UnaryOpKind::Ceil => v.ceil(),
                    UnaryOpKind::Floor => v.floor(),
                    UnaryOpKind::Recip => 1.0 / v,
                    UnaryOpKind::Sin => v.sin(),
                    UnaryOpKind::Cos => v.cos(),
                    UnaryOpKind::Exp2 => 2.0f64.powf(v),
                    UnaryOpKind::Log2 => v.log2(),
                    UnaryOpKind::Erf => erf_f64(v),
                    UnaryOpKind::Sign =>
                        if v > 0.0 {
                            1.0
                        } else if v < 0.0 {
                            -1.0
                        } else {
                            0.0
                        },
                    UnaryOpKind::Round => v.round(),
                    UnaryOpKind::Trunc => v.trunc(),
                };
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Scalar(out));
                }
            },

            Op::Activation { kind, value } => {
                let x = self.get_scalar(*value);
                let out = match kind {
                    ActKind::Silu => x / (1.0 + (-x).exp()),
                    ActKind::Gelu => {
                        let k = 0.7978845608_f64;
                        0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh())
                    },
                    ActKind::Relu => x.max(0.0),
                    ActKind::Tanh => x.tanh(),
                    ActKind::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                };
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Scalar(out));
                }
            },

            Op::Select { cond, on_true, on_false } => {
                let c = self.get_scalar(*cond);
                let val =
                    if c != 0.0 { self.get_scalar(*on_true) } else { self.get_scalar(*on_false) };
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Scalar(val));
                }
            },

            Op::Broadcast { value, shape } => {
                let scalar = self.get_scalar(*value);
                let n = shape.num_elements().unwrap_or(1);
                let mut td =
                    crate::tensor::TensorData::zeros(&[n], metaltile_core::dtype::DType::F32);
                for i in 0..n {
                    td.write_scalar(i, scalar);
                }
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Tensor(td));
                }
            },

            Op::Splat { value: sv, dtype, shape } => {
                let mut shape_vec: Vec<usize> = shape
                    .iter()
                    .map(|d| match d {
                        metaltile_core::shape::Dim::Known(n) => *n,
                        metaltile_core::shape::Dim::ConstExpr(ce) => self.constexprs.resolve(ce),
                        metaltile_core::shape::Dim::Any => 1,
                    })
                    .collect();
                if shape_vec.is_empty() {
                    shape_vec.push(1);
                }
                let mut td = crate::tensor::TensorData::zeros(&shape_vec, *dtype);
                let total = td.num_elements();
                for i in 0..total {
                    td.write_scalar(i, *sv);
                }
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Tensor(td));
                }
            },

            // New SIMD/threadgroup ops — no-op on CPU interpreter
            Op::SimdReduce { value, op: _ } => {
                // On CPU, SIMD reduction = scalar identity (single-thread)
                let v = self.get_scalar(*value);
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Scalar(v));
                }
            },
            Op::ThreadgroupAlloc { .. }
            | Op::ThreadgroupLoad { .. }
            | Op::ThreadgroupStore { .. }
            | Op::Barrier =>
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Scalar(0.0));
                },
            // Mutable locals: DeclareLocal stores bare name (e.g. "lm");
            // reads use Op::Load { src: "__ml_lm" }, so insert with "__ml_" prefix.
            Op::DeclareLocal { name, value } => {
                let v = self.get_scalar(*value);
                let mut td = TensorData::zeros(&[1], DType::F32);
                td.write_scalar(0, v);
                self.inputs.insert(format!("__ml_{name}"), td);
            },
            Op::SetLocal { name, value } => {
                let v = self.get_scalar(*value);
                let mut td = TensorData::zeros(&[1], DType::F32);
                td.write_scalar(0, v);
                self.inputs.insert(format!("__ml_{name}"), td);
            },
            Op::ArgReduce { value, op, .. } => {
                // Return index 0 as placeholder (tensor-level argreduce not implemented yet)
                let _ = (value, op);
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Scalar(0.0));
                }
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
                let tensor = self
                    .inputs
                    .get(src)
                    .cloned()
                    .unwrap_or_else(|| TensorData::zeros(&[1], DType::F32));
                let sec_tensor = secondary_src.as_ref().map(|s| {
                    self.inputs
                        .get(s)
                        .cloned()
                        .unwrap_or_else(|| TensorData::zeros(&[1], DType::F32))
                });
                let base_off = secondary_base.map(|b| self.get_scalar(b) as usize).unwrap_or(0);
                let off = self.get_scalar(*offset) as usize;
                let st = (self.get_scalar(*stride) as usize).max(1);
                let en = self.get_scalar(*end) as usize;
                let mut acc = match rk {
                    ReduceKind::Sum | ReduceKind::Mean => 0.0f64,
                    ReduceKind::Max => f64::NEG_INFINITY,
                    ReduceKind::Min => f64::INFINITY,
                };
                let mut count = 0usize;
                let mut idx = off;
                while idx < en {
                    let mut v = tensor.read_scalar(idx);
                    // Multiply by secondary tensor for dot-product reductions.
                    if let Some(ref sec) = sec_tensor {
                        let sec_idx = idx.saturating_sub(base_off);
                        v *= sec.read_scalar(sec_idx);
                    }
                    // Apply per-element transform chain.
                    if let Some(chain) = transform {
                        for sub_op in chain {
                            v = match sub_op {
                                Op::UnaryOp { op, .. } => self.apply_unary(*op, v),
                                Op::BinOp { op, rhs, .. } => {
                                    let rv =
                                        if rhs.as_u32() == 0 { v } else { self.get_scalar(*rhs) };
                                    self.apply_binop(*op, v, rv)
                                },
                                Op::Cast { .. } => v,
                                _ => v,
                            };
                        }
                    }
                    acc = match rk {
                        ReduceKind::Sum | ReduceKind::Mean => acc + v,
                        ReduceKind::Max => acc.max(v),
                        ReduceKind::Min => acc.min(v),
                    };
                    count += 1;
                    idx += st;
                }
                if matches!(rk, ReduceKind::Mean) {
                    acc /= count.max(1) as f64;
                }
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Scalar(acc));
                }
            },

            Op::StrideStore { src, dst, offset, end, scalar, aux_src } => {
                let off = self.get_scalar(*offset) as usize;
                let en = self.get_scalar(*end) as usize;
                let sc = self.get_scalar(*scalar);
                let src_tensor = self
                    .inputs
                    .get(src)
                    .cloned()
                    .unwrap_or_else(|| TensorData::zeros(&[1], DType::F32));
                let aux_tensor = aux_src.as_ref().and_then(|n| self.inputs.get(n).cloned());
                let mut dst_tensor = self
                    .inputs
                    .get(dst)
                    .cloned()
                    .unwrap_or_else(|| TensorData::zeros(&[1], DType::F32));
                for i in off..en {
                    let rel = i - off;
                    let v = src_tensor.read_scalar(i)
                        * sc
                        * aux_tensor.as_ref().map(|t| t.read_scalar(rel)).unwrap_or(1.0);
                    dst_tensor.write_scalar(i, v);
                }
                self.inputs.insert(dst.clone(), dst_tensor);
            },

            Op::StrideScan { src, dst, offset, end, op: rk } => {
                let tensor = self
                    .inputs
                    .get(src)
                    .cloned()
                    .unwrap_or_else(|| TensorData::zeros(&[1], DType::F32));
                let start = self.get_scalar(*offset) as usize;
                let stop = self.get_scalar(*end) as usize;
                let mut dst_tensor = self
                    .inputs
                    .get(dst)
                    .cloned()
                    .unwrap_or_else(|| TensorData::zeros(&[stop.max(1)], DType::F32));
                let mut acc = match rk {
                    ReduceKind::Sum | ReduceKind::Mean => 0.0f64,
                    ReduceKind::Max => f64::NEG_INFINITY,
                    ReduceKind::Min => f64::INFINITY,
                };
                for i in start..stop {
                    let v = tensor.read_scalar(i);
                    acc = match rk {
                        ReduceKind::Sum | ReduceKind::Mean => acc + v,
                        ReduceKind::Max => acc.max(v),
                        ReduceKind::Min => acc.min(v),
                    };
                    dst_tensor.write_scalar(i, acc);
                }
                self.inputs.insert(dst.clone(), dst_tensor);
            },

            Op::StrideArgReduce { src, offset, end, op: rk } => {
                let tensor = self
                    .inputs
                    .get(src)
                    .cloned()
                    .unwrap_or_else(|| TensorData::zeros(&[1], DType::F32));
                let start = self.get_scalar(*offset) as usize;
                let stop = self.get_scalar(*end) as usize;
                let mut best_idx = start;
                let mut best_val = tensor.read_scalar(start);
                for i in (start + 1)..stop {
                    let v = tensor.read_scalar(i);
                    let better = match rk {
                        ReduceKind::Max => v > best_val,
                        ReduceKind::Min => v < best_val,
                        _ => v > best_val,
                    };
                    if better {
                        best_val = v;
                        best_idx = i;
                    }
                }
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Scalar(best_idx as f64));
                }
            },

            Op::InlineMsl { .. }
            | Op::FlashAttention { .. }
            | Op::SlidingWindowAttention { .. }
            | Op::RmsNorm { .. }
            | Op::GatedMlp { .. }
            | Op::FusedElementwise { .. }
            | Op::VectorLoad { .. }
            | Op::VectorStore { .. }
            | Op::Atomic { .. }
            | Op::Dequantize { .. } => {
                // High-level ops not yet interpretable; return zero.
                if let Some(rid) = result {
                    self.registers.insert(rid, RegisterValue::Scalar(0.0));
                }
            },
        }

        Ok(())
    }

    fn register_value(&self, id: ValueId) -> metaltile_core::error::Result<RegisterValue> {
        self.registers.get(&id).cloned().ok_or_else(|| {
            metaltile_core::error::Error::UnknownValue(format!("register {}", id.as_u32()))
        })
    }

    fn register_as_tensor(&self, id: ValueId) -> metaltile_core::error::Result<TensorData> {
        match self.register_value(id)? {
            RegisterValue::Tensor(tensor) => Ok(tensor),
            RegisterValue::Scalar(value) => Ok(scalar_tensor(value, DType::F32)),
        }
    }

    fn resolve_shape(
        &self,
        shape: &Shape,
        num_elems: usize,
    ) -> metaltile_core::error::Result<Vec<usize>> {
        let mut resolved = Vec::with_capacity(shape.rank());
        let mut infer_dim = None;
        let mut known_product = 1usize;

        for (idx, dim) in shape.iter().enumerate() {
            match dim {
                Dim::Known(size) => {
                    resolved.push(*size);
                    known_product = known_product.saturating_mul(*size);
                },
                Dim::ConstExpr(expr) => {
                    let size = self.constexprs.resolve(expr);
                    resolved.push(size);
                    known_product = known_product.saturating_mul(size);
                },
                Dim::Any => {
                    if infer_dim.replace(idx).is_some() {
                        return Err(metaltile_core::error::Error::Validation(
                            "reshape supports at most one dynamic dimension".into(),
                        ));
                    }
                    resolved.push(0);
                },
            }
        }

        if let Some(idx) = infer_dim {
            if known_product == 0 || !num_elems.is_multiple_of(known_product) {
                return Err(metaltile_core::error::Error::ShapeMismatch {
                    expected: shape.to_string(),
                    actual: format!("{num_elems} elements"),
                });
            }
            resolved[idx] = num_elems / known_product;
        } else if num_elements(&resolved) != num_elems {
            return Err(metaltile_core::error::Error::ShapeMismatch {
                expected: shape.to_string(),
                actual: format!("{num_elems} elements"),
            });
        }

        Ok(resolved)
    }

    fn read_bounded_index(
        &self,
        value: f64,
        upper_bound: usize,
    ) -> metaltile_core::error::Result<usize> {
        if !value.is_finite() || value.fract() != 0.0 {
            return Err(metaltile_core::error::Error::Validation(format!(
                "index value {value} is not an integer"
            )));
        }
        let idx = value as i64;
        if idx < 0 || idx as usize >= upper_bound {
            return Err(metaltile_core::error::Error::Validation(format!(
                "index {idx} is out of bounds for length {upper_bound}"
            )));
        }
        Ok(idx as usize)
    }

    fn read_index_from_tensor(
        &self,
        indices: &TensorData,
        coords: &[usize],
        upper_bound: usize,
    ) -> metaltile_core::error::Result<usize> {
        let linear = indices.linear_index(coords);
        self.read_bounded_index(indices.read_scalar(linear), upper_bound)
    }

    fn reduce_tensor(
        &self,
        tensor: &TensorData,
        axis: usize,
        op: ReduceKind,
    ) -> metaltile_core::error::Result<TensorData> {
        if tensor.rank() == 0 {
            return Ok(tensor.clone());
        }

        if tensor.rank() == 1 || axis >= tensor.rank() {
            let mut acc = reduce_identity(op);
            for idx in 0..tensor.num_elements() {
                acc = reduce_combine(op, acc, tensor.read_scalar(idx));
            }
            if matches!(op, ReduceKind::Mean) {
                acc /= tensor.num_elements().max(1) as f64;
            }
            return Ok(scalar_tensor(acc, tensor.dtype));
        }

        let mut out_shape = tensor.shape.clone();
        let axis_len = out_shape.remove(axis);
        let mut out = TensorData::zeros(&out_shape, tensor.dtype);
        for out_idx in 0..out.num_elements() {
            let out_coords = coords_from_linear(&out_shape, out_idx);
            let mut acc = reduce_identity(op);
            for axis_idx in 0..axis_len {
                let mut in_coords = out_coords.clone();
                in_coords.insert(axis, axis_idx);
                acc = reduce_combine(op, acc, tensor.read_scalar(tensor.linear_index(&in_coords)));
            }
            if matches!(op, ReduceKind::Mean) {
                acc /= axis_len.max(1) as f64;
            }
            out.write_scalar(out_idx, acc);
        }
        Ok(out)
    }

    fn transpose_tensor(&self, tensor: &TensorData) -> TensorData {
        if tensor.rank() < 2 {
            return tensor.clone();
        }

        let mut out_shape = tensor.shape.clone();
        out_shape.swap(0, 1);
        let mut out = TensorData::zeros(&out_shape, tensor.dtype);
        for out_idx in 0..out.num_elements() {
            let mut in_coords = coords_from_linear(&out_shape, out_idx);
            in_coords.swap(0, 1);
            let value = tensor.read_scalar(tensor.linear_index(&in_coords));
            out.write_scalar(out_idx, value);
        }
        out
    }

    fn cat_tensors(
        &self,
        values: &[ValueId],
        axis: usize,
    ) -> metaltile_core::error::Result<TensorData> {
        if values.is_empty() {
            return Err(metaltile_core::error::Error::Validation(
                "Op::Cat requires at least one input".into(),
            ));
        }

        let tensors: Vec<TensorData> = values
            .iter()
            .map(|value| self.register_as_tensor(*value))
            .collect::<metaltile_core::error::Result<_>>()?;
        let first = &tensors[0];

        if first.rank() == 0 {
            if axis != 0 {
                return Err(metaltile_core::error::Error::InvalidRank {
                    expected: 0,
                    actual: axis,
                });
            }
            let mut out = TensorData::zeros(&[tensors.len()], first.dtype);
            for (idx, tensor) in tensors.iter().enumerate() {
                if tensor.rank() != 0 || tensor.dtype != first.dtype {
                    return Err(metaltile_core::error::Error::Validation(
                        "Op::Cat scalar inputs must all have matching scalar shapes and dtypes"
                            .into(),
                    ));
                }
                out.write_scalar(idx, tensor.read_scalar(0));
            }
            return Ok(out);
        }

        if axis >= first.rank() {
            return Err(metaltile_core::error::Error::InvalidRank {
                expected: first.rank().saturating_sub(1),
                actual: axis,
            });
        }

        let mut out_shape = first.shape.clone();
        out_shape[axis] = 0;
        for tensor in &tensors {
            if tensor.rank() != first.rank() || tensor.dtype != first.dtype {
                return Err(metaltile_core::error::Error::Validation(
                    "Op::Cat inputs must have matching rank and dtype".into(),
                ));
            }
            for dim in 0..tensor.rank() {
                if dim != axis && tensor.shape[dim] != first.shape[dim] {
                    return Err(metaltile_core::error::Error::ShapeMismatch {
                        expected: format!("{:?}", first.shape),
                        actual: format!("{:?}", tensor.shape),
                    });
                }
            }
            out_shape[axis] += tensor.shape[axis];
        }

        let mut out = TensorData::zeros(&out_shape, first.dtype);
        let mut axis_offset = 0usize;
        for tensor in &tensors {
            for idx in 0..tensor.num_elements() {
                let mut out_coords = coords_from_linear(&tensor.shape, idx);
                out_coords[axis] += axis_offset;
                let out_linear = out.linear_index(&out_coords);
                out.write_scalar(out_linear, tensor.read_scalar(idx));
            }
            axis_offset += tensor.shape[axis];
        }
        Ok(out)
    }

    fn gather_tensor(
        &self,
        src: &TensorData,
        indices: &TensorData,
        axis: usize,
    ) -> metaltile_core::error::Result<TensorData> {
        if src.rank() == 0 || axis >= src.rank() {
            return Err(metaltile_core::error::Error::InvalidRank {
                expected: src.rank().saturating_sub(1),
                actual: axis,
            });
        }

        let mut out_shape = Vec::new();
        out_shape.extend_from_slice(&src.shape[..axis]);
        out_shape.extend_from_slice(&indices.shape);
        out_shape.extend_from_slice(&src.shape[axis + 1..]);

        let mut out = TensorData::zeros(&out_shape, src.dtype);
        let index_rank = indices.rank();
        for out_idx in 0..out.num_elements() {
            let out_coords = coords_from_linear(&out_shape, out_idx);
            let index_coords = &out_coords[axis..axis + index_rank];
            let gather_idx = self.read_index_from_tensor(indices, index_coords, src.shape[axis])?;

            let mut src_coords = Vec::with_capacity(src.rank());
            src_coords.extend_from_slice(&out_coords[..axis]);
            src_coords.push(gather_idx);
            src_coords.extend_from_slice(&out_coords[axis + index_rank..]);

            let value = src.read_scalar(src.linear_index(&src_coords));
            out.write_scalar(out_idx, value);
        }

        Ok(out)
    }

    fn scatter_tensor(
        &self,
        dst: &mut TensorData,
        indices: &TensorData,
        value: &RegisterValue,
        axis: usize,
    ) -> metaltile_core::error::Result<()> {
        if dst.rank() == 0 || axis >= dst.rank() {
            return Err(metaltile_core::error::Error::InvalidRank {
                expected: dst.rank().saturating_sub(1),
                actual: axis,
            });
        }

        let mut expected_value_shape = Vec::new();
        expected_value_shape.extend_from_slice(&dst.shape[..axis]);
        expected_value_shape.extend_from_slice(&indices.shape);
        expected_value_shape.extend_from_slice(&dst.shape[axis + 1..]);

        let broadcast_scalar = match value {
            RegisterValue::Scalar(scalar) => Some(*scalar),
            RegisterValue::Tensor(tensor) if tensor.num_elements() == 1 =>
                Some(tensor.read_scalar(0)),
            RegisterValue::Tensor(tensor) => {
                if tensor.shape != expected_value_shape {
                    return Err(metaltile_core::error::Error::ShapeMismatch {
                        expected: format!("{expected_value_shape:?}"),
                        actual: format!("{:?}", tensor.shape),
                    });
                }
                None
            },
        };

        let iter_shape = expected_value_shape;
        let total = num_elements(&iter_shape);
        let index_rank = indices.rank();
        for iter_idx in 0..total {
            let iter_coords = coords_from_linear(&iter_shape, iter_idx);
            let index_coords = &iter_coords[axis..axis + index_rank];
            let scatter_idx =
                self.read_index_from_tensor(indices, index_coords, dst.shape[axis])?;

            let mut dst_coords = Vec::with_capacity(dst.rank());
            dst_coords.extend_from_slice(&iter_coords[..axis]);
            dst_coords.push(scatter_idx);
            dst_coords.extend_from_slice(&iter_coords[axis + index_rank..]);

            let scatter_value = match value {
                RegisterValue::Scalar(_) => broadcast_scalar.unwrap_or(0.0),
                RegisterValue::Tensor(tensor) => broadcast_scalar
                    .unwrap_or_else(|| tensor.read_scalar(tensor.linear_index(&iter_coords))),
            };

            let dst_linear = dst.linear_index(&dst_coords);
            dst.write_scalar(dst_linear, scatter_value);
        }

        Ok(())
    }

    fn scan_tensor(
        &self,
        tensor: &TensorData,
        axis: usize,
        op: ReduceKind,
        exclusive: bool,
    ) -> metaltile_core::error::Result<TensorData> {
        if tensor.rank() == 0 {
            let value = tensor.read_scalar(0);
            let scanned = if exclusive {
                match op {
                    ReduceKind::Mean => 0.0,
                    _ => reduce_identity(op),
                }
            } else {
                value
            };
            return Ok(scalar_tensor(scanned, tensor.dtype));
        }

        if axis >= tensor.rank() {
            return Err(metaltile_core::error::Error::InvalidRank {
                expected: tensor.rank().saturating_sub(1),
                actual: axis,
            });
        }

        let mut out = TensorData::zeros(&tensor.shape, tensor.dtype);
        let mut outer_shape = tensor.shape.clone();
        let axis_len = outer_shape.remove(axis);
        let outer_total = num_elements(&outer_shape);

        for outer_idx in 0..outer_total {
            let outer_coords = coords_from_linear(&outer_shape, outer_idx);
            let mut acc = reduce_identity(op);
            let mut count = 0usize;
            for axis_idx in 0..axis_len {
                let mut coords = outer_coords.clone();
                coords.insert(axis, axis_idx);
                let value = tensor.read_scalar(tensor.linear_index(&coords));

                let out_value = if exclusive {
                    match op {
                        ReduceKind::Mean =>
                            if count == 0 {
                                0.0
                            } else {
                                acc / count as f64
                            },
                        _ => acc,
                    }
                } else {
                    let next_acc = reduce_combine(op, acc, value);
                    match op {
                        ReduceKind::Mean => next_acc / (count + 1) as f64,
                        _ => next_acc,
                    }
                };

                let out_linear = out.linear_index(&coords);
                out.write_scalar(out_linear, out_value);
                acc = reduce_combine(op, acc, value);
                count += 1;
            }
        }

        Ok(out)
    }

    fn apply_unary(&self, op: UnaryOpKind, v: f64) -> f64 {
        match op {
            UnaryOpKind::Neg => -v,
            UnaryOpKind::Recip => 1.0 / v,
            UnaryOpKind::Exp => v.exp(),
            UnaryOpKind::Log => v.ln(),
            UnaryOpKind::Sqrt => v.sqrt(),
            UnaryOpKind::Rsqrt => 1.0 / v.sqrt(),
            UnaryOpKind::Abs => v.abs(),
            UnaryOpKind::Ceil => v.ceil(),
            UnaryOpKind::Floor => v.floor(),
            UnaryOpKind::Sin => v.sin(),
            UnaryOpKind::Cos => v.cos(),
            UnaryOpKind::Erf => erf_f64(v),
            UnaryOpKind::Exp2 => v.exp2(),
            UnaryOpKind::Log2 => v.log2(),
            UnaryOpKind::Sign => v.signum(),
            UnaryOpKind::Round => v.round(),
            UnaryOpKind::Trunc => v.trunc(),
        }
    }

    fn apply_binop(&self, op: BinOpKind, l: f64, r: f64) -> f64 {
        match op {
            BinOpKind::Add => l + r,
            BinOpKind::Sub => l - r,
            BinOpKind::Mul => l * r,
            BinOpKind::Div => l / r,
            BinOpKind::Max => l.max(r),
            BinOpKind::Min => l.min(r),
            BinOpKind::Pow => l.powf(r),
            BinOpKind::And => ((l != 0.0) && (r != 0.0)) as u8 as f64,
            BinOpKind::Or => ((l != 0.0) || (r != 0.0)) as u8 as f64,
            BinOpKind::Xor => ((l != 0.0) != (r != 0.0)) as u8 as f64,
            BinOpKind::CmpLt => (l < r) as u8 as f64,
            BinOpKind::CmpGt => (l > r) as u8 as f64,
            BinOpKind::CmpLe => (l <= r) as u8 as f64,
            BinOpKind::CmpGe => (l >= r) as u8 as f64,
            BinOpKind::CmpEq => ((l - r).abs() < f64::EPSILON) as u8 as f64,
            BinOpKind::CmpNe => ((l - r).abs() >= f64::EPSILON) as u8 as f64,
            BinOpKind::BitAnd => (l as i64 & r as i64) as f64,
            BinOpKind::BitOr => (l as i64 | r as i64) as f64,
            BinOpKind::BitXor => (l as i64 ^ r as i64) as f64,
            BinOpKind::Shl => ((l as i64) << (r as u32)) as f64,
            BinOpKind::Shr => ((l as i64) >> (r as u32)) as f64,
        }
    }

    fn get_scalar(&self, id: ValueId) -> f64 {
        match self.registers.get(&id) {
            Some(RegisterValue::Scalar(v)) => *v,
            Some(RegisterValue::Tensor(t)) => t.read_scalar(0),
            None => 0.0,
        }
    }

    fn resolve_indices(&self, indices: &[IndexExpr], tensor: &TensorData) -> usize {
        if indices.is_empty() {
            return 0;
        }

        let mut coords = vec![0usize; tensor.rank()];
        for (dim, idx) in indices.iter().enumerate() {
            if dim >= coords.len() {
                break;
            }
            coords[dim] = match idx {
                IndexExpr::Value(vid) => self.get_scalar(*vid) as usize,
                IndexExpr::Const(c) => *c as usize,
                IndexExpr::Range(vid, _len) => self.get_scalar(*vid) as usize,
            };
        }

        tensor.linear_index(&coords)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use metaltile_core::{
        constexpr::ConstExprValues,
        dtype::DType,
        ir::{BinOpKind, Block, BlockId, IndexExpr, Kernel, Op, Param, ReduceKind, ValueId},
        shape::Shape,
    };

    use super::*;
    use crate::tensor::TensorData;

    fn f32_tensor(data: &[f32]) -> TensorData {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        TensorData::from_bytes(&[data.len()], DType::F32, bytes)
    }

    fn f32_tensor_with_shape(shape: &[usize], data: &[f32]) -> TensorData {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        TensorData::from_bytes(shape, DType::F32, bytes)
    }

    fn i32_tensor_with_shape(shape: &[usize], data: &[i32]) -> TensorData {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        TensorData::from_bytes(shape, DType::I32, bytes)
    }

    fn read_f32(t: &TensorData, idx: usize) -> f32 {
        // read_scalar already returns the numeric value as f64 (not raw bits)
        t.read_scalar(idx) as f32
    }

    fn read_all_f32(t: &TensorData) -> Vec<f32> {
        (0..t.num_elements()).map(|idx| read_f32(t, idx)).collect()
    }

    fn register_tensor<'a>(interp: &'a Interpreter, id: u32) -> &'a TensorData {
        match interp.registers.get(&ValueId::new(id)) {
            Some(RegisterValue::Tensor(tensor)) => tensor,
            Some(RegisterValue::Scalar(_)) => panic!("expected tensor register {id}"),
            None => panic!("missing register {id}"),
        }
    }

    /// Build a scalar vector-add kernel (single-element, no arange).
    fn make_vadd_kernel() -> Kernel {
        let mut k = Kernel::new("vadd");
        k.params.push(Param {
            name: "a".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: false,
            kind: Default::default(),
        });
        k.params.push(Param {
            name: "b".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: false,
            kind: Default::default(),
        });
        k.params.push(Param {
            name: "c".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: true,
            kind: Default::default(),
        });
        // v0 = program_id(0)
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        // v1 = load a[v0]
        k.body.push_op(
            Op::Load {
                src: "a".into(),
                mask: None,
                other: None,
                indices: vec![IndexExpr::Value(ValueId::new(0))],
            },
            ValueId::new(1),
        );
        // v2 = load b[v0]
        k.body.push_op(
            Op::Load {
                src: "b".into(),
                mask: None,
                other: None,
                indices: vec![IndexExpr::Value(ValueId::new(0))],
            },
            ValueId::new(2),
        );
        // v3 = v1 + v2
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(1), rhs: ValueId::new(2) },
            ValueId::new(3),
        );
        // store c[v0] = v3
        k.body.push_op_no_result(Op::Store {
            dst: "c".into(),
            mask: None,
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(3),
        });
        k
    }

    #[test]
    fn interp_vadd_single_element() {
        let kernel = make_vadd_kernel();
        let mut inputs = BTreeMap::new();
        inputs.insert("a".into(), f32_tensor(&[3.0]));
        inputs.insert("b".into(), f32_tensor(&[4.0]));
        inputs.insert("c".into(), f32_tensor(&[0.0]));

        let mut interp = Interpreter::new(inputs, ConstExprValues::new());
        let result = interp.run(&kernel).unwrap();

        let c = result.outputs.get("c").expect("output c missing");
        assert_eq!(read_f32(c, 0), 7.0, "3+4 should be 7");
    }

    #[test]
    fn interp_vadd_full_array_run_grid() {
        let kernel = make_vadd_kernel();
        let n = 8usize;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let c = vec![0.0f32; n];

        let mut inputs = BTreeMap::new();
        inputs.insert("a".into(), f32_tensor(&a));
        inputs.insert("b".into(), f32_tensor(&b));
        inputs.insert("c".into(), f32_tensor(&c));

        let mut interp = Interpreter::new(inputs, ConstExprValues::new());
        let result = interp.run_grid(&kernel, n).unwrap();

        let c_out = result.outputs.get("c").expect("output c missing");
        for i in 0..n {
            let expected = a[i] + b[i];
            let got = read_f32(c_out, i);
            assert_eq!(got, expected, "element {i}: expected {expected}, got {got}");
        }
    }

    #[test]
    fn interp_binop_sub_mul_div() {
        let mut k = Kernel::new("binops");
        k.params.push(Param {
            name: "out".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: true,
            kind: Default::default(),
        });
        k.body.push_op(Op::Const { value: 10 }, ValueId::new(0)); // 10
        k.body.push_op(Op::Const { value: 3 }, ValueId::new(1)); // 3
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Sub, lhs: ValueId::new(0), rhs: ValueId::new(1) },
            ValueId::new(2),
        ); // 7
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Mul, lhs: ValueId::new(2), rhs: ValueId::new(1) },
            ValueId::new(3),
        ); // 21
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Div, lhs: ValueId::new(3), rhs: ValueId::new(1) },
            ValueId::new(4),
        ); // 7
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            mask: None,
            indices: vec![IndexExpr::Const(0)],
            value: ValueId::new(4),
        });

        let mut inputs = BTreeMap::new();
        inputs.insert("out".into(), f32_tensor(&[0.0]));

        let mut interp = Interpreter::new(inputs, ConstExprValues::new());
        let result = interp.run(&k).unwrap();
        let out = result.outputs.get("out").unwrap();
        assert_eq!(read_f32(out, 0), 7.0);
    }

    #[test]
    fn interp_reduce_tensor_axis_sum_and_mean() {
        let mut interp = Interpreter::new(BTreeMap::new(), ConstExprValues::new());
        interp.registers.insert(
            ValueId::new(0),
            RegisterValue::Tensor(f32_tensor_with_shape(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        );

        interp
            .execute_op(
                &Op::Reduce { value: ValueId::new(0), axis: 1, op: ReduceKind::Sum },
                Some(ValueId::new(1)),
                &BTreeMap::new(),
            )
            .unwrap();
        let reduced_sum = register_tensor(&interp, 1);
        assert_eq!(reduced_sum.shape, vec![2]);
        assert_eq!(read_all_f32(reduced_sum), vec![6.0, 15.0]);

        interp
            .execute_op(
                &Op::Reduce { value: ValueId::new(0), axis: 0, op: ReduceKind::Mean },
                Some(ValueId::new(2)),
                &BTreeMap::new(),
            )
            .unwrap();
        let reduced_mean = register_tensor(&interp, 2);
        assert_eq!(reduced_mean.shape, vec![3]);
        assert_eq!(read_all_f32(reduced_mean), vec![2.5, 3.5, 4.5]);

        interp
            .registers
            .insert(ValueId::new(3), RegisterValue::Tensor(f32_tensor(&[2.0, 4.0, 6.0])));
        interp
            .execute_op(
                &Op::Reduce { value: ValueId::new(3), axis: 0, op: ReduceKind::Sum },
                Some(ValueId::new(4)),
                &BTreeMap::new(),
            )
            .unwrap();
        let scalar_reduce = register_tensor(&interp, 4);
        assert!(scalar_reduce.shape.is_empty());
        assert_eq!(read_f32(scalar_reduce, 0), 12.0);
    }

    #[test]
    fn interp_transpose_tensor_swaps_rows_and_cols() {
        let mut interp = Interpreter::new(BTreeMap::new(), ConstExprValues::new());
        interp.registers.insert(
            ValueId::new(0),
            RegisterValue::Tensor(f32_tensor_with_shape(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        );

        interp
            .execute_op(
                &Op::Transpose { value: ValueId::new(0) },
                Some(ValueId::new(1)),
                &BTreeMap::new(),
            )
            .unwrap();

        let transposed = register_tensor(&interp, 1);
        assert_eq!(transposed.shape, vec![3, 2]);
        assert_eq!(read_all_f32(transposed), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn interp_if_executes_then_or_else_block() {
        let mut then_block = Block::new(BlockId::new(1));
        then_block.push_op(Op::Const { value: 11 }, ValueId::new(10));
        then_block.push_op_no_result(Op::Store {
            dst: "out".into(),
            mask: None,
            indices: vec![IndexExpr::Const(0)],
            value: ValueId::new(10),
        });

        let mut else_block = Block::new(BlockId::new(2));
        else_block.push_op(Op::Const { value: 22 }, ValueId::new(20));
        else_block.push_op_no_result(Op::Store {
            dst: "out".into(),
            mask: None,
            indices: vec![IndexExpr::Const(0)],
            value: ValueId::new(20),
        });

        let mut blocks = BTreeMap::new();
        blocks.insert(then_block.id, then_block);
        blocks.insert(else_block.id, else_block);

        let mut inputs = BTreeMap::new();
        inputs.insert("out".into(), f32_tensor(&[0.0]));
        let mut interp = Interpreter::new(inputs, ConstExprValues::new());
        interp.registers.insert(ValueId::new(0), RegisterValue::Scalar(1.0));
        interp
            .execute_op(
                &Op::If {
                    cond: ValueId::new(0),
                    then_block: BlockId::new(1),
                    else_block: Some(BlockId::new(2)),
                },
                None,
                &blocks,
            )
            .unwrap();
        assert_eq!(read_f32(interp.inputs.get("out").unwrap(), 0), 11.0);

        let mut inputs = BTreeMap::new();
        inputs.insert("out".into(), f32_tensor(&[0.0]));
        let mut interp = Interpreter::new(inputs, ConstExprValues::new());
        interp.registers.insert(ValueId::new(0), RegisterValue::Scalar(0.0));
        interp
            .execute_op(
                &Op::If {
                    cond: ValueId::new(0),
                    then_block: BlockId::new(1),
                    else_block: Some(BlockId::new(2)),
                },
                None,
                &blocks,
            )
            .unwrap();
        assert_eq!(read_f32(interp.inputs.get("out").unwrap(), 0), 22.0);
    }

    #[test]
    fn interp_expand_dims_and_reshape_preserve_tensor_data() {
        let mut interp = Interpreter::new(BTreeMap::new(), ConstExprValues::new());
        interp.registers.insert(
            ValueId::new(0),
            RegisterValue::Tensor(f32_tensor_with_shape(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        );

        interp
            .execute_op(
                &Op::ExpandDims { value: ValueId::new(0), axis: 1 },
                Some(ValueId::new(1)),
                &BTreeMap::new(),
            )
            .unwrap();
        let expanded = register_tensor(&interp, 1);
        assert_eq!(expanded.shape, vec![2, 1, 3]);
        assert_eq!(read_all_f32(expanded), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        interp
            .execute_op(
                &Op::Reshape {
                    value: ValueId::new(1),
                    shape: Shape::new([3usize.into(), 2usize.into()]),
                },
                Some(ValueId::new(2)),
                &BTreeMap::new(),
            )
            .unwrap();
        let reshaped = register_tensor(&interp, 2);
        assert_eq!(reshaped.shape, vec![3, 2]);
        assert_eq!(read_all_f32(reshaped), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn interp_cat_concatenates_tensors_along_axis() {
        let mut interp = Interpreter::new(BTreeMap::new(), ConstExprValues::new());
        interp.registers.insert(
            ValueId::new(0),
            RegisterValue::Tensor(f32_tensor_with_shape(&[2, 2], &[1.0, 2.0, 3.0, 4.0])),
        );
        interp.registers.insert(
            ValueId::new(1),
            RegisterValue::Tensor(f32_tensor_with_shape(&[2, 1], &[5.0, 6.0])),
        );

        interp
            .execute_op(
                &Op::Cat { values: vec![ValueId::new(0), ValueId::new(1)], axis: 1 },
                Some(ValueId::new(2)),
                &BTreeMap::new(),
            )
            .unwrap();

        let cat = register_tensor(&interp, 2);
        assert_eq!(cat.shape, vec![2, 3]);
        assert_eq!(read_all_f32(cat), vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn interp_gather_reads_indexed_slices() {
        let mut inputs = BTreeMap::new();
        inputs.insert(
            "src".into(),
            f32_tensor_with_shape(&[2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        );
        let mut interp = Interpreter::new(inputs, ConstExprValues::new());
        interp
            .registers
            .insert(ValueId::new(0), RegisterValue::Tensor(i32_tensor_with_shape(&[2], &[3, 1])));

        interp
            .execute_op(
                &Op::Gather { src: "src".into(), indices: ValueId::new(0), axis: 1 },
                Some(ValueId::new(1)),
                &BTreeMap::new(),
            )
            .unwrap();

        let gathered = register_tensor(&interp, 1);
        assert_eq!(gathered.shape, vec![2, 2]);
        assert_eq!(read_all_f32(gathered), vec![4.0, 2.0, 8.0, 6.0]);
    }

    #[test]
    fn interp_scatter_writes_values_to_indexed_positions() {
        let mut inputs = BTreeMap::new();
        inputs.insert("dst".into(), f32_tensor_with_shape(&[2, 4], &[0.0; 8]));
        let mut interp = Interpreter::new(inputs, ConstExprValues::new());
        interp
            .registers
            .insert(ValueId::new(0), RegisterValue::Tensor(i32_tensor_with_shape(&[2], &[2, 0])));
        interp.registers.insert(
            ValueId::new(1),
            RegisterValue::Tensor(f32_tensor_with_shape(&[2, 2], &[10.0, 11.0, 20.0, 21.0])),
        );

        interp
            .execute_op(
                &Op::Scatter {
                    dst: "dst".into(),
                    indices: ValueId::new(0),
                    value: ValueId::new(1),
                    axis: 1,
                },
                None,
                &BTreeMap::new(),
            )
            .unwrap();

        let dst = interp.inputs.get("dst").unwrap();
        assert_eq!(read_all_f32(dst), vec![11.0, 0.0, 10.0, 0.0, 21.0, 0.0, 20.0, 0.0]);
    }

    #[test]
    fn interp_scan_supports_inclusive_and_exclusive_prefix_sum() {
        let mut interp = Interpreter::new(BTreeMap::new(), ConstExprValues::new());
        interp.registers.insert(
            ValueId::new(0),
            RegisterValue::Tensor(f32_tensor_with_shape(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        );

        interp
            .execute_op(
                &Op::Scan {
                    value: ValueId::new(0),
                    axis: 1,
                    op: ReduceKind::Sum,
                    exclusive: false,
                },
                Some(ValueId::new(1)),
                &BTreeMap::new(),
            )
            .unwrap();
        let inclusive = register_tensor(&interp, 1);
        assert_eq!(read_all_f32(inclusive), vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);

        interp
            .execute_op(
                &Op::Scan { value: ValueId::new(0), axis: 1, op: ReduceKind::Sum, exclusive: true },
                Some(ValueId::new(2)),
                &BTreeMap::new(),
            )
            .unwrap();
        let exclusive = register_tensor(&interp, 2);
        assert_eq!(read_all_f32(exclusive), vec![0.0, 1.0, 3.0, 0.0, 4.0, 9.0]);
    }
}
