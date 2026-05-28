//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MetalTile IR: SSA-form intermediate representation for tile-level kernels.
//!
//! The IR is the central data structure of the compiler. It is:
//! - **SSA-form**: every value is produced once, by one operation.
//! - **Explicit**: no implicit broadcasts, no hidden state.
//! - **Small**: designed to be traversed and transformed efficiently.
//!
//! ## Structure
//!
//! A [`Kernel`] contains:
//! - Parameters (tensor inputs/outputs with shapes)
//! - Constexpr declarations
//! - A body [`Block`] with a sequence of [`Op`]s
//!
//! ## Algorithm vs Schedule IR
//!
//! The algorithm IR (defined here) describes *what* to compute.
//! The schedule IR (in `metaltile-codegen`) annotates ops with *how* to compute it:
//! thread mapping, tile sizes, unroll factors, pipelining.

use std::{collections::BTreeMap, fmt};

use metaltile_macros::{OpFlags, ValueRefs, VariantName};
use rustc_hash::FxHashMap;

use crate::{constexpr::ConstExpr, dtype::DType, shape::Shape};

// ---------------------------------------------------------------------------
// ID types
// ---------------------------------------------------------------------------

/// Unique identifier for a value in the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ValueId(u32);

impl ValueId {
    pub const fn new(id: u32) -> Self { ValueId(id) }

    pub const fn as_u32(self) -> u32 { self.0 }
}

impl std::fmt::Display for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "%{}", self.0) }
}

/// Unique identifier for a block in the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct BlockId(u32);

impl BlockId {
    pub const fn new(id: u32) -> Self { BlockId(id) }

    pub const fn as_u32(self) -> u32 { self.0 }
}

/// Unique identifier for a loop/block-level variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct VarId(u32);

impl VarId {
    pub const fn new(id: u32) -> Self { VarId(id) }

    pub const fn as_u32(self) -> u32 { self.0 }
}

// ---------------------------------------------------------------------------
// Kernel-level types
// ---------------------------------------------------------------------------

/// How a kernel parameter is bound and represented in MSL.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ParamKind {
    /// `device T*` — a flat tensor buffer (default).
    #[default]
    Tensor,
    /// `device T*` + `constant uint* name_shape` + `constant uint* name_strides`
    /// — a strided tensor that also passes its shape and stride arrays.
    Strided,
    /// `constant T& name` — a single scalar value (e.g., `eps`, `scale`, `n`).
    Scalar,
}

/// A kernel parameter: a tensor input or output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    /// Human-readable name.
    pub name: String,
    /// Data type of the tensor elements.
    pub dtype: DType,
    /// Shape of the tensor.
    pub shape: Shape,
    /// Whether this is read-write (output) or read-only (input).
    pub is_output: bool,
    /// How this parameter is bound in Metal.
    pub kind: ParamKind,
}

/// A typed slot: used for inline MSL outputs and other typed holes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedSlot {
    pub dtype: DType,
    pub shape: Shape,
}

/// A constexpr declaration in the kernel signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstExprDecl {
    pub name: ConstExpr,
    /// The scalar type of this constexpr (default `U32`).
    pub dtype: DType,
    /// Optional fixed value if known at definition time.
    pub value: Option<usize>,
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

/// Unary math operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOpKind {
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Abs,
    Neg,
    Ceil,
    Floor,
    Recip,
    Sin,
    Cos,
    Erf,
    Exp2,
    Log2,
    Sign,
    Round,
    Trunc,
    /// Hyperbolic sine: `sinh(x)`.
    Sinh,
    /// Hyperbolic cosine: `cosh(x)`.
    Cosh,
    /// Tangent: `tan(x)`.
    Tan,
    /// Arc sine: `asin(x)`.
    Asin,
    /// Arc tangent: `atan(x)`.
    Atan,
    /// Inverse hyperbolic sine: `asinh(x)`.
    Asinh,
    /// Arc cosine: `acos(x)`.
    Acos,
    /// Inverse hyperbolic cosine: `acosh(x)`.
    Acosh,
    /// Inverse hyperbolic tangent: `atanh(x)`.
    Atanh,
    /// exp(x)-1 with high precision for small x: `expm1(x)`.
    Expm1,
    /// Base-10 logarithm: `log10(x)`.
    Log10,
    /// Inverse error function: `erfinv(x)`.
    ErfInv,
}

impl UnaryOpKind {
    /// Emit the MSL expression for this unary op applied to `arg`.
    pub fn msl_emit(self, arg: &str) -> String {
        match self {
            UnaryOpKind::Neg => format!("(-{arg})"),
            UnaryOpKind::Recip => format!("(1.0f / {arg})"),
            UnaryOpKind::Exp => format!("exp({arg})"),
            UnaryOpKind::Log => format!("log({arg})"),
            UnaryOpKind::Sqrt => format!("sqrt({arg})"),
            UnaryOpKind::Rsqrt => format!("rsqrt({arg})"),
            UnaryOpKind::Abs => format!("abs({arg})"),
            UnaryOpKind::Ceil => format!("ceil({arg})"),
            UnaryOpKind::Floor => format!("floor({arg})"),
            UnaryOpKind::Sin => format!("sin({arg})"),
            UnaryOpKind::Cos => format!("cos({arg})"),
            UnaryOpKind::Erf => format!("mt_erf_impl({arg})"),
            UnaryOpKind::Exp2 => format!("exp2({arg})"),
            UnaryOpKind::Log2 => format!("log2({arg})"),
            UnaryOpKind::Sign => format!("sign({arg})"),
            // rint() maps to the hardware RINT instruction (round-to-even, IEEE 754 default).
            // round() requires software emulation for half-away-from-zero on bfloat, making it
            // ~2× slower. MLX also uses rint() for its Round op (see unary.metal).
            UnaryOpKind::Round => format!("rint({arg})"),
            UnaryOpKind::Trunc => format!("trunc({arg})"),
            UnaryOpKind::Sinh => format!("sinh({arg})"),
            UnaryOpKind::Cosh => format!("cosh({arg})"),
            UnaryOpKind::Tan => format!("tan({arg})"),
            UnaryOpKind::Asin => format!("asin({arg})"),
            UnaryOpKind::Atan => format!("atan({arg})"),
            UnaryOpKind::Asinh => format!("asinh({arg})"),
            UnaryOpKind::Acos => format!("acos({arg})"),
            UnaryOpKind::Acosh => format!("acosh({arg})"),
            UnaryOpKind::Atanh => format!("atanh({arg})"),
            UnaryOpKind::Expm1 => format!("mt_expm1_impl({arg})"),
            UnaryOpKind::Log10 => format!("log10({arg})"),
            UnaryOpKind::ErfInv => format!("mt_erfinv_impl({arg})"),
        }
    }
}

/// Neural activation function kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActKind {
    Silu,
    Gelu,
    Relu,
    Tanh,
    Sigmoid,
}

impl ActKind {
    /// MSL helper function name. `Tanh` is a Metal built-in; others need a preamble helper.
    pub fn msl_fn(self) -> &'static str {
        match self {
            ActKind::Silu => "mt_silu",
            ActKind::Gelu => "mt_gelu",
            ActKind::Relu => "mt_relu",
            ActKind::Tanh => "tanh",
            ActKind::Sigmoid => "mt_sigmoid",
        }
    }

    /// Whether this activation needs a preamble helper function emitted before the kernel.
    pub fn needs_helper(self) -> bool { !matches!(self, ActKind::Tanh) }
}

/// Binary operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    And,
    Or,
    Xor,
    /// Less-than comparison (a < b). Result is bool.
    CmpLt,
    /// Greater-than comparison (a > b).
    CmpGt,
    /// Less-than-or-equal (a <= b).
    CmpLe,
    /// Greater-than-or-equal (a >= b).
    CmpGe,
    /// Equal-to (a == b).
    CmpEq,
    /// Not-equal (a != b).
    CmpNe,
    /// Power: a^b (maps to MSL `pow(a, b)`).
    Pow,
    /// Arc tangent of y/x: `atan2(y, x)`.
    ATan2,
    /// Floating-point remainder: `fmod(a, b)`.
    Rem,
    /// Left shift: a << b.
    Shl,
    /// Right shift: a >> b.
    Shr,
    /// Bitwise AND (integer).
    BitAnd,
    /// Bitwise OR (integer).
    BitOr,
    /// Bitwise XOR (integer).
    BitXor,
    /// Integer modulo: `a % b`.
    Mod,
}

impl BinOpKind {
    pub fn msl_symbol(self) -> &'static str {
        match self {
            BinOpKind::Add => "+",
            BinOpKind::Sub => "-",
            BinOpKind::Mul => "*",
            BinOpKind::Div => "/",
            BinOpKind::Max => "max",
            BinOpKind::Min => "min",
            BinOpKind::And => "&&",
            BinOpKind::Or => "||",
            BinOpKind::Xor => "^",
            BinOpKind::CmpLt => "<",
            BinOpKind::CmpGt => ">",
            BinOpKind::CmpLe => "<=",
            BinOpKind::CmpGe => ">=",
            BinOpKind::CmpEq => "==",
            BinOpKind::CmpNe => "!=",
            BinOpKind::Pow => "pow",
            BinOpKind::ATan2 => "atan2",
            BinOpKind::Rem => "fmod",
            BinOpKind::Shl => "<<",
            BinOpKind::Shr => ">>",
            BinOpKind::BitAnd => "&",
            BinOpKind::BitOr => "|",
            BinOpKind::BitXor => "^",
            BinOpKind::Mod => "%",
        }
    }

    /// Whether this op produces a boolean result.
    pub fn is_cmp(self) -> bool {
        matches!(
            self,
            BinOpKind::CmpLt
                | BinOpKind::CmpGt
                | BinOpKind::CmpLe
                | BinOpKind::CmpGe
                | BinOpKind::CmpEq
                | BinOpKind::CmpNe
        )
    }

    /// Whether this op is emitted as `fn(a, b)` rather than infix `a op b`.
    pub fn is_fn_call(self) -> bool {
        matches!(
            self,
            BinOpKind::Max | BinOpKind::Min | BinOpKind::Pow | BinOpKind::ATan2 | BinOpKind::Rem
        )
    }
}

/// Reduction kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceKind {
    Sum,
    Max,
    Min,
    Mean,
    Product,
}

/// Atomic operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomicKind {
    Add,
    Max,
    Min,
    And,
    Or,
    Xor,
}

/// Memory scope for an atomic op.  Drives whether the MSL emitter
/// treats `dst` as a device-memory buffer (typical kernel param) or a
/// threadgroup-allocated array (needs reinterpret-cast to
/// `threadgroup atomic_uint*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AtomicScope {
    /// Device memory — `dst` is a kernel buffer parameter.  Emits
    /// `atomic_fetch_<op>_explicit(dst + idx, val, memory_order_relaxed)`.
    #[default]
    Device,
    /// Threadgroup memory — `dst` is a `threadgroup_alloc`'d array.
    /// Emits `atomic_fetch_<op>_explicit((threadgroup atomic_uint*)&dst[idx], …)`.
    /// AURA encode's pack stage uses this so threads racing on the same
    /// u32 word are properly serialised.
    Threadgroup,
}

impl AtomicKind {
    pub fn msl_fn(self) -> &'static str {
        match self {
            AtomicKind::Add => "atomic_fetch_add_explicit",
            AtomicKind::Max => "atomic_fetch_max_explicit",
            AtomicKind::Min => "atomic_fetch_min_explicit",
            AtomicKind::And => "atomic_fetch_and_explicit",
            AtomicKind::Or => "atomic_fetch_or_explicit",
            AtomicKind::Xor => "atomic_fetch_xor_explicit",
        }
    }
}

/// Index expression for loads/stores.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndexExpr {
    /// An SSA value used as an index.
    Value(ValueId),
    /// A constant.
    Const(i64),
    /// A range: value..value+offset.
    Range(ValueId, i64),
}

impl IndexExpr {
    /// The `ValueId` embedded in this index expression, if any.
    /// Both `Value` and `Range` carry a `ValueId`; `Const` does not.
    pub fn value_id(&self) -> Option<&ValueId> {
        match self {
            IndexExpr::Value(v) | IndexExpr::Range(v, _) => Some(v),
            IndexExpr::Const(_) => None,
        }
    }

    pub fn value_id_mut(&mut self) -> Option<&mut ValueId> {
        match self {
            IndexExpr::Value(v) | IndexExpr::Range(v, _) => Some(v),
            IndexExpr::Const(_) => None,
        }
    }
}

/// An argument to a cross-kernel call ([`Op::KernelCall`]).
///
/// - [`KernelCallArg::Value`]: a computed scalar value in the caller's SSA.
///   The call's result is the callee's single output value.
/// - [`KernelCallArg::Tensor`]: a buffer / constexpr name in the caller.
///   Substituted for all loads/stores referencing that name in the callee's IR.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KernelCallArg {
    Value(ValueId),
    Tensor(String),
}

impl KernelCallArg {
    /// Returns the [`ValueId`] if this is a [`Value`][Self::Value] variant.
    pub fn as_value(&self) -> Option<&ValueId> {
        if let KernelCallArg::Value(v) = self { Some(v) } else { None }
    }

    /// Returns a mutable reference to the inner [`ValueId`] if this is a
    /// [`Value`][Self::Value] variant, or `None` for [`Tensor`][Self::Tensor].
    pub fn as_value_mut(&mut self) -> Option<&mut ValueId> {
        if let KernelCallArg::Value(v) = self { Some(v) } else { None }
    }
}

/// A single operation in the IR.
#[derive(Debug, Clone, PartialEq, ValueRefs, OpFlags, VariantName)]
pub enum Op {
    /// `program_id(axis)` — which block this threadgroup handles along an axis.
    #[cheap_alu]
    #[result_u32]
    ProgramId { axis: u32 },

    /// A constant integer value (from a literal in the DSL).
    #[cheap_alu]
    #[op_const]
    #[result_i32]
    Const { value: i64 },

    /// `arange(start, step, len)` — creates a 1D range [start, start+step, ...].
    /// `start` and `step` default to 0.0 and 1.0 respectively.
    #[shape_op]
    #[result_custom]
    Arange { start: Option<f64>, step: Option<f64>, len: ConstExpr },

    /// Load a tile from a tensor at given indices.
    #[op_load]
    #[result_custom]
    Load {
        /// The parameter to load from.
        src: String,
        /// Per-dimension index expressions.
        #[vid_exprs]
        indices: Vec<IndexExpr>,
        /// Optional mask: load only where mask is true (false → fill with `other`).
        #[vid_opt]
        mask: Option<ValueId>,
        /// Fill value when mask is false (default 0.0).
        other: Option<f64>,
    },

    /// Store a tile to a tensor at given indices.
    #[side_effect]
    #[op_store]
    #[no_result]
    Store {
        /// The parameter to store to.
        dst: String,
        /// Per-dimension index expressions.
        #[vid_exprs]
        indices: Vec<IndexExpr>,
        /// The value to store.
        #[vid]
        value: ValueId,
        /// Optional mask: store only where mask is true.
        #[vid_opt]
        mask: Option<ValueId>,
    },

    /// Elementwise binary operation.
    #[elementwise]
    #[cheap_alu]
    #[result_same_type]
    BinOp {
        op: BinOpKind,
        #[vid]
        lhs: ValueId,
        #[vid]
        rhs: ValueId,
    },

    /// Fused multiply-add: `a * b + c` as a single op.  Lowers to
    /// `fma(a, b, c)` in MSL.  Floats-only — the GPU FMA path requires
    /// IEEE float operands.
    ///
    /// Created by `FmaFusionPass` from the pattern
    ///     `Op::Mul(a, b) → v_mul`
    ///     `Op::Add(v_mul, c) → v_add`   (or `Sub(v_mul, c)` with `c` negated)
    /// when `v_mul` has no consumers other than the absorbing Add/Sub
    /// AND all operand types are floats (`Mul` of integers stays as a
    /// plain BinOp; `Add(i64, i64)` doesn't have an FMA path either).
    /// The Mul becomes dead and is swept up by the DCE pass.
    ///
    /// Pre-fix, this fusion lived as an emit-time peephole in
    /// `emit_block.rs` that turned the textual `auto v_add = v_mul +
    /// c;` into `auto v_add = fma(a, b, c);` while leaving the
    /// upstream `Op::Mul` in the IR — the standalone Mul then emitted
    /// `auto v_mul = a * b;` as a dead variable in MSL.  #207 worked
    /// around it with `compute_fma_absorbed_mul_skips` in the emit
    /// path.  Lifting the fusion into the IR makes the Mul orphan a
    /// real producer-with-no-consumer and lets DCE handle it
    /// naturally.
    #[elementwise]
    #[cheap_alu]
    #[result_same_type]
    Fma {
        #[vid]
        a: ValueId,
        #[vid]
        b: ValueId,
        #[vid]
        c: ValueId,
    },

    /// Tile matrix multiply: `dot(a, b)`.
    #[result_custom]
    Dot {
        #[vid]
        a: ValueId,
        #[vid]
        b: ValueId,
    },

    /// Reduction along an axis.
    #[result_custom]
    Reduce {
        #[vid]
        value: ValueId,
        axis: u32,
        op: ReduceKind,
    },

    /// Per-thread strided reduction over a device buffer.
    /// Reduces `src[offset]`, `src[offset+stride]`, `src[offset+2*stride]`, ... while index < `end`.
    /// If `transform` is set, the op is applied to each loaded element before accumulation.
    #[result_custom]
    StrideReduce {
        src: String,
        /// First index to load (= tid for intra-row; = row*N + tid for full buffer).
        #[vid]
        offset: ValueId,
        /// Step between successive loads (= lsize).
        #[vid]
        stride: ValueId,
        /// Exclusive upper bound (= N for intra-row; = row*N + N for full buffer).
        #[vid]
        end: ValueId,
        op: ReduceKind,
        dtype: DType,
        /// Optional per-element transform chain applied to the loaded value before accumulation.
        /// Each op in the chain takes the previous result as input.
        transform: Option<Vec<Op>>,
        /// For dot-product reductions (GEMV): multiply each `src[_i]` by `secondary_src[_i - secondary_base]`.
        secondary_src: Option<String>,
        /// Base offset subtracted from the loop index when accessing secondary_src.
        #[vid_opt]
        secondary_base: Option<ValueId>,
    },

    /// Type cast.
    #[elementwise]
    #[cheap_alu]
    #[result_custom]
    Cast {
        #[vid]
        value: ValueId,
        dtype: DType,
    },

    /// Loop: iterate a variable from start to end with step.
    #[unpredictable]
    #[op_loop]
    #[no_result]
    Loop {
        var: VarId,
        #[vid]
        start: ValueId,
        #[vid]
        end: ValueId,
        #[vid]
        step: ValueId,
        body: BlockId,
    },

    /// Conditional branch: if `cond` is true, execute `then_block`, else `else_block`.
    #[unpredictable]
    #[op_if]
    #[no_result]
    If {
        #[vid]
        cond: ValueId,
        then_block: BlockId,
        else_block: Option<BlockId>,
    },

    /// Create a zero-filled tile.
    #[elementwise]
    #[result_custom]
    Zeros {
        dtype: DType,
        /// Shape of the tile (usually a 2D tile).
        shape: Shape,
    },

    /// Transpose a 2D tile.
    #[shape_op]
    #[result_custom]
    Transpose {
        #[vid]
        value: ValueId,
    },

    /// Insert a size-1 dimension at `axis`. Zero-cost reshape.
    #[result_same_type]
    #[shape_op]
    ExpandDims {
        #[vid]
        value: ValueId,
        axis: u32,
    },

    /// Reshape a tile to a new shape (same element count). Zero-cost if contiguous.
    #[result_same_type]
    #[shape_op]
    Reshape {
        #[vid]
        value: ValueId,
        shape: Shape,
    },

    /// Concatenate tiles along `axis`.
    #[result_custom]
    Cat {
        #[vid_vec]
        values: Vec<ValueId>,
        axis: u32,
    },

    /// Extract a slice of a tile.
    #[shape_op]
    #[result_custom]
    Slice {
        #[vid]
        value: ValueId,
        /// Which dimensions to slice; (axis, start_offset, length).
        ranges: Vec<(u32, i64, i64)>,
    },

    /// Inline raw MSL code. Escape hatch.
    #[result_custom]
    InlineMsl {
        source: String,
        #[vid_vec]
        inputs: Vec<ValueId>,
        outputs: Vec<TypedSlot>,
    },

    // ---- Scalar / element-wise math ----
    /// Unary math operation: exp, log, sqrt, rsqrt, abs, neg, ceil, floor, recip.
    #[elementwise]
    #[cheap_alu]
    #[result_same_type]
    UnaryOp {
        op: UnaryOpKind,
        #[vid]
        value: ValueId,
    },

    /// Neural activation function: silu, gelu, relu, tanh, sigmoid.
    #[elementwise]
    #[result_same_type]
    Activation {
        kind: ActKind,
        #[vid]
        value: ValueId,
    },

    /// Conditional select: `cond ? on_true : on_false`.
    /// Maps to MSL `select(on_false, on_true, bool(cond))`.
    #[elementwise]
    #[cheap_alu]
    #[result_same_type]
    Select {
        #[vid]
        cond: ValueId,
        #[vid]
        on_true: ValueId,
        #[vid]
        on_false: ValueId,
    },

    /// Broadcast a scalar value to fill a tile shape (replication, no copy to device memory).
    #[elementwise]
    #[result_custom]
    Broadcast {
        #[vid]
        value: ValueId,
        shape: Shape,
    },

    /// Create a tile filled with a constant floating-point value (generalization of Zeros).
    #[elementwise]
    #[result_custom]
    Splat { value: f64, dtype: DType, shape: Shape },

    /// Fused chain of elementwise operations.
    /// Created by the FusionPass to merge adjacent ops like
    /// `UnaryOp(Exp) → Activation(Silu)` into a single expression.
    #[op_fused]
    #[result_custom]
    FusedElementwise {
        /// The elementwise ops in execution order (producer first).
        /// Each op's inputs reference either external ValueIds or
        /// the output of a preceding op in this chain (index 0..n-1).
        #[vid_recursive]
        ops: Vec<Op>,
    },

    /// Vectorized load: loads `len` consecutive elements as a vector.
    /// `len` is 2, 4, or 8. Created by the VectorizePass from consecutive scalar Loads.
    #[op_load]
    #[result_custom]
    VectorLoad {
        /// The parameter to load from.
        src: String,
        /// Flat byte offset into the buffer (already aligned).
        #[vid]
        byte_offset: ValueId,
        /// Number of elements: 2, 4, or 8.
        len: u32,
    },

    /// Vectorized store: stores `len` consecutive elements as a vector.
    #[side_effect]
    #[op_store]
    #[no_result]
    VectorStore {
        /// The parameter to store to.
        dst: String,
        /// Flat byte offset into the buffer (already aligned).
        #[vid]
        byte_offset: ValueId,
        /// Number of elements: 2, 4, or 8.
        len: u32,
        /// The value to store (scalar or vector ValueId).
        #[vid]
        value: ValueId,
    },

    /// Project one scalar lane (0..len) out of a VectorLoad result.
    /// Emitted by VectorizePass to feed each original scalar consumer.
    #[result_custom]
    VectorExtract {
        #[vid]
        vec: ValueId,
        lane: u32,
    },

    /// Pack scalar values into a vector.
    ///
    /// Assembles `elements` into a single vector value.  Used by the vectorize
    /// pass to fuse stores with interleaved computation: collect the stored
    /// values, emit a Pack op, then emit a single VectorStore referencing it.
    ///
    /// The MSL emitter lowers this to a vector constructor: `float4(v0, v1, v2, v3)`.
    #[result_custom]
    Pack {
        /// The element data type (determines the vector type: float4, half4, bfloat4).
        dtype: DType,
        /// Scalar values to pack, in order.
        #[vid_vec]
        elements: Vec<ValueId>,
    },

    /// Gather: indexed load from a buffer. `out[i] = src[indices[i]]`.
    #[result_custom]
    Gather {
        src: String,
        #[vid]
        indices: ValueId,
        axis: u32,
    },

    /// Scatter: indexed store to a buffer. `dst[indices[i]] = value[i]`.
    #[side_effect]
    #[op_store]
    #[no_result]
    Scatter {
        dst: String,
        #[vid]
        indices: ValueId,
        #[vid]
        value: ValueId,
        axis: u32,
    },

    /// Atomic operation on device memory.
    #[side_effect]
    #[no_result]
    Atomic {
        op: AtomicKind,
        scope: AtomicScope,
        dst: String,
        #[vid]
        index: ValueId,
        #[vid]
        value: ValueId,
    },

    /// Prefix scan along an axis (inclusive or exclusive).
    #[result_same_type]
    Scan {
        #[vid]
        value: ValueId,
        axis: u32,
        op: ReduceKind,
        exclusive: bool,
    },

    /// Serial inclusive prefix scan over a contiguous slice of a device buffer.
    /// Writes `dst[i] = src[offset] + src[offset+1] + ... + src[i]` for i in [offset, end).
    /// Single-threaded: dispatch with [B, 1, 1] × [1, 1, 1] (one thread per row).
    #[unpredictable]
    #[result_custom]
    StrideScan {
        src: String,
        dst: String,
        #[vid]
        offset: ValueId,
        #[vid]
        end: ValueId,
        op: ReduceKind,
    },

    /// Serial argmax/argmin over a contiguous slice of a device buffer.
    /// Returns the flat index of the extreme element in [offset, end).
    /// Single-threaded: dispatch with [1, 1, 1] × [1, 1, 1] for a single row.
    #[unpredictable]
    #[result_u32]
    StrideArgReduce {
        src: String,
        #[vid]
        offset: ValueId,
        #[vid]
        end: ValueId,
        op: ReduceKind,
    },

    /// Strided per-element compute + store: for each element in the stride pattern,
    /// load from src, apply optional transform with a scalar operand, and store to dst.
    /// Used for write-back in reduction kernels (e.g., rout[i] = rx[i] * rms * w[i]).
    #[side_effect]
    #[op_store]
    #[no_result]
    StrideStore {
        src: String,
        dst: String,
        #[vid]
        offset: ValueId,
        #[vid]
        end: ValueId,
        /// First operand: the scalar from the reduction step (e.g., rms, mean, 1/std).
        #[vid]
        scalar: ValueId,
        /// Optional second operand: another device buffer (e.g., w[i] for weighted norm).
        aux_src: Option<String>,
    },

    // ---- SIMD-group and threadgroup primitives ----
    /// SIMD-group reduction: reduce all lanes within the SIMD group.
    /// Maps to `simd_sum(v)`, `simd_max(v)`, `simd_min(v)` (Metal 2.1+).
    #[result_same_type]
    SimdReduce {
        #[vid]
        value: ValueId,
        op: ReduceKind,
    },

    /// SIMD-group butterfly shuffle: `simd_shuffle_xor(value, mask)`.
    /// Used by Steel attention row reductions, where lanes sharing the same
    /// MMA row exchange values through fixed xor masks (for example 1 and 8).
    #[result_same_type]
    SimdShuffleXor {
        #[vid]
        value: ValueId,
        mask: u32,
    },

    /// Allocate a simdgroup matrix of shape M×N with given element type.
    /// Emits `simdgroup_matrix<T, M, N> name;` in MSL.
    #[needs_simdgroup_matrix]
    #[result_f32_scalar]
    SimdgroupAlloc { dtype: DType, m: u32, n: u32 },

    /// Load one element from a simdgroup matrix: `result = name.thread_elements()[index]`.
    /// Produces a scalar value.
    #[needs_simdgroup_matrix]
    #[result_f32_scalar]
    SimdgroupElemLoad {
        #[vid]
        value: ValueId,
        index: u32,
    },

    /// Store one element into a simdgroup matrix: `name.thread_elements()[index] = data`.
    /// No result (side-effecting).
    #[needs_simdgroup_matrix]
    #[no_result]
    SimdgroupElemStore {
        #[vid]
        value: ValueId,
        index: u32,
        #[vid]
        data: ValueId,
    },

    /// Hardware-fused simdgroup load: fill all 64 elements of an 8×8
    /// `simdgroup_matrix<T,M,N>` from a contiguous threadgroup-memory tile
    /// in one MSL `simdgroup_load(matrix, &tg[offset], stride, origin,
    /// transpose)` instruction.
    /// Bypasses the per-lane scatter of repeated `simdgroup_elem_store(
    /// frag, idx, threadgroup_load(...))`, which suffers TG-bank conflicts
    /// at f16 stride geometries (see `qmm_mma_ftrans_report.md` §7).
    /// `offset` is a ValueId computing the starting element offset of the
    /// fragment's top-left corner inside the named TG array. `stride` is
    /// the row stride in elements (const). `transpose=true` swaps the row
    /// and column dimensions of the loaded fragment — used to load a B
    /// operand stored row-major `[N, K]` as if it were `[K, N]` for the
    /// standard `C = A * B` MMA layout (MLX `qmm_t` pattern).
    #[needs_simdgroup_matrix]
    #[no_result]
    SimdgroupLoad {
        #[vid]
        dest: ValueId,
        tg: String,
        #[vid]
        offset: ValueId,
        stride: u32,
        transpose: bool,
    },

    /// simdgroup multiply-accumulate: `C = A * B + C`.
    /// All three operands must be simdgroup matrices of compatible shapes.
    #[needs_simdgroup_matrix]
    #[no_result]
    SimdgroupMatMul {
        #[vid]
        a: ValueId,
        #[vid]
        b: ValueId,
        #[vid]
        c: ValueId,
    },

    /// Built-in: returns the SIMD lane index (thread_index_in_simdgroup).
    #[needs_simd_lane]
    #[result_u32]
    #[shape_op]
    SimdLaneId,

    /// Built-in: returns the SIMD group index (simdgroup_index_in_threadgroup).
    #[needs_simd_group]
    #[result_u32]
    #[shape_op]
    SimdGroupId,

    /// SIMD-group inclusive prefix scan.
    /// Maps to `simd_scan_inclusive_<op>(v)` (Metal 3.0+).
    #[result_f32_scalar]
    SimdScan {
        #[vid]
        value: ValueId,
        op: ReduceKind,
        exclusive: bool,
    },

    /// SIMD-group broadcast: every lane receives the value held by the
    /// specified `lane` (a u32 index 0..simd_size). Maps to
    /// `simd_broadcast(v, lane)` (Metal 2.1+). Cooperative codebook hoist
    /// in AURA score/value kernels uses this to share one lane's loaded
    /// codebook word across the group.
    #[result_same_type]
    SimdBroadcast {
        #[vid]
        value: ValueId,
        #[vid]
        lane: ValueId,
    },

    /// Allocate a named threadgroup (shared) memory array.
    /// Emits `threadgroup T name[size]` in the kernel body.
    #[side_effect]
    #[unpredictable]
    #[no_result]
    ThreadgroupAlloc {
        dtype: DType,
        /// Number of elements in the array.
        size: u32,
        /// Variable name for the threadgroup array in MSL.
        name: String,
    },

    /// Load one element from a named threadgroup array: `val = name[index]`.
    #[op_load]
    #[result_custom]
    ThreadgroupLoad {
        name: String,
        #[vid]
        index: ValueId,
    },

    /// Store one element to a named threadgroup array: `name[index] = value`.
    #[side_effect]
    #[op_store]
    #[no_result]
    ThreadgroupStore {
        name: String,
        #[vid]
        index: ValueId,
        #[vid]
        value: ValueId,
    },

    /// Allocate a per-thread stack-resident array.  Emits `T name[size];`
    /// inside the kernel body (no `threadgroup` qualifier — each thread
    /// gets its own copy).  Metal keeps small fixed-size stack arrays in
    /// registers; AURA flash kernels need this for `q_vals[DIMS_PER_LANE]`,
    /// `o[DIMS_PER_LANE]`, and the per-thread codebook cache that
    /// amortises lookup across the dim-strided inner loop.
    #[side_effect]
    #[unpredictable]
    #[no_result]
    StackAlloc { dtype: DType, size: u32, name: String },

    /// Load one element from a per-thread stack array: `val = name[index]`.
    /// Identical emission to `ThreadgroupLoad`; kept distinct in the IR so
    /// liveness / scoping passes know the buffer is thread-private.
    #[op_load]
    #[result_custom]
    StackLoad {
        name: String,
        #[vid]
        index: ValueId,
    },

    /// Store one element to a per-thread stack array: `name[index] = value`.
    #[side_effect]
    #[no_result]
    StackStore {
        name: String,
        #[vid]
        index: ValueId,
        #[vid]
        value: ValueId,
    },

    /// Threadgroup barrier: `threadgroup_barrier(mem_flags::mem_threadgroup)`.
    /// Ensures all prior threadgroup stores are visible to all threads before
    /// any subsequent threadgroup loads.
    #[side_effect]
    #[unpredictable]
    #[barrier]
    #[no_result]
    Barrier,

    /// Compiler-only simdgroup barrier: `simdgroup_barrier(mem_flags::mem_none)`.
    /// Zero-cost at runtime — pins instruction ordering across the simdgroup
    /// so the compiler can't hoist a subsequent matmul/load past a prior one.
    /// Apple MLX uses these around V-tile loads when BD≥128
    /// (`steel_attention.h:431-443`) to keep `simdgroup_load → simdgroup_mma`
    /// ordering stable through aggressive scheduling.
    #[side_effect]
    #[unpredictable]
    #[barrier]
    #[no_result]
    SimdgroupBarrier,

    /// Declare a mutable register-local scalar variable.
    /// Emits: `auto __ml_{name} = {init_value};`
    /// Used for loop-carried state (running prefix, best_val/best_idx, etc.).
    #[unpredictable]
    #[result_custom]
    DeclareLocal {
        name: String,
        #[vid]
        value: ValueId,
    },

    /// Assign to a mutable register-local scalar variable.
    /// Emits: `__ml_{name} = {value};`
    #[side_effect]
    #[unpredictable]
    #[no_result]
    SetLocal {
        name: String,
        #[vid]
        value: ValueId,
    },

    /// Return the index of the min/max element along an axis.
    #[result_u32]
    ArgReduce {
        #[vid]
        value: ValueId,
        axis: u32,
        op: ReduceKind,
    },

    /// Cross-kernel call: inline another kernel's computation at this site.
    ///
    /// Resolved by `KernelInlinePass` (runs as the first pass in the
    /// standard pipeline) so all subsequent passes see only flat scalar ops.
    /// `callee` is the registered kernel name; `args` are positionally
    /// matched to the callee's params; `dtype` is the generic type.
    #[result_custom]
    KernelCall { callee: String, args: Vec<KernelCallArg>, dtype: DType },

    // ---- Cooperative register-tile matmul primitives ----
    //
    // These six ops model tile-level matrix multiply using a "cooperative
    // register tile" — a matrix fragment distributed across all threads in a
    // simdgroup / threadgroup.  Together they can express any tiled matmul
    // kernel, including multi-K-block loops, quantized weights, and MoE
    // expert routing.  Control flow (K-loops, barriers, dequant, gating)
    // stays as ordinary DSL ops; only the tile-matmul operations themselves
    // need these primitives.
    //
    // Codegen mapping:
    //   Metal 4 (macOS 26+, gen ≥17): `mpp::tensor_ops::matmul2d` + `cooperative_tensor`
    //   Metal 2/3 fallback:            zero-fill stub (metallib links but gives zeros)
    //
    // All six ops emit `#if __METAL_VERSION__ >= 400` guards so that
    // metallibs still link on older toolchains.
    /// Declare descriptor + A/B/C register tiles as kernel-local variables.
    ///
    /// `name` is a logical prefix: `{name}_op`, `{name}_ct_a`, `{name}_ct_b`, `{name}_ct_c`.
    /// Must appear in the kernel body before any other `CoopTile*` op with the same name.
    CoopTileSetup {
        name: String,
        m: u32,
        n: u32,
        k: u32,
        /// Transpose A before multiplying.
        ta: bool,
        /// Transpose B before multiplying.
        tb: bool,
        /// Transpose C on store.
        tc: bool,
        /// Whether C accumulates across calls or is overwritten on each run.
        acc_mode: CoopTileAccMode,
        /// How many threads cooperate per tile operation.
        exec_scope: CoopTileScope,
        /// Element type for A and B tiles.
        act_dtype: DType,
        /// Accumulator element type for C (typically F32).
        acc_dtype: DType,
        /// When `true`, A and B are passed as direct `metal::tensor` views instead of
        /// cooperative tensors (required when M ∉ {16, 32}, e.g. M=8).
        /// The setup emits `using {name}_tA_t` / `{name}_tB_t` type aliases and only
        /// `{name}_ct_c` (no ct_a/ct_b).
        direct_inputs: bool,
        /// Address space for the A type alias when `direct_inputs = true`.
        a_is_tg: bool,
        /// Inner extent for the A type alias (`metal::extents<int, A_EI, A_EO>`).
        a_ei: u32,
        /// Outer extent for the A type alias.
        a_eo: u32,
        /// Address space for the B type alias when `direct_inputs = true`.
        b_is_tg: bool,
        /// Inner extent for the B type alias.
        b_ei: u32,
        /// Outer extent for the B type alias.
        b_eo: u32,
    },

    /// Zero every element of the C accumulator tile.
    /// Call once before the K-loop when `acc_mode = Accumulate`.
    CoopTileZero { name: String },

    /// Load the A register tile from a `tensor_inline` view over `ptr_name[+ptr_offset]`.
    ///
    /// `dtype` is the element type of the A/B buffers (must match `act_dtype` in the setup).
    /// `ei` / `eo` are the inner and outer extents for `metal::extents<int, EI, EO>`.
    ///
    /// When `direct = true` (direct-input mode), emits only the `{name}_tA_t {name}_tA(...)`
    /// tensor-view declaration without calling `{name}_ct_a.load()`.
    CoopTileLoadA {
        name: String,
        ptr_name: String,
        #[vid_opt]
        ptr_offset: Option<ValueId>,
        is_tg: bool,
        dtype: DType,
        ei: u32,
        eo: u32,
        /// See `CoopTileSetup::direct_inputs`.
        direct: bool,
    },

    /// Load the B register tile from a `tensor_inline` view.
    ///
    /// When `direct = true`, emits only `{name}_tB_t {name}_tB(...)` without `.load()`.
    CoopTileLoadB {
        name: String,
        ptr_name: String,
        #[vid_opt]
        ptr_offset: Option<ValueId>,
        is_tg: bool,
        dtype: DType,
        ei: u32,
        eo: u32,
        /// See `CoopTileSetup::direct_inputs`.
        direct: bool,
    },

    /// Execute A·B → C (accumulates if `acc_mode = Accumulate`).
    ///
    /// When `direct = true`, calls `{name}_op.run({name}_tA, {name}_tB, {name}_ct_c)`
    /// (direct tensor views) instead of the cooperative-tensor overload.
    CoopTileRun {
        name: String,
        /// See `CoopTileSetup::direct_inputs`.
        direct: bool,
    },

    /// Store the C register tile through a `tensor_inline` view into `ptr_name[+ptr_offset]`.
    ///
    /// `dtype` is the accumulator element type (must match `acc_dtype` in the setup).
    CoopTileStoreC {
        name: String,
        ptr_name: String,
        #[vid_opt]
        ptr_offset: Option<ValueId>,
        is_tg: bool,
        dtype: DType,
        ei: u32,
        eo: u32,
    },
}

// ---------------------------------------------------------------------------
// CoopTile enums
// ---------------------------------------------------------------------------

/// Execution scope for cooperative tile operations.
///
/// Maps to `metal::execution_simdgroup` / `metal::execution_threadgroup` in
/// Metal 4 but is named hardware-agnostically so the same IR works with other
/// backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoopTileScope {
    /// One simdgroup (32 lanes) cooperates per tile.
    SimdGroup,
    /// The whole threadgroup cooperates per tile.
    Threadgroup,
}

/// Accumulation mode for a cooperative tile matmul.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoopTileAccMode {
    /// C tile is overwritten with A·B (no prior accumulation).
    Overwrite,
    /// C tile accumulates: `C += A·B` across K-block iterations.
    MultiplyAccumulate,
}

// ---------------------------------------------------------------------------
// KernelMode
// ---------------------------------------------------------------------------

/// Controls which Metal built-in position attributes are emitted in the
/// kernel signature.  All built-in attributes **must** share the same vector
/// width (Metal constraint), so each mode is self-contained.
///
/// ## Which mode emits which alias
///
/// Kernel bodies refer to position aliases by name (`tid`, `tgid_x`,
/// `tgid_y`, `tgid_z`, `lsize`, `simd_lane`, `simd_group`, `n_simd`).
/// Each mode emits a different subset of those:
///
/// | mode         | tid | tgid_x | tgid_y | tgid_z | lsize | simd_lane | simd_group |
/// |--------------|:---:|:------:|:------:|:------:|:-----:|:---------:|:----------:|
/// | Elementwise  |  ✓  |        |        |        |       |           |            |
/// | Reduction    |  ✓  |   ✓    |   ✓ †  |        |   ✓   |           |            |
/// | Grid3D       |  ✓  |        |        |        |       |           |            |
/// | Tile2D       |  ✓  |   ✓    |   ✓    |        |       |           |            |
/// | SimdGroup2D  |     |   ✓    |   ✓    |   ✓    |       |     ✓     |     ✓      |
///
/// † Reduction emits `tgid_y` only when the kernel actually references
/// axis 1 (avoids `-Wunused-variable`). Reduction does **not** emit
/// `tgid_z` — kernels needing 3-axis grid + simdgroup primitives must
/// use SimdGroup2D.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KernelMode {
    /// `uint tid [[thread_position_in_grid]]`
    /// Used for flat elementwise kernels.
    #[default]
    Elementwise,
    /// `uint3 _tid3/tgid3/lsize3` with `.x`/`.y` aliases injected.
    /// Used for row-reduction kernels (softmax, rms_norm, layer_norm, …).
    Reduction,
    /// `uint3 gid [[thread_position_in_grid]]`
    /// Used for 3-axis grid kernels (rope).
    Grid3D,
    /// `uint2 tid [[thread_position_in_threadgroup]] + uint2 tgid`
    /// Used for tiled 2-D kernels (gemv, matmul).
    Tile2D,
    /// `uint3 tid [[threadgroup_position_in_grid]]` + `uint3 lid` +
    /// `uint simd_lane` + `uint simd_group`.
    /// Used for tiled simdgroup-matmul kernels (steel GEMM) and
    /// any 3-axis kernel that needs `tgid_z` (e.g. batched SDPA
    /// prefill).
    SimdGroup2D,
}

impl fmt::Display for KernelMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            KernelMode::Elementwise => "Elementwise",
            KernelMode::Reduction => "Reduction",
            KernelMode::Grid3D => "Grid3D",
            KernelMode::Tile2D => "Tile2D",
            KernelMode::SimdGroup2D => "SimdGroup",
        })
    }
}

// ---------------------------------------------------------------------------
// Block & Kernel
// ---------------------------------------------------------------------------

/// A basic block: a sequence of operations with a terminator.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub id: BlockId,
    /// Operations in this block.
    pub ops: Vec<Op>,
    /// Parallel to `ops`: the SSA value ID produced by each op, or `None` for
    /// no-result ops (Store, Loop, Barrier, etc.).
    /// Invariant: `results.len() == ops.len()`.
    pub results: Vec<Option<ValueId>>,
    /// Name hints for values (for debugging and MSL variables).
    pub names: BTreeMap<ValueId, String>,
}

impl Block {
    pub fn new(id: BlockId) -> Self {
        Block { id, ops: Vec::new(), results: Vec::new(), names: BTreeMap::new() }
    }

    /// Push an op that produces a value.
    pub fn push_op(&mut self, op: Op, value_id: ValueId) {
        self.ops.push(op);
        self.results.push(Some(value_id));
    }

    /// Push an op that does not produce a value (Store, Loop, Barrier, etc.).
    pub fn push_op_no_result(&mut self, op: Op) {
        self.ops.push(op);
        self.results.push(None);
    }

    /// Give a name hint to a value for prettier MSL output.
    pub fn name_value(&mut self, id: ValueId, name: impl Into<String>) {
        self.names.insert(id, name.into());
    }
}

/// A complete kernel in the IR.
#[derive(Debug, PartialEq)]
pub struct Kernel {
    /// Kernel name.
    pub name: String,
    /// Thread-indexing mode — controls which Metal built-in attributes are emitted.
    pub mode: KernelMode,
    /// Input/output parameters (tensors).
    pub params: Vec<Param>,
    /// Constexpr declarations.
    pub constexprs: Vec<ConstExprDecl>,
    /// Entry block of the kernel body.
    pub body: Block,
    /// All blocks in this kernel (including nested loop bodies, etc.).
    pub blocks: FxHashMap<BlockId, Block>,
    /// Return shapes — for each output tensor, the shape of the written region.
    pub return_shapes: Vec<Shape>,
    /// Tile schedule annotations set by SchedulePass.
    /// Keys are ValueId of Dot ops; values are (tile_m, tile_n, tile_k).
    pub tile_annotations: FxHashMap<ValueId, (u32, u32, u32)>,
    /// Per-kernel opt-in for the MFA-style f32→bf16 reinterpret cast.
    /// Overrides `MslConfig::bfloat_reinterpret_cast` when set true:
    /// the codegen emits `as_type<bfloat2>(fp32)[1]` (truncation, fast)
    /// instead of `bfloat(fp32)` (round-to-nearest, IEEE-correct). Off
    /// by default — kernels that can prove the ≤1 ULP truncation drift
    /// is acceptable for their numeric profile (heavy-tailed attention
    /// mass, accumulated dot products with limited final-cast count)
    /// opt in via the kernel module's wrapper. Currently used by the
    /// SDPA-prefill MMA family on M2 where it buys ~2pts bf16.
    pub bfloat_reinterpret_cast: bool,
    /// Per-kernel opt-in for the indirect-dispatch Swift wrapper variant.
    /// When `true`, `render_swift_wrappers` emits a `<name>_indirect`
    /// alongside the regular wrapper that takes an `MTLBuffer` carrying
    /// `MTLDispatchThreadgroupsIndirectArguments` instead of an `MTLSize`
    /// grid. Used by FFAI's GPU-router work to chain successive MoE-layer
    /// dispatches without per-layer host stalls. Replaces the previous
    /// hardcoded kernel-name allowlist in `metaltile-codegen::emit` —
    /// kernels now declare their own indirect-dispatch eligibility via
    /// the DSL / IR rather than the codegen having a special-case match
    /// on `name`.
    pub wants_indirect_variant: bool,
}

impl Kernel {
    pub fn new(name: impl Into<String>) -> Self {
        Kernel {
            name: name.into(),
            mode: KernelMode::default(),
            params: Vec::new(),
            constexprs: Vec::new(),
            body: Block::new(BlockId::new(0)),
            // `kernel.blocks` holds NESTED blocks only — loop bodies,
            // if-then/else branches, etc.  The entry block lives at
            // `kernel.body` and is the canonical source of truth for
            // block 0.  Earlier versions also inserted a clone of the
            // entry block here under `BlockId(0)`; that copy was never
            // updated by passes mutating `kernel.body`, and any code
            // walking `kernel.blocks.values()` for analysis ended up
            // reading stale state.  Lookups now go through
            // `kernel.body` for the entry block and `kernel.blocks`
            // for everything else — see [`iter_blocks`] for a unified
            // walk.
            blocks: FxHashMap::default(),
            return_shapes: Vec::new(),
            tile_annotations: FxHashMap::default(),
            bfloat_reinterpret_cast: false,
            wants_indirect_variant: false,
        }
    }

    /// Add a block to the kernel, returning its ID.
    ///
    /// Panics if `block.id == self.body.id` — the entry block lives in
    /// `self.body`, not `self.blocks`.
    pub fn add_block(&mut self, block: Block) -> BlockId {
        debug_assert_ne!(
            block.id, self.body.id,
            "entry block lives in kernel.body, not kernel.blocks"
        );
        let id = block.id;
        self.blocks.insert(id, block);
        id
    }

    /// Iterate every block in the kernel — entry block first, then
    /// nested blocks in `kernel.blocks` insertion order.  Use this when
    /// an analysis needs to walk all SSA defs / uses across the kernel
    /// (liveness, use-counting, identifier scanning).  Walking only
    /// `self.blocks.values()` would skip the entry block; walking only
    /// `self.body` would skip nested loop/if bodies.
    pub fn iter_blocks(&self) -> impl Iterator<Item = &Block> {
        std::iter::once(&self.body).chain(self.blocks.values())
    }

    /// Mutable variant of [`iter_blocks`].  Order matches [`iter_blocks`]:
    /// entry block first, then nested blocks.
    pub fn iter_blocks_mut(&mut self) -> impl Iterator<Item = &mut Block> {
        std::iter::once(&mut self.body).chain(self.blocks.values_mut())
    }

    /// Get a block by ID.
    pub fn get_block(&self, id: BlockId) -> Option<&Block> {
        if id == self.body.id {
            return Some(&self.body);
        }
        self.blocks.get(&id)
    }

    /// Get a mutable block by ID.
    pub fn get_block_mut(&mut self, id: BlockId) -> Option<&mut Block> {
        if id == self.body.id {
            return Some(&mut self.body);
        }
        self.blocks.get_mut(&id)
    }
}

impl Clone for Kernel {
    fn clone(&self) -> Self {
        // `kernel.blocks` no longer holds the entry block — see the
        // `Kernel::new` comment for the data-structure rationale.  A
        // plain field-by-field clone is correct now; previously this
        // re-inserted the entry block under `body.id` to paper over
        // the stale-copy invariant.
        Kernel {
            name: self.name.clone(),
            mode: self.mode,
            params: self.params.clone(),
            constexprs: self.constexprs.clone(),
            body: self.body.clone(),
            blocks: self.blocks.clone(),
            return_shapes: self.return_shapes.clone(),
            tile_annotations: self.tile_annotations.clone(),
            bfloat_reinterpret_cast: self.bfloat_reinterpret_cast,
            wants_indirect_variant: self.wants_indirect_variant,
        }
    }
}

// ── Display / pretty-printing ────────────────────────────────────────────────

impl std::fmt::Display for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Header
        let mode_str = match self.mode {
            KernelMode::Elementwise => "Elementwise",
            KernelMode::Reduction => "Reduction",
            KernelMode::Grid3D => "Grid3D",
            KernelMode::Tile2D => "Tile2D",
            KernelMode::SimdGroup2D => "SimdGroup2D",
        };
        let params_str: Vec<String> = self
            .params
            .iter()
            .map(|p| {
                let io = if p.is_output { "out:" } else { "" };
                format!("{io}{}:{:?}", p.name, p.dtype)
            })
            .collect();
        writeln!(f, "kernel {}  mode={mode_str}  params=[{}]", self.name, params_str.join(", "))?;

        // Entry block
        write!(f, "{}", self.body)?;

        // Nested blocks (sorted by ID)
        let mut block_ids: Vec<BlockId> = self.blocks.keys().copied().collect();
        block_ids.sort_unstable();
        for id in block_ids {
            if id == self.body.id {
                continue;
            }
            if let Some(block) = self.blocks.get(&id) {
                write!(f, "{}", block)?;
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  block b{}:", self.id.as_u32())?;
        for (i, op) in self.ops.iter().enumerate() {
            let result_id = self.results.get(i).and_then(|r| *r);
            if let Some(vid) = result_id {
                write!(f, "    v{:<4} = ", vid.as_u32())?;
            } else {
                write!(f, "         ")?;
            }
            op.fmt_ir(f)?;
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Op {
    // -----------------------------------------------------------------------
    // Typed accessors — extract fields from specific variants without a match
    // -----------------------------------------------------------------------

    /// Returns the constant integer value if this is `Op::Const`.
    pub fn as_const(&self) -> Option<i64> {
        if let Op::Const { value } = self { Some(*value) } else { None }
    }

    /// Returns a mutable reference to the constant value if this is `Op::Const`.
    pub fn as_const_mut(&mut self) -> Option<&mut i64> {
        if let Op::Const { value } = self { Some(value) } else { None }
    }

    /// Returns `(var, start, end, step, body)` if this is `Op::Loop`.
    pub fn as_loop(&self) -> Option<(VarId, ValueId, ValueId, ValueId, BlockId)> {
        if let Op::Loop { var, start, end, step, body } = self {
            Some((*var, *start, *end, *step, *body))
        } else {
            None
        }
    }

    /// Returns `(cond, then_block, else_block)` if this is `Op::If`.
    pub fn as_if(&self) -> Option<(ValueId, BlockId, Option<BlockId>)> {
        if let Op::If { cond, then_block, else_block } = self {
            Some((*cond, *then_block, *else_block))
        } else {
            None
        }
    }

    /// Returns the axis if this is `Op::ProgramId`.
    pub fn program_id_axis(&self) -> Option<u32> {
        if let Op::ProgramId { axis } = self { Some(*axis) } else { None }
    }

    /// Returns the destination buffer name for any store op
    /// (`Store`, `VectorStore`, `StrideStore`, `ThreadgroupStore`).
    pub fn store_dst(&self) -> Option<&str> {
        match self {
            Op::Store { dst, .. } | Op::VectorStore { dst, .. } | Op::StrideStore { dst, .. } =>
                Some(dst),
            Op::ThreadgroupStore { name, .. } => Some(name),
            _ => None,
        }
    }

    /// Returns the source buffer name for any load op
    /// (`Load`, `VectorLoad`, `ThreadgroupLoad`).
    pub fn load_src(&self) -> Option<&str> {
        match self {
            Op::Load { src, .. } | Op::VectorLoad { src, .. } => Some(src),
            Op::ThreadgroupLoad { name, .. } => Some(name),
            _ => None,
        }
    }

    /// Returns the load indices slice if this is `Op::Load`; empty slice otherwise.
    pub fn load_indices(&self) -> &[IndexExpr] {
        if let Op::Load { indices, .. } = self { indices } else { &[] }
    }

    /// True if this `Store` carries a predicate mask (may-write semantics).
    pub fn has_store_mask(&self) -> bool { matches!(self, Op::Store { mask: Some(_), .. }) }

    /// Returns the sub-ops if this is `Op::FusedElementwise`.
    pub fn fused_ops(&self) -> Option<&[Op]> {
        if let Op::FusedElementwise { ops } = self { Some(ops) } else { None }
    }

    /// Returns the sub-ops mutably if this is `Op::FusedElementwise`.
    pub fn fused_ops_mut(&mut self) -> Option<&mut Vec<Op>> {
        if let Op::FusedElementwise { ops } = self { Some(ops) } else { None }
    }

    /// Returns `(name, &mut size)` if this is `Op::ThreadgroupAlloc`.
    pub fn as_threadgroup_alloc_mut(&mut self) -> Option<(&str, &mut u32)> {
        if let Op::ThreadgroupAlloc { name, size, .. } = self {
            Some((name.as_str(), size))
        } else {
            None
        }
    }

    // variant_name() → auto-generated by #[derive(VariantName)]

    // -----------------------------------------------------------------------
    // Display impl
    // -----------------------------------------------------------------------

    /// Write a compact IR representation of this op.
    fn fmt_ir(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::ProgramId { axis } => write!(f, "ProgramId(axis={axis})"),
            Op::Const { value } => write!(f, "Const({value})"),
            Op::Arange { start, step, len } => {
                let s = start.map_or("0.0".into(), |v| format!("{v}"));
                let st = step.map_or("1.0".into(), |v| format!("{v}"));
                write!(f, "Arange(start={s}, step={st}, len={len:?})")
            },
            Op::Load { src, indices, mask, other } => {
                let idx_str: Vec<String> = indices.iter().map(fmt_index).collect();
                write!(f, "Load({src}, [{}]", idx_str.join(", "))?;
                if let Some(m) = mask {
                    write!(f, ", mask=v{}", m.as_u32())?;
                }
                if let Some(o) = other {
                    write!(f, ", other={o}")?;
                }
                write!(f, ")")
            },
            Op::Store { dst, indices, value, mask } => {
                let idx_str: Vec<String> = indices.iter().map(fmt_index).collect();
                write!(f, "Store({dst}, v{}, [{}]", value.as_u32(), idx_str.join(", "))?;
                if let Some(m) = mask {
                    write!(f, ", mask=v{}", m.as_u32())?;
                }
                write!(f, ")")
            },
            Op::BinOp { op, lhs, rhs } => {
                write!(f, "BinOp({op:?}, v{}, v{})", lhs.as_u32(), rhs.as_u32())
            },
            Op::Fma { a, b, c } => {
                write!(f, "Fma(v{}, v{}, v{})", a.as_u32(), b.as_u32(), c.as_u32())
            },
            Op::Dot { a, b } => write!(f, "Dot(v{}, v{})", a.as_u32(), b.as_u32()),
            Op::Reduce { value, axis, op } => {
                write!(f, "Reduce({op:?}, axis={axis}, v{})", value.as_u32())
            },
            Op::StrideReduce {
                src,
                offset,
                stride,
                end,
                op,
                dtype,
                transform,
                secondary_src,
                secondary_base,
            } => {
                write!(
                    f,
                    "StrideReduce({src}, offset=v{}, stride=v{}, end=v{}, op={op:?}, dtype={dtype:?}",
                    offset.as_u32(),
                    stride.as_u32(),
                    end.as_u32()
                )?;
                if let Some(t) = transform {
                    write!(f, ", transform=[{} ops]", t.len())?;
                }
                if secondary_src.is_some() {
                    write!(f, ", secondary")?;
                }
                if let Some(sb) = secondary_base {
                    write!(f, ", secondary_base=v{}", sb.as_u32())?;
                }
                write!(f, ")")
            },
            Op::Cast { value, dtype } => write!(f, "Cast(v{}, {dtype:?})", value.as_u32()),
            Op::Loop { var, start, end, step, body } => {
                write!(
                    f,
                    "Loop(var{}, v{}..v{}, step=v{}, body=b{})",
                    var.as_u32(),
                    start.as_u32(),
                    end.as_u32(),
                    step.as_u32(),
                    body.as_u32()
                )
            },
            Op::If { cond, then_block, else_block } => {
                write!(f, "If(v{}, b{}", cond.as_u32(), then_block.as_u32())?;
                if let Some(eb) = else_block {
                    write!(f, ", b{}", eb.as_u32())?;
                }
                write!(f, ")")
            },
            Op::Zeros { dtype, shape } => write!(f, "Zeros({dtype:?}, {shape:?})"),
            Op::Transpose { value } => write!(f, "Transpose(v{})", value.as_u32()),
            Op::ExpandDims { value, axis } => {
                write!(f, "ExpandDims(v{}, axis={axis})", value.as_u32())
            },
            Op::Reshape { value, shape } => write!(f, "Reshape(v{}, {shape:?})", value.as_u32()),
            Op::Cat { values, axis } => {
                let vals: Vec<String> = values.iter().map(|v| format!("v{}", v.as_u32())).collect();
                write!(f, "Cat([{}], axis={axis})", vals.join(", "))
            },
            Op::Slice { value, ranges } => {
                let r: Vec<String> =
                    ranges.iter().map(|(a, s, l)| format!("dim{a}[{s}..{s}+{l}]")).collect();
                write!(f, "Slice(v{}, [{}])", value.as_u32(), r.join(", "))
            },
            Op::InlineMsl { source, inputs, outputs } => {
                write!(
                    f,
                    "InlineMsl(\"{}\", inputs=[{}], outputs={})",
                    source.chars().take(40).collect::<String>(),
                    inputs
                        .iter()
                        .map(|v| format!("v{}", v.as_u32()))
                        .collect::<Vec<_>>()
                        .join(", "),
                    outputs.len()
                )
            },
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
                ..
            } => {
                write!(
                    f,
                    "CoopTileSetup({name}, {m}×{n}×{k}, ta={ta} tb={tb} tc={tc}, acc={acc_mode:?}, scope={exec_scope:?}, T={act_dtype:?}, acc={acc_dtype:?}, direct={direct_inputs})"
                )
            },
            Op::CoopTileZero { name } => write!(f, "CoopTileZero({name})"),
            Op::CoopTileLoadA { name, ptr_name, ei, eo, direct, .. } => {
                write!(f, "CoopTileLoadA({name}, {ptr_name}, extents<{ei},{eo}>, direct={direct})")
            },
            Op::CoopTileLoadB { name, ptr_name, ei, eo, direct, .. } => {
                write!(f, "CoopTileLoadB({name}, {ptr_name}, extents<{ei},{eo}>, direct={direct})")
            },
            Op::CoopTileRun { name, direct } => write!(f, "CoopTileRun({name}, direct={direct})"),
            Op::CoopTileStoreC { name, ptr_name, ei, eo, .. } => {
                write!(f, "CoopTileStoreC({name}, {ptr_name}, extents<{ei},{eo}>)")
            },
            Op::UnaryOp { op, value } => write!(f, "UnaryOp({op:?}, v{})", value.as_u32()),
            Op::Activation { kind, value } => {
                write!(f, "Activation({kind:?}, v{})", value.as_u32())
            },
            Op::Select { cond, on_true, on_false } => {
                write!(
                    f,
                    "Select(v{}, v{}, v{})",
                    cond.as_u32(),
                    on_true.as_u32(),
                    on_false.as_u32()
                )
            },
            Op::Broadcast { value, shape } => {
                write!(f, "Broadcast(v{}, {shape:?})", value.as_u32())
            },
            Op::Splat { value, dtype, shape } => write!(f, "Splat({value}, {dtype:?}, {shape:?})"),
            Op::FusedElementwise { ops } => {
                write!(f, "FusedElementwise([")?;
                for (i, op) in ops.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    op.fmt_ir(f)?;
                }
                write!(f, "])")
            },
            Op::VectorLoad { src, byte_offset, len } => {
                write!(f, "VectorLoad({src}, offset=v{}, len={len})", byte_offset.as_u32())
            },
            Op::VectorStore { dst, byte_offset, len, value } => {
                write!(
                    f,
                    "VectorStore({dst}, offset=v{}, len={len}, v{})",
                    byte_offset.as_u32(),
                    value.as_u32()
                )
            },
            Op::VectorExtract { vec, lane } => {
                write!(f, "VectorExtract(v{}, lane={lane})", vec.as_u32())
            },
            Op::Pack { elements, .. } => {
                let ids: Vec<String> =
                    elements.iter().map(|v| format!("v{}", v.as_u32())).collect();
                write!(f, "Pack({})", ids.join(", "))
            },
            Op::Gather { src, indices, axis } => {
                write!(f, "Gather({src}, v{}, axis={axis})", indices.as_u32())
            },
            Op::Scatter { dst, indices, value, axis } => {
                write!(f, "Scatter({dst}, v{}, v{}, axis={axis})", indices.as_u32(), value.as_u32())
            },
            Op::Atomic { op, scope, dst, index, value } => {
                write!(
                    f,
                    "Atomic({op:?}, scope={scope:?}, {dst}, v{}, v{})",
                    index.as_u32(),
                    value.as_u32()
                )
            },
            Op::Scan { value, axis, op, exclusive } => {
                write!(
                    f,
                    "Scan(v{}, axis={axis}, op={op:?}, exclusive={exclusive})",
                    value.as_u32()
                )
            },
            Op::StrideScan { src, dst, offset, end, op } => {
                write!(
                    f,
                    "StrideScan({src}->{dst}, v{}..v{}, op={op:?})",
                    offset.as_u32(),
                    end.as_u32()
                )
            },
            Op::KernelCall { callee, args, dtype } => {
                let args_str = args
                    .iter()
                    .map(|a| match a {
                        KernelCallArg::Value(v) => format!("v{}", v.as_u32()),
                        KernelCallArg::Tensor(s) => format!("\"{}\"", s),
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "KernelCall(\"{callee}\", args=[{args_str}], dtype={dtype:?})")
            },
            Op::StrideArgReduce { src, offset, end, op } => {
                write!(
                    f,
                    "StrideArgReduce({src}, v{}..v{}, op={op:?})",
                    offset.as_u32(),
                    end.as_u32()
                )
            },
            Op::StrideStore { src, dst, offset, end, scalar, aux_src } => {
                write!(
                    f,
                    "StrideStore({src}->{dst}, v{}..v{}, scalar=v{}",
                    offset.as_u32(),
                    end.as_u32(),
                    scalar.as_u32()
                )?;
                if let Some(aux) = aux_src {
                    write!(f, ", aux_src={aux}")?;
                }
                write!(f, ")")
            },
            Op::SimdReduce { value, op } => write!(f, "SimdReduce(v{}, {op:?})", value.as_u32()),
            Op::SimdShuffleXor { value, mask } => {
                write!(f, "SimdShuffleXor(v{}, mask={mask})", value.as_u32())
            },
            Op::SimdBroadcast { value, lane } => {
                write!(f, "SimdBroadcast(v{}, lane=v{})", value.as_u32(), lane.as_u32())
            },
            Op::StackAlloc { dtype, size, name } => {
                write!(f, "StackAlloc({dtype:?}, {size}, {name})")
            },
            Op::StackLoad { name, index } => {
                write!(f, "StackLoad({name}, v{})", index.as_u32())
            },
            Op::StackStore { name, index, value } => {
                write!(f, "StackStore({name}, v{}, v{})", index.as_u32(), value.as_u32())
            },
            Op::ThreadgroupAlloc { dtype, size, name } => {
                write!(f, "ThreadgroupAlloc({dtype:?}, {size}, {name})")
            },
            Op::ThreadgroupLoad { name, index } => {
                write!(f, "ThreadgroupLoad({name}, v{})", index.as_u32())
            },
            Op::ThreadgroupStore { name, index, value } => {
                write!(f, "ThreadgroupStore({name}, v{}, v{})", index.as_u32(), value.as_u32())
            },
            Op::Barrier => write!(f, "Barrier"),
            Op::SimdgroupBarrier => write!(f, "SimdgroupBarrier"),
            Op::DeclareLocal { name, value } => {
                write!(f, "DeclareLocal({name}, v{})", value.as_u32())
            },
            Op::SetLocal { name, value } => {
                write!(f, "SetLocal({name}, v{})", value.as_u32())
            },
            Op::ArgReduce { value, axis, op } => {
                write!(f, "ArgReduce(v{}, axis={axis}, {op:?})", value.as_u32())
            },
            Op::SimdgroupAlloc { dtype, m, n } => {
                write!(f, "SimdgroupAlloc({dtype:?}, {m}×{n})")
            },
            Op::SimdgroupElemLoad { value, index } => {
                write!(f, "SimdgroupElemLoad(v{}, [{index}])", value.as_u32())
            },
            Op::SimdgroupElemStore { value, index, data } => {
                write!(f, "SimdgroupElemStore(v{}, [{index}], v{})", value.as_u32(), data.as_u32())
            },
            Op::SimdgroupLoad { dest, tg, offset, stride, transpose } => {
                write!(
                    f,
                    "SimdgroupLoad(v{}, {tg}, off=v{}, stride={stride}, transpose={transpose})",
                    dest.as_u32(),
                    offset.as_u32()
                )
            },
            Op::SimdgroupMatMul { a, b, c } => {
                write!(f, "SimdgroupMatMul(v{}, v{}, v{})", a.as_u32(), b.as_u32(), c.as_u32())
            },
            Op::SimdLaneId => write!(f, "SimdLaneId"),
            Op::SimdGroupId => write!(f, "SimdGroupId"),
            Op::SimdScan { value, op, exclusive } => {
                write!(f, "SimdScan(v{}, {op:?}, exclusive={exclusive})", value.as_u32())
            },
        }
    }
}

/// Format an index expression for IR display.
fn fmt_index(idx: &IndexExpr) -> String {
    match idx {
        IndexExpr::Value(v) => format!("v{}", v.as_u32()),
        IndexExpr::Const(n) => format!("{n}"),
        IndexExpr::Range(v, offset) => format!("v{}..v{}+{offset}", v.as_u32(), v.as_u32()),
    }
}

#[cfg(test)]
mod tests {
    use super::{Block, BlockId, Kernel, Op, ValueId};

    #[test]
    fn entry_block_lives_in_body_not_blocks() {
        // The entry block is `kernel.body`; `kernel.blocks` only holds
        // nested blocks (loop bodies, if-then/else).  Earlier versions
        // also kept a never-updated clone of the entry block in
        // `kernel.blocks[body.id]`; that was the source of the
        // "stale block" footgun documented in #209/2.
        let mut kernel = Kernel::new("body");
        kernel.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        assert_eq!(kernel.body.ops.len(), 1);
        assert!(
            !kernel.blocks.contains_key(&kernel.body.id),
            "entry block must NOT be present in kernel.blocks"
        );
    }

    #[test]
    fn getters_treat_body_as_authoritative_entry_block() {
        // `get_block(body.id)` returns `&kernel.body` directly; pushing
        // to it is observed both via the field and via the getter (no
        // stale copy to refresh).
        let mut kernel = Kernel::new("body");
        kernel.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        assert_eq!(kernel.get_block(BlockId::new(0)).expect("entry block must exist").ops.len(), 1);

        let body = kernel.get_block_mut(BlockId::new(0)).expect("entry block must be mutable");
        body.push_op(Op::Const { value: 2 }, ValueId::new(1));
        assert_eq!(kernel.body.ops.len(), 2);
        assert_eq!(
            kernel.get_block(BlockId::new(0)).expect("entry block must exist").ops.len(),
            2,
            "get_block(body.id) must reflect post-mutation state"
        );
    }

    #[test]
    fn iter_blocks_yields_body_then_nested() {
        // `iter_blocks` is the canonical full-kernel walk: entry block
        // first, then nested blocks.  Replaces the
        // `body + kernel.blocks.values()` pattern that callers used to
        // open-code, where it was easy to forget the body half.
        let mut kernel = Kernel::new("iter");
        kernel.body.push_op(Op::Const { value: 10 }, ValueId::new(0));
        let mut nested = Block::new(BlockId::new(1));
        nested.push_op(Op::Const { value: 20 }, ValueId::new(1));
        kernel.add_block(nested);

        let ids: Vec<u32> = kernel.iter_blocks().map(|b| b.id.as_u32()).collect();
        assert_eq!(ids, vec![0, 1], "iter_blocks yields body first then nested");
    }

    #[test]
    fn display_format_shows_kernel_structure() {
        use super::{BinOpKind, IndexExpr, KernelMode, Param, ParamKind};
        use crate::{dtype::DType, shape::Shape};

        let mut k = Kernel::new("mt_vadd");
        k.mode = KernelMode::Elementwise;
        k.params.push(Param {
            name: "a".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: false,
            kind: ParamKind::Tensor,
        });
        k.params.push(Param {
            name: "b".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: false,
            kind: ParamKind::Tensor,
        });
        k.params.push(Param {
            name: "out".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: true,
            kind: ParamKind::Tensor,
        });
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(
            Op::Load {
                src: "a".into(),
                indices: vec![IndexExpr::Value(ValueId::new(0))],
                mask: None,
                other: None,
            },
            ValueId::new(1),
        );
        k.body.push_op(
            Op::Load {
                src: "b".into(),
                indices: vec![IndexExpr::Value(ValueId::new(0))],
                mask: None,
                other: None,
            },
            ValueId::new(2),
        );
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(1), rhs: ValueId::new(2) },
            ValueId::new(3),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(3),
            mask: None,
        });

        let output = format!("{k}");
        assert!(output.contains("kernel mt_vadd"), "should show kernel name: {output}");
        assert!(output.contains("mode=Elementwise"), "should show mode: {output}");
        assert!(output.contains("v0    = ProgramId(axis=0)"), "should show ProgramId: {output}");
        assert!(output.contains("BinOp(Add, v1, v2)"), "should show BinOp: {output}");
        assert!(output.contains("Store(out, v3, [v0])"), "should show Store: {output}");
    }
}
