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

use std::collections::BTreeMap;

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
            BinOpKind::Shl => "<<",
            BinOpKind::Shr => ">>",
            BinOpKind::BitAnd => "&",
            BinOpKind::BitOr => "|",
            BinOpKind::BitXor => "^",
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
        matches!(self, BinOpKind::Max | BinOpKind::Min | BinOpKind::Pow)
    }
}

/// Reduction kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceKind {
    Sum,
    Max,
    Min,
    Mean,
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

/// Attention parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct AttnParams {
    pub scale: Option<f32>,
    pub is_causal: bool,
    pub dropout_p: f32,
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

/// A single operation in the IR.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// `program_id(axis)` — which block this threadgroup handles along an axis.
    ProgramId { axis: u32 },

    /// A constant integer value (from a literal in the DSL).
    Const { value: i64 },

    /// `arange(start, step, len)` — creates a 1D range [start, start+step, ...].
    /// `start` and `step` default to 0.0 and 1.0 respectively.
    Arange { start: Option<f64>, step: Option<f64>, len: ConstExpr },

    /// Load a tile from a tensor at given indices.
    Load {
        /// The parameter to load from.
        src: String,
        /// Per-dimension index expressions.
        indices: Vec<IndexExpr>,
        /// Optional mask: load only where mask is true (false → fill with `other`).
        mask: Option<ValueId>,
        /// Fill value when mask is false (default 0.0).
        other: Option<f64>,
    },

    /// Store a tile to a tensor at given indices.
    Store {
        /// The parameter to store to.
        dst: String,
        /// Per-dimension index expressions.
        indices: Vec<IndexExpr>,
        /// The value to store.
        value: ValueId,
        /// Optional mask: store only where mask is true.
        mask: Option<ValueId>,
    },

    /// Elementwise binary operation.
    BinOp { op: BinOpKind, lhs: ValueId, rhs: ValueId },

    /// Tile matrix multiply: `dot(a, b)`.
    Dot { a: ValueId, b: ValueId },

    /// Reduction along an axis.
    Reduce { value: ValueId, axis: u32, op: ReduceKind },

    /// Per-thread strided reduction over a device buffer.
    /// Reduces `src[offset]`, `src[offset+stride]`, `src[offset+2*stride]`, ... while index < `end`.
    /// If `transform` is set, the op is applied to each loaded element before accumulation.
    StrideReduce {
        src: String,
        /// First index to load (= tid for intra-row; = row*N + tid for full buffer).
        offset: ValueId,
        /// Step between successive loads (= lsize).
        stride: ValueId,
        /// Exclusive upper bound (= N for intra-row; = row*N + N for full buffer).
        end: ValueId,
        op: ReduceKind,
        dtype: DType,
        /// Optional per-element transform chain applied to the loaded value before accumulation.
        /// Each op in the chain takes the previous result as input.
        transform: Option<Vec<Op>>,
        /// For dot-product reductions (GEMV): multiply each `src[_i]` by `secondary_src[_i - secondary_base]`.
        secondary_src: Option<String>,
        /// Base offset subtracted from the loop index when accessing secondary_src.
        secondary_base: Option<ValueId>,
    },

    /// Type cast.
    Cast { value: ValueId, dtype: DType },

    /// Loop: iterate a variable from start to end with step.
    Loop { var: VarId, start: ValueId, end: ValueId, step: ValueId, body: BlockId },

    /// Conditional branch: if `cond` is true, execute `then_block`, else `else_block`.
    If { cond: ValueId, then_block: BlockId, else_block: Option<BlockId> },

    /// Create a zero-filled tile.
    Zeros {
        dtype: DType,
        /// Shape of the tile (usually a 2D tile).
        shape: Shape,
    },

    /// Transpose a 2D tile.
    Transpose { value: ValueId },

    /// Insert a size-1 dimension at `axis`. Zero-cost reshape.
    ExpandDims { value: ValueId, axis: u32 },

    /// Reshape a tile to a new shape (same element count). Zero-cost if contiguous.
    Reshape { value: ValueId, shape: Shape },

    /// Concatenate tiles along `axis`.
    Cat { values: Vec<ValueId>, axis: u32 },

    /// Extract a slice of a tile.
    Slice {
        value: ValueId,
        /// Which dimensions to slice; (axis, start_offset, length).
        ranges: Vec<(u32, i64, i64)>,
    },

    /// Inline raw MSL code. Escape hatch.
    InlineMsl { source: String, inputs: Vec<ValueId>, outputs: Vec<TypedSlot> },

    // ---- High-level ML primitives (lowered in a pass) ----
    /// Flash attention.
    FlashAttention { q: ValueId, k: ValueId, v: ValueId, params: AttnParams },

    /// Sliding window attention.
    SlidingWindowAttention { q: ValueId, k: ValueId, v: ValueId, window: u32 },

    /// RMS normalization.
    RmsNorm { x: ValueId, scale: ValueId, eps: f32 },

    /// Gated MLP block.
    GatedMlp { x: ValueId, gate_proj: ValueId, up_proj: ValueId, down_proj: ValueId },

    // ---- Scalar / element-wise math ----
    /// Unary math operation: exp, log, sqrt, rsqrt, abs, neg, ceil, floor, recip.
    UnaryOp { op: UnaryOpKind, value: ValueId },

    /// Neural activation function: silu, gelu, relu, tanh, sigmoid.
    Activation { kind: ActKind, value: ValueId },

    /// Conditional select: `cond ? on_true : on_false`.
    /// Maps to MSL `select(on_false, on_true, bool(cond))`.
    Select { cond: ValueId, on_true: ValueId, on_false: ValueId },

    /// Broadcast a scalar value to fill a tile shape (replication, no copy to device memory).
    Broadcast { value: ValueId, shape: Shape },

    /// Create a tile filled with a constant floating-point value (generalization of Zeros).
    Splat { value: f64, dtype: DType, shape: Shape },

    /// Fused chain of elementwise operations.
    /// Created by the FusionPass to merge adjacent ops like
    /// `UnaryOp(Exp) → Activation(Silu)` into a single expression.
    FusedElementwise {
        /// The elementwise ops in execution order (producer first).
        /// Each op's inputs reference either external ValueIds or
        /// the output of a preceding op in this chain (index 0..n-1).
        ops: Vec<Op>,
    },

    /// Vectorized load: loads `len` consecutive elements as a vector.
    /// `len` is 2, 4, or 8. Created by the VectorizePass from consecutive scalar Loads.
    VectorLoad {
        /// The parameter to load from.
        src: String,
        /// Flat byte offset into the buffer (already aligned).
        byte_offset: ValueId,
        /// Number of elements: 2, 4, or 8.
        len: u32,
    },

    /// Vectorized store: stores `len` consecutive elements as a vector.
    VectorStore {
        /// The parameter to store to.
        dst: String,
        /// Flat byte offset into the buffer (already aligned).
        byte_offset: ValueId,
        /// Number of elements: 2, 4, or 8.
        len: u32,
        /// The value to store (scalar or vector ValueId).
        value: ValueId,
    },

    /// Gather: indexed load from a buffer. `out[i] = src[indices[i]]`.
    Gather { src: String, indices: ValueId, axis: u32 },

    /// Scatter: indexed store to a buffer. `dst[indices[i]] = value[i]`.
    Scatter { dst: String, indices: ValueId, value: ValueId, axis: u32 },

    /// Atomic operation on device memory.
    Atomic { op: AtomicKind, dst: String, index: ValueId, value: ValueId },

    /// Prefix scan along an axis (inclusive or exclusive).
    Scan { value: ValueId, axis: u32, op: ReduceKind, exclusive: bool },

    /// Serial inclusive prefix scan over a contiguous slice of a device buffer.
    /// Writes `dst[i] = src[offset] + src[offset+1] + ... + src[i]` for i in [offset, end).
    /// Single-threaded: dispatch with [B, 1, 1] × [1, 1, 1] (one thread per row).
    StrideScan { src: String, dst: String, offset: ValueId, end: ValueId, op: ReduceKind },

    /// Serial argmax/argmin over a contiguous slice of a device buffer.
    /// Returns the flat index of the extreme element in [offset, end).
    /// Single-threaded: dispatch with [1, 1, 1] × [1, 1, 1] for a single row.
    StrideArgReduce { src: String, offset: ValueId, end: ValueId, op: ReduceKind },

    /// Strided per-element compute + store: for each element in the stride pattern,
    /// load from src, apply optional transform with a scalar operand, and store to dst.
    /// Used for write-back in reduction kernels (e.g., rout[i] = rx[i] * rms * w[i]).
    StrideStore {
        src: String,
        dst: String,
        offset: ValueId,
        end: ValueId,
        /// First operand: the scalar from the reduction step (e.g., rms, mean, 1/std).
        scalar: ValueId,
        /// Optional second operand: another device buffer (e.g., w[i] for weighted norm).
        aux_src: Option<String>,
    },

    /// Dequantize packed integer weights to floating-point tiles.
    ///
    /// Unpacks `bits`-bit integers from `weights`, scales by `scales`, and offsets
    /// by `zeros`.  Used for quantized LLM weight loading (int4/int8 GEMM).
    ///
    /// Layout: `weights[N_out, N_in/2]` (2 int4 per byte), `scales/zeros[N_out, N_in/group_size]`.
    Dequantize {
        /// Packed int4/int8 weight buffer param name.
        weights: String,
        /// FP16 scales param name.
        scales: String,
        /// FP16 zeros/bias param name.
        zeros: String,
        /// Quantization group size (e.g. 64).
        group_size: u32,
        /// Bits per weight: 4 or 8.
        bits: u8,
    },

    // ---- SIMD-group and threadgroup primitives ----
    /// SIMD-group reduction: reduce all lanes within the SIMD group.
    /// Maps to `simd_sum(v)`, `simd_max(v)`, `simd_min(v)` (Metal 2.1+).
    SimdReduce { value: ValueId, op: ReduceKind },

    /// Allocate a named threadgroup (shared) memory array.
    /// Emits `threadgroup T name[size]` in the kernel body.
    ThreadgroupAlloc {
        dtype: DType,
        /// Number of elements in the array.
        size: u32,
        /// Variable name for the threadgroup array in MSL.
        name: String,
    },

    /// Load one element from a named threadgroup array: `val = name[index]`.
    ThreadgroupLoad { name: String, index: ValueId },

    /// Store one element to a named threadgroup array: `name[index] = value`.
    ThreadgroupStore { name: String, index: ValueId, value: ValueId },

    /// Threadgroup barrier: `threadgroup_barrier(mem_flags::mem_threadgroup)`.
    /// Ensures all prior threadgroup stores are visible to all threads before
    /// any subsequent threadgroup loads.
    Barrier,

    /// Declare a mutable register-local scalar variable.
    /// Emits: `auto __ml_{name} = {init_value};`
    /// Used for loop-carried state (running prefix, best_val/best_idx, etc.).
    DeclareLocal { name: String, value: ValueId },

    /// Assign to a mutable register-local scalar variable.
    /// Emits: `__ml_{name} = {value};`
    SetLocal { name: String, value: ValueId },

    /// Return the index of the min/max element along an axis.
    ArgReduce { value: ValueId, axis: u32, op: ReduceKind },
}

// ---------------------------------------------------------------------------
// KernelMode
// ---------------------------------------------------------------------------

/// Controls which Metal built-in position attributes are emitted in the
/// kernel signature.  All built-in attributes **must** share the same vector
/// width (Metal constraint), so each mode is self-contained.
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
    pub blocks: BTreeMap<BlockId, Block>,
    /// Return shapes — for each output tensor, the shape of the written region.
    pub return_shapes: Vec<Shape>,
    /// Tile schedule annotations set by SchedulePass.
    /// Keys are ValueId of Dot ops; values are (tile_m, tile_n, tile_k).
    pub tile_annotations: BTreeMap<ValueId, (u32, u32, u32)>,
}

impl Kernel {
    pub fn new(name: impl Into<String>) -> Self {
        let body = Block::new(BlockId::new(0));
        let mut blocks = BTreeMap::new();
        blocks.insert(BlockId::new(0), body.clone());

        Kernel {
            name: name.into(),
            mode: KernelMode::default(),
            params: Vec::new(),
            constexprs: Vec::new(),
            body,
            blocks,
            return_shapes: Vec::new(),
            tile_annotations: BTreeMap::new(),
        }
    }

    /// Add a block to the kernel, returning its ID.
    pub fn add_block(&mut self, block: Block) -> BlockId {
        let id = block.id;
        self.blocks.insert(id, block);
        id
    }

    /// Synchronize the canonical entry block into the block map.
    ///
    /// The public `body` field is the source of truth for block 0. Some call
    /// sites still inspect `blocks`, so keep the entry in sync when cloning or
    /// before handing the block map to consumers.
    pub fn sync_entry_block(&mut self) { self.blocks.insert(self.body.id, self.body.clone()); }

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
        let mut blocks = self.blocks.clone();
        blocks.insert(self.body.id, self.body.clone());
        Kernel {
            name: self.name.clone(),
            mode: self.mode,
            params: self.params.clone(),
            constexprs: self.constexprs.clone(),
            body: self.body.clone(),
            blocks,
            return_shapes: self.return_shapes.clone(),
            tile_annotations: self.tile_annotations.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Block, BlockId, Kernel, Op, ValueId};

    #[test]
    fn clone_refreshes_entry_block_snapshot() {
        let mut kernel = Kernel::new("sync");
        kernel.body.push_op(Op::Const { value: 7 }, ValueId::new(0));
        let cloned = kernel.clone();
        let entry = cloned.blocks.get(&BlockId::new(0)).unwrap();
        assert_eq!(entry.ops, cloned.body.ops);
        assert_eq!(entry.results, cloned.body.results);
    }

    #[test]
    fn getters_treat_body_as_authoritative_entry_block() {
        let mut kernel = Kernel::new("body");
        kernel.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        assert_eq!(kernel.get_block(BlockId::new(0)).unwrap().ops.len(), 1);

        let body = kernel.get_block_mut(BlockId::new(0)).unwrap();
        body.push_op(Op::Const { value: 2 }, ValueId::new(1));
        assert_eq!(kernel.body.ops.len(), 2);
        assert_eq!(kernel.blocks.get(&BlockId::new(0)).unwrap().ops.len(), 0);

        kernel.sync_entry_block();
        assert_eq!(kernel.blocks.get(&BlockId::new(0)).unwrap().ops.len(), 2);
        let _ = Block::new(BlockId::new(1));
    }
}
