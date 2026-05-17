//! `BenchSpec` — declarative kernel benchmark descriptors.
//!
//! Each `#[bench_kernel(...)]` annotation on a `#[kernel]` fn generates one
//! `BenchSpec` and registers it via `inventory::submit!`. The bench runner
//! iterates `inventory::iter::<BenchSpec>`, sorts by `(op, subop)`, then calls
//! `run_spec(spec, runner, dt)` per dtype (run_spec lives in metaltile-cli).
//!
//! For the 10 "simple" class types (Unary, Binary, AllReduce, RowReduce,
//! Arange, BinaryTwo, Select, RowNorm, MatVec, MatVecMasked), the macro
//! generates a `ShapeSpec` and sets `dispatch = BenchDispatch::Generic`.
//! The generic runner handles all of these uniformly.
//!
//! The 9 "complex" class types (Sort, Scan, ArgReduce, Random, FpQuantized,
//! QuantizedMatVec, Rope, Attention, StridedCopy) keep specialized runners.

use metaltile_core::{
    dtype::DType,
    ir::{Kernel, KernelMode},
};

// ── Default sizes ───────────────────────────────────────────────────────

pub const ELEMENTWISE_N_BENCH: usize = 64 * 1024 * 1024;
pub const ELEMENTWISE_N_CHECK: usize = 2_048;
pub const ELEMENTWISE_TPG: usize = 256;

pub const BINARY_TPG: usize = 1_024;
pub const BINARY_N_PER_THREAD: usize = 2;

pub const ALL_REDUCE_N: usize = 64 * 1024 * 1024;
pub const ALL_REDUCE_N_CHECK: usize = 16_384;
pub const ALL_REDUCE_TPG: usize = 256;

pub const ROW_REDUCE_SHAPES: &[(usize, usize)] = &[(1_024, 4_096)];
pub const ROW_REDUCE_CHECK_B: usize = 8;
pub const ROW_REDUCE_CHECK_N: usize = 512;
pub const ROW_REDUCE_TPG: usize = 256;

pub const ARANGE_N: usize = 64 * 1024 * 1024;
pub const ARANGE_N_CHECK: usize = 4_096;
pub const ARANGE_TPG: usize = 1_024;

pub const BINARY_TWO_TPG: usize = 1_024;

pub const SELECT_TPG: usize = 256;

// ── Single-dtype shorthands ──────────────────────────────────────────────

pub const F32_ONLY: &[DType] = &[DType::F32];
pub const F16_ONLY: &[DType] = &[DType::F16];

// ── Dim: runtime size expression ─────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum Dim {
    N,   // n
    B,   // b
    BxN, // b * n
    One, // 1
}

impl Dim {
    pub fn resolve(self, n: usize, b: usize) -> usize {
        match self {
            Dim::N => n,
            Dim::B => b,
            Dim::BxN => b * n,
            Dim::One => 1,
        }
    }
}

// ── BufInit ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum BufInit {
    Zeros,
    Half,
    Signed,
    Positive,
    Unit,
    Fill(f32),
    AltZeroOne,
}

impl BufInit {
    pub fn generate(self, n: usize) -> Vec<f32> {
        match self {
            BufInit::Zeros => vec![0.0; n],
            BufInit::Half => vec![0.5; n],
            BufInit::Signed => (0..n)
                .map(|i| match i % 8 {
                    0 => -3.0,
                    1 => -1.5,
                    2 => -0.5,
                    3 => 0.0,
                    4 => 0.25,
                    5 => 0.75,
                    6 => 1.5,
                    _ => 3.0,
                })
                .collect(),
            BufInit::Positive => (0..n).map(|i| 0.25 + (i % 16) as f32 * 0.25).collect(),
            BufInit::Unit => vec![1.0; n],
            BufInit::Fill(v) => vec![v; n],
            BufInit::AltZeroOne => (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect(),
        }
    }
}

// ── TensorBufSpec ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct TensorBufSpec {
    pub count: Dim,
    pub init: BufInit,
    pub dtype_override: Option<DType>,
}

// ── ScalarBufSpec ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum ScalarBufSpec {
    U32N,
    U32B,
    U64N,
    U64B,
    I64B,
}

// ── DispatchGrid ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum DispatchGrid {
    DivCeilN,
    DivCeilN2,
    RowsB,
    RowsBY,
    Single,
}

impl DispatchGrid {
    pub fn eval(self, n: usize, b: usize, tpg: usize) -> [usize; 3] {
        match self {
            DispatchGrid::DivCeilN => [n.div_ceil(tpg), 1, 1],
            DispatchGrid::DivCeilN2 => [n.div_ceil(tpg * 2), 1, 1],
            DispatchGrid::RowsB => [b, 1, 1],
            DispatchGrid::RowsBY => [1, b, 1],
            DispatchGrid::Single => [1, 1, 1],
        }
    }
}

// ── MlxArg ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub enum MlxArg {
    TensorBuf(usize),
    FreshOut(usize),
    U32N,
    U64N,
    U64B,
    I64B,
    Zeros8,
    BoolAltN,
    U32V(u32),
}

// ── ShapeSpec ────────────────────────────────────────────────────────────

pub struct ShapeSpec {
    pub label: &'static str,
    pub n: usize,
    pub b: usize,
    pub check_n: usize,
    pub check_b: usize,
    pub mode: KernelMode,
    pub tpg: usize,
    pub grid: DispatchGrid,
    pub tensor_bufs: &'static [TensorBufSpec],
    pub scalar_bufs: &'static [ScalarBufSpec],
    pub cexprs: &'static [(&'static str, Dim)],
    pub out_elems: Dim,
    pub reads: usize,
    pub bytes_fn: fn(usize, usize, usize, usize, usize) -> usize,
    pub mlx_args: Option<&'static [MlxArg]>,
    pub mlx_grid: Option<DispatchGrid>,
    pub mlx_tpg: usize,
}

// ── BenchDispatch ────────────────────────────────────────────────────────

pub enum BenchDispatch {
    Generic,
    Sort { b: usize, n: usize, tpg: usize },
    Scan { shapes: &'static [(usize, usize)], tpg: usize },
    ArgReduce { n: usize, check_n: usize, tpg: usize },
    Random { n: usize, tpg: usize },
    FpQuantized { n: usize, tpg: usize },
    QuantizedMatVec { shapes: &'static [(usize, usize)], group_size: usize, tpg: usize },
    Rope { b: usize, h: usize, l: usize, d: usize, n_per_group: usize },
    Attention { shapes: &'static [(usize, usize, usize)], tpg: usize },
    StridedCopy { m: usize, n: usize, pad: usize },
}

// ── BenchSpec ───────────────────────────────────────────────────────────

pub struct BenchSpec {
    pub op: &'static str,
    pub subop: &'static str,
    pub kernel_name: &'static str,
    pub kernel_ir: fn(DType) -> Kernel,
    pub dtypes: &'static [DType],
    pub tol: f32,
    pub mlx_src: Option<&'static str>,
    pub mlx_pattern: Option<&'static str>,
    pub shapes: &'static [ShapeSpec],
    pub dispatch: BenchDispatch,
    /// Optional explicit kernel mode override. When `None`, downstream
    /// tooling (e.g. `tile build`) infers the mode from `dispatch`/
    /// `shapes` via `first_mode(spec)`. Used by codegen-only kernels
    /// (empty `shapes`, `dispatch: Generic`) that need a non-default
    /// mode — e.g. Reduction-mode dequant GEMV kernels that rely on
    /// `lsize`/`tid` aliases the Elementwise mode doesn't provide.
    pub kernel_mode: Option<KernelMode>,
}

inventory::collect!(BenchSpec);

// ── Standard bytes formulas ──────────────────────────────────────────────

pub fn bytes_elementwise(n: usize, _b: usize, reads: usize, _out: usize, eb: usize) -> usize {
    n * eb * (reads + 1)
}

pub fn bytes_row_op(n: usize, b: usize, reads: usize, out: usize, eb: usize) -> usize {
    b * n * eb * reads + out * eb
}

pub fn bytes_mat_vec(n: usize, b: usize, _reads: usize, out: usize, eb: usize) -> usize {
    (b * n + n + out) * eb
}

pub fn bytes_mat_vec_masked(n: usize, b: usize, _reads: usize, out: usize, eb: usize) -> usize {
    (b * n + 2 * n + out) * eb
}
