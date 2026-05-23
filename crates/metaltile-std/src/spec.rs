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
            BufInit::Unit =>
                (0..n).map(|i| [-0.9f32, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9][i % 7]).collect(),
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

/// Which kernel architecture backs an [`BenchDispatch::SdpaBatchedDecode`] row.
///
/// The fork is structural, not just a tuning parameter: at small K the
/// lane-quartile decode pattern wins (low register pressure, easy
/// constexpr unroll); at larger K the FA-2 simdgroup-matrix prefill tile
/// wins (the BQ×BK MMA already implements the KV-reuse pattern the spec
/// wants and is what dflash-mlx's `verify_qmm` runs).
pub enum BatchedDecodeVariant {
    /// Decode-form. Extends `sdpa_decode`'s lane-quartile pattern with
    /// `#[constexpr] batch_q` independent online-softmax streams. Used at
    /// K=2 and K=4. Threadgroup of 1024 (Reduction kernel mode).
    Decode,
    /// Prefill-tile reuse. Dispatches `mt_sdpa_prefill_mma` with
    /// `q_len = batch_q, k_len = n_kv`, causal-mask-trimmed so each Q
    /// position sees its own prefix. Used at K=8 and K=16. SimdGroup2D
    /// kernel mode; `bq`/`bk`/`wm`/`wn` mirror the SdpaPrefill tuning.
    PrefillTile { bq: usize, bk: usize, wm: usize, wn: usize },
}

pub enum BenchDispatch {
    Generic,
    Sort {
        b: usize,
        n: usize,
        tpg: usize,
    },
    Scan {
        shapes: &'static [(usize, usize)],
        tpg: usize,
    },
    ArgReduce {
        n: usize,
        check_n: usize,
        tpg: usize,
    },
    Random {
        n: usize,
        tpg: usize,
    },
    FpQuantized {
        n: usize,
        tpg: usize,
    },
    QuantizedMatVec {
        shapes: &'static [(usize, usize)],
        group_size: usize,
        tpg: usize,
        /// Bit-width of the affine-quant W codes (4 or 8). Drives the W
        /// buffer pack-factor (4 → 8 nibbles/u32, 8 → 4 bytes/u32) and
        /// the correctness oracle's dequant extract in
        /// `run_quantized_mat_vec`. Defaults to 4 in the macro to
        /// preserve the int4 contract every existing kernel relies on.
        bits: u32,
    },
    /// Quantized matmul (B>1 / prefill path). `shapes` is the same
    /// `(out_dim, in_dim)` list `QuantizedMatVec` uses; `m` is the
    /// fixed token count the kernel dispatches with (1 collapses to
    /// matvec). Compared against MLX `affine_qmm_t` with `batched=0`
    /// — see `run_quantized_mat_mul` for the reference dispatch.
    QuantizedMatMul {
        shapes: &'static [(usize, usize)],
        m: usize,
        group_size: usize,
        tpg: usize,
        /// Bit-width of the affine-quant W codes (4 or 8). Drives W
        /// buffer sizing in `run_quantized_mat_mul`.
        bits: u32,
    },
    Rope {
        b: usize,
        h: usize,
        l: usize,
        d: usize,
        n_per_group: usize,
    },
    Attention {
        shapes: &'static [(usize, usize, usize)],
        tpg: usize,
    },
    StridedCopy {
        m: usize,
        n: usize,
        pad: usize,
    },
    /// Affine dequantize one packed weight tensor into floats. Mirrors
    /// MLX `affine_dequantize<T, group_size, bits>`. One thread per pack
    /// (each pack holds `pack_factor = 32 / bits` quantized values in
    /// one uint32 for power-of-2 bits, or `bytes_per_pack` bytes for
    /// the byte-stream variants). `n_groups * batch` total groups; each
    /// group covers `group_size` output elements.
    AffineDequantize {
        bits: usize,
        group_size: usize,
        n_groups: usize,
        batch: usize,
        tpg: usize,
    },
    /// Affine quantize floats into packed weights + per-group scale +
    /// bias. Mirrors MLX `affine_quantize<T, group_size, bits>`. One
    /// threadgroup of 32 threads (one simd-group) per group; each lane
    /// handles `group_size / 32` input values, reduces min/max via
    /// `simd_min`/`simd_max`, then packs nibbles cooperatively.
    AffineQuantize {
        bits: usize,
        group_size: usize,
        n_groups: usize,
        batch: usize,
        tpg: usize,
    },
    /// Decode-form scaled dot-product attention. Mirrors MLX
    /// `sdpa_vector<T, D, V=D>`. One threadgroup per `(q_head, q_seq)`
    /// output position; `tpg` threads (one simdgroup) cooperatively
    /// reduce the dot product across `head_dim`, serial over `n_kv`
    /// positions for the online softmax. GQA via `gqa_factor` (number
    /// of Q heads per KV head).
    SdpaVector {
        head_dim: usize,
        n_kv: usize,
        n_q_heads: usize,
        gqa_factor: usize,
        batch: usize,
        tpg: usize,
    },
    /// Two-pass decode SDPA. Mirrors `sdpa_decode_2pass_pass1` +
    /// `sdpa_decode_2pass_pass2`. The `BenchSpec` carries pass1; this
    /// variant carries pass2's name + IR getter so a single
    /// `inventory::submit!` describes the whole chained dispatch.
    /// MLX reference is single-pass `sdpa_vector` at the same shape.
    SdpaVector2Pass {
        head_dim: usize,
        n_kv: usize,
        n_q_heads: usize,
        gqa_factor: usize,
        batch: usize,
        blocks: usize,
        pass2_kernel_name: &'static str,
        pass2_kernel_ir: fn(DType) -> Kernel,
    },
    /// Prefill-form scaled dot-product attention (Flash-Attention 2 tile).
    /// Mirrors MLX `steel_attention_{tname}_bq{BQ}_bk{BK}_bd{BD}_wm{WM}_wn{WN}_mask{mname}`.
    /// Per-TG tiles `BQ` Q rows × full `head_dim`, looping over `BK`-wide K/V
    /// blocks with online softmax in registers; causal mask trims the K-block
    /// range. GQA via `gqa_factor`. `q_len`/`k_len` carry the prompt geometry
    /// (self-attn = equal; chunked-prefill = `q_len < k_len` with `qL_off`
    /// shift in the causal mask).
    SdpaPrefill {
        head_dim: usize,
        n_q_heads: usize,
        gqa_factor: usize,
        batch: usize,
        q_len: usize,
        k_len: usize,
        bq: usize,
        bk: usize,
        wm: usize,
        wn: usize,
        tpg: usize,
    },
    /// Batched-Q SDPA decode for speculative decoding (M7). K query
    /// positions share one KV walk per dispatch, amortizing KV bandwidth
    /// by K× vs. K independent `SdpaVector` dispatches at the same shape.
    ///
    /// Two architectures, selected by `variant`:
    ///   * `BatchedDecodeVariant::Decode` (K=2,4): extends `sdpa_decode`'s
    ///     lane-quartile pattern with K independent online-softmax streams.
    ///     Threadgroup of 1024 (32 simdgroups × 32 lanes), same as single-Q.
    ///   * `BatchedDecodeVariant::PrefillTile` (K=8,16): reuses
    ///     `mt_sdpa_prefill_mma` driven with `q_len = batch_q, k_len = n_kv`.
    ///     KV reuse comes from the FA-2 simdgroup-matrix tile (BQ×BK with
    ///     online softmax in registers) — the same pattern dflash-mlx's
    ///     `verify_qmm` uses.
    SdpaBatchedDecode {
        head_dim: usize,
        n_kv: usize,
        n_q_heads: usize,
        gqa_factor: usize,
        /// K — number of query positions sharing the KV walk (2, 4, 8, 16).
        batch_q: usize,
        variant: BatchedDecodeVariant,
        tpg: usize,
    },
    /// Tiled simdgroup GEMM (steel_gemm_fused).
    SteelGemm {
        m: usize,
        n: usize,
        k: usize,
        check_m: usize,
        check_n: usize,
        check_k: usize,
        bm: usize,
        bn: usize,
        tpg: usize,
    },
}

impl BenchDispatch {
    /// The default [`KernelMode`] inferred from the dispatch variant.
    ///
    /// For `Generic` dispatch, returns the mode of the first shape
    /// (or `Elementwise` if there are no shapes). All other variants
    /// map to a fixed mode.
    pub fn default_mode(&self, shapes: &[ShapeSpec]) -> KernelMode {
        match self {
            BenchDispatch::Generic =>
                shapes.first().map(|s| s.mode).unwrap_or(KernelMode::Elementwise),
            BenchDispatch::Sort { .. }
            | BenchDispatch::Scan { .. }
            | BenchDispatch::ArgReduce { .. }
            | BenchDispatch::QuantizedMatVec { .. }
            | BenchDispatch::QuantizedMatMul { .. }
            | BenchDispatch::Attention { .. }
            | BenchDispatch::AffineQuantize { .. }
            | BenchDispatch::SdpaVector { .. }
            | BenchDispatch::SdpaVector2Pass { .. } => KernelMode::Reduction,
            BenchDispatch::SdpaBatchedDecode { variant, .. } => match variant {
                BatchedDecodeVariant::Decode => KernelMode::Reduction,
                BatchedDecodeVariant::PrefillTile { .. } => KernelMode::SimdGroup2D,
            },
            BenchDispatch::Random { .. }
            | BenchDispatch::FpQuantized { .. }
            | BenchDispatch::AffineDequantize { .. } => KernelMode::Elementwise,
            BenchDispatch::Rope { .. } | BenchDispatch::StridedCopy { .. } => KernelMode::Grid3D,
            BenchDispatch::SteelGemm { .. } => KernelMode::SimdGroup2D,
            BenchDispatch::SdpaPrefill { .. } => KernelMode::SimdGroup2D,
        }
    }

    /// The dispatch threadgroup size this variant uses, when one is fixed
    /// at the bench/dispatch layer. `cmd/build.rs` reads this so the MSL
    /// emitted by `tile build --emit all` matches the MSL `tile bench`
    /// measures — both feed the same value into `MslConfig::expected_tpg`.
    ///
    /// Returns `None` for `Generic` (the caller falls back to
    /// `spec.shapes.first().map(|s| s.tpg)` instead, since `Generic`'s TPG
    /// lives on `ShapeSpec`), and for variants without a single fixed TPG
    /// (`Rope`, `StridedCopy` — Grid3D/Elementwise modes that don't use
    /// the Reduction-mode `Op::Reduce` emit; `SdpaVector2Pass` — pass1
    /// dispatches at the kernel's design TPG of 1024 with no spec-level
    /// override).
    pub fn tpg_hint(&self) -> Option<u32> {
        match self {
            BenchDispatch::Generic
            | BenchDispatch::Rope { .. }
            | BenchDispatch::StridedCopy { .. }
            | BenchDispatch::SdpaVector2Pass { .. } => None,
            BenchDispatch::Sort { tpg, .. }
            | BenchDispatch::Scan { tpg, .. }
            | BenchDispatch::ArgReduce { tpg, .. }
            | BenchDispatch::Random { tpg, .. }
            | BenchDispatch::FpQuantized { tpg, .. }
            | BenchDispatch::QuantizedMatVec { tpg, .. }
            | BenchDispatch::QuantizedMatMul { tpg, .. }
            | BenchDispatch::Attention { tpg, .. }
            | BenchDispatch::AffineDequantize { tpg, .. }
            | BenchDispatch::AffineQuantize { tpg, .. }
            | BenchDispatch::SdpaVector { tpg, .. }
            | BenchDispatch::SdpaPrefill { tpg, .. }
            | BenchDispatch::SdpaBatchedDecode { tpg, .. }
            | BenchDispatch::SteelGemm { tpg, .. } => Some(*tpg as u32),
        }
    }
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

/// Select: cond is always 1-byte bool (matching MLX v_Select{T} interface).
pub fn bytes_select(n: usize, _b: usize, _reads: usize, _out: usize, eb: usize) -> usize {
    n + 3 * n * eb // cond(1 byte) + on_true(eb) + on_false(eb) + out(eb)
}

pub fn effective_mode(spec: &BenchSpec) -> metaltile_core::ir::KernelMode {
    spec.kernel_mode.unwrap_or_else(|| spec.dispatch.default_mode(spec.shapes))
}
