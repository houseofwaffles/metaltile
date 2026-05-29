//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    Expr,
    Ident,
    LitFloat,
    LitInt,
    LitStr,
    Token,
    parse::{Parse, ParseStream},
};

pub enum ClassKind {
    Unary,
    Binary,
    AllReduce,
    RowReduce,
    Arange,
    BinaryTwo,
    Select,
    RowNorm,
    Sort,
    Scan,
    ArgReduce,
    Random,
    FpQuantized,
    MatVec,
    MatVecMasked,
    QuantizedMatVec,
    QuantizedMatMul,
    Rope,
    Attention,
    StridedCopy,
    AffineDequantize,
    AffineQuantize,
    SdpaVector,
    SdpaPrefill,
    /// Generic dispatch with empty shapes and explicit kernel_mode.
    /// Used by ffai kernels that register via `BenchDispatch::Generic`
    /// but don't use the auto-generated ShapeSpec from the simple classes.
    GenericEmpty,
    /// Two-pass SDPA decode: pass1 + pass2 chained dispatch.
    /// Requires: h (head_dim), n_kv, n_heads, gqa_factor, batch,
    /// blocks, pass2_kernel (module name).
    SdpaVector2Pass,
    /// Batched-Q SDPA decode for speculative decoding.
    /// Requires: h (head_dim), n_kv, n_heads, gqa_factor, batch_q,
    /// variant (Decode | PrefillTile), tpg.  For PrefillTile variant,
    /// also requires bq, bk, wm, wn.
    SdpaBatchedDecode,
    /// Steel GEMM tile geometry.
    /// Requires: bm, bn, tpg.
    SteelGemm,
}

pub enum InputKind {
    Signed,
    Positive,
    Half,
    Unit,
}

pub enum KernelModeArg {
    None,
    Elementwise,
    Reduction,
    Grid3D,
    SimdGroup2D,
}

/// `BatchedDecodeVariant` discriminator for `SdpaBatchedDecode`.
pub enum BatchedDecodeVariantArg {
    Decode,
    PrefillTile,
}

pub struct BenchArgs {
    pub op: LitStr,
    pub subop: LitStr,
    pub class: ClassKind,
    pub input: InputKind,
    pub input_a: InputKind,
    pub input_b: InputKind,
    pub tol: LitFloat,
    pub start: Option<LitFloat>,
    pub step: Option<LitFloat>,
    pub mlx: Option<LitStr>,
    pub dtypes: Option<Expr>,
    pub metal_file: Option<LitStr>,
    pub kernel_mode: KernelModeArg,
    // RowNorm-specific
    pub reads: Option<LitInt>,
    pub out_elements: Option<LitInt>,
    pub tpg: Option<LitInt>,
    pub pre_weight: Option<LitFloat>,
    pub pre_bias: Option<LitFloat>,
    pub post_eps: Option<LitFloat>,
    // Complex dispatch fields
    pub shapes: Option<Expr>,
    pub n: Option<LitInt>,
    pub check_n: Option<LitInt>,
    pub b: Option<LitInt>,
    pub group_size: Option<LitInt>,
    pub h: Option<LitInt>,
    pub l: Option<LitInt>,
    pub d: Option<LitInt>,
    pub n_per_group: Option<LitInt>,
    pub m: Option<LitInt>,
    pub pad: Option<LitInt>,
    // Affine quantize/dequantize-specific
    pub bits: Option<LitInt>,
    pub n_groups: Option<LitInt>,
    pub batch: Option<LitInt>,
    // SDPA vector
    pub n_kv: Option<LitInt>,
    pub n_heads: Option<LitInt>,
    pub gqa_factor: Option<LitInt>,
    // SDPA prefill (steel_attention tile geometry)
    pub q_len: Option<LitInt>,
    pub k_len: Option<LitInt>,
    pub bq: Option<LitInt>,
    pub bk: Option<LitInt>,
    pub wm: Option<LitInt>,
    pub wn: Option<LitInt>,
    // SdpaVector2Pass
    pub blocks: Option<LitInt>,
    pub pass2_kernel: Option<Ident>,
    // SdpaBatchedDecode
    pub batch_q: Option<LitInt>,
    pub variant: Option<BatchedDecodeVariantArg>,
    // SteelGemm
    pub bm: Option<LitInt>,
    pub bn: Option<LitInt>,
}

fn parse_input(s: &str, span: proc_macro2::Span) -> syn::Result<InputKind> {
    match s {
        "Signed" => Ok(InputKind::Signed),
        "Positive" => Ok(InputKind::Positive),
        "Half" => Ok(InputKind::Half),
        "Unit" => Ok(InputKind::Unit),
        o => Err(syn::Error::new(
            span,
            format!("input must be Signed|Positive|Half|Unit, got `{o}`"),
        )),
    }
}

impl Parse for BenchArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut op: Option<LitStr> = None;
        let mut subop: Option<LitStr> = None;
        let mut class: Option<ClassKind> = None;
        let mut inp: Option<InputKind> = None;
        let mut inp_a: Option<InputKind> = None;
        let mut inp_b: Option<InputKind> = None;
        let mut tol: Option<LitFloat> = None;
        let mut start: Option<LitFloat> = None;
        let mut step: Option<LitFloat> = None;
        let mut mlx: Option<LitStr> = None;
        let mut dtypes: Option<Expr> = None;
        let mut metal_file: Option<LitStr> = None;
        let mut shapes: Option<Expr> = None;
        let mut reads: Option<LitInt> = None;
        let mut out_elements: Option<LitInt> = None;
        let mut tpg_field: Option<LitInt> = None;
        let mut pre_weight_field: Option<LitFloat> = None;
        let mut pre_bias_field: Option<LitFloat> = None;
        let mut post_eps_field: Option<LitFloat> = None;
        let mut n_field: Option<LitInt> = None;
        let mut check_n_field: Option<LitInt> = None;
        let mut b_field: Option<LitInt> = None;
        let mut group_size_field: Option<LitInt> = None;
        let mut h_field: Option<LitInt> = None;
        let mut l_field: Option<LitInt> = None;
        let mut d_field: Option<LitInt> = None;
        let mut n_per_group_field: Option<LitInt> = None;
        let mut m_field: Option<LitInt> = None;
        let mut pad_field: Option<LitInt> = None;
        let mut bits_field: Option<LitInt> = None;
        let mut n_groups_field: Option<LitInt> = None;
        let mut batch_field: Option<LitInt> = None;
        let mut n_kv_field: Option<LitInt> = None;
        let mut n_heads_field: Option<LitInt> = None;
        let mut gqa_factor_field: Option<LitInt> = None;
        let mut q_len_field: Option<LitInt> = None;
        let mut k_len_field: Option<LitInt> = None;
        let mut bq_field: Option<LitInt> = None;
        let mut bk_field: Option<LitInt> = None;
        let mut wm_field: Option<LitInt> = None;
        let mut wn_field: Option<LitInt> = None;
        let mut km_field: Option<KernelModeArg> = None;
        let mut blocks_field: Option<LitInt> = None;
        let mut pass2_kernel_field: Option<Ident> = None;
        let mut batch_q_field: Option<LitInt> = None;
        let mut variant_field: Option<BatchedDecodeVariantArg> = None;
        let mut bm_field: Option<LitInt> = None;
        let mut bn_field: Option<LitInt> = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "op" => op = Some(input.parse()?),
                "subop" => subop = Some(input.parse()?),
                "class" => {
                    let id: Ident = input.parse()?;
                    class = Some(match id.to_string().as_str() {
                        "Unary" => ClassKind::Unary,
                        "Binary" => ClassKind::Binary,
                        "AllReduce" => ClassKind::AllReduce,
                        "RowReduce" => ClassKind::RowReduce,
                        "Arange" => ClassKind::Arange,
                        "BinaryTwo" => ClassKind::BinaryTwo,
                        "Select" => ClassKind::Select,
                        "RowNorm" => ClassKind::RowNorm,
                        "Sort" => ClassKind::Sort,
                        "Scan" => ClassKind::Scan,
                        "ArgReduce" => ClassKind::ArgReduce,
                        "Random" => ClassKind::Random,
                        "FpQuantized" => ClassKind::FpQuantized,
                        "MatVec" => ClassKind::MatVec,
                        "MatVecMasked" => ClassKind::MatVecMasked,
                        "QuantizedMatVec" => ClassKind::QuantizedMatVec,
                        "QuantizedMatMul" => ClassKind::QuantizedMatMul,
                        "Rope" => ClassKind::Rope,
                        "Attention" => ClassKind::Attention,
                        "StridedCopy" => ClassKind::StridedCopy,
                        "AffineDequantize" => ClassKind::AffineDequantize,
                        "AffineQuantize" => ClassKind::AffineQuantize,
                        "SdpaVector" => ClassKind::SdpaVector,
                        "SdpaPrefill" => ClassKind::SdpaPrefill,
                        "GenericEmpty" => ClassKind::GenericEmpty,
                        "SdpaVector2Pass" => ClassKind::SdpaVector2Pass,
                        "SdpaBatchedDecode" => ClassKind::SdpaBatchedDecode,
                        "SteelGemm" => ClassKind::SteelGemm,
                        o => {
                            return Err(syn::Error::new(id.span(), format!("unknown class `{o}`")));
                        },
                    });
                },
                "cpu" | "cpu_c" | "cpu_d" => {
                    // No-op — correctness is via interpreter, not cpu functions.
                    let _: Expr = input.parse()?;
                },
                "input" => {
                    let id: Ident = input.parse()?;
                    inp = Some(parse_input(&id.to_string(), id.span())?);
                },
                "input_a" => {
                    let id: Ident = input.parse()?;
                    inp_a = Some(parse_input(&id.to_string(), id.span())?);
                },
                "input_b" => {
                    let id: Ident = input.parse()?;
                    inp_b = Some(parse_input(&id.to_string(), id.span())?);
                },
                "tol" => tol = Some(input.parse()?),
                "start" => start = Some(input.parse()?),
                "step" => step = Some(input.parse()?),
                "mlx" => mlx = Some(input.parse()?),
                "dtypes" => dtypes = Some(input.parse()?),
                "metal_file" => metal_file = Some(input.parse()?),
                "kernel_mode" => {
                    let id: Ident = input.parse()?;
                    let km = match id.to_string().as_str() {
                        "None" => KernelModeArg::None,
                        "Elementwise" => KernelModeArg::Elementwise,
                        "Reduction" => KernelModeArg::Reduction,
                        "Grid3D" => KernelModeArg::Grid3D,
                        "SimdGroup2D" => KernelModeArg::SimdGroup2D,
                        o =>
                            return Err(syn::Error::new(
                                id.span(),
                                format!(
                                    "kernel_mode must be None|Elementwise|Reduction|Grid3D|SimdGroup2D, got `{o}`"
                                ),
                            )),
                    };
                    km_field = Some(km);
                },
                "shapes" => shapes = Some(input.parse()?),
                "reads" => reads = Some(input.parse()?),
                "out_elements" => out_elements = Some(input.parse()?),
                "tpg" => tpg_field = Some(input.parse()?),
                "pre_weight" => pre_weight_field = Some(input.parse()?),
                "pre_bias" => pre_bias_field = Some(input.parse()?),
                "post_eps" => post_eps_field = Some(input.parse()?),
                "n" => n_field = Some(input.parse()?),
                "check_n" => check_n_field = Some(input.parse()?),
                "b" => b_field = Some(input.parse()?),
                "group_size" => group_size_field = Some(input.parse()?),
                "h" => h_field = Some(input.parse()?),
                "l" => l_field = Some(input.parse()?),
                "d" => d_field = Some(input.parse()?),
                "n_per_group" => n_per_group_field = Some(input.parse()?),
                "m" => m_field = Some(input.parse()?),
                "pad" => pad_field = Some(input.parse()?),
                "bits" => bits_field = Some(input.parse()?),
                "n_groups" => n_groups_field = Some(input.parse()?),
                "batch" => batch_field = Some(input.parse()?),
                "n_kv" => n_kv_field = Some(input.parse()?),
                "n_heads" => n_heads_field = Some(input.parse()?),
                "gqa_factor" => gqa_factor_field = Some(input.parse()?),
                "q_len" => q_len_field = Some(input.parse()?),
                "k_len" => k_len_field = Some(input.parse()?),
                "bq" => bq_field = Some(input.parse()?),
                "bk" => bk_field = Some(input.parse()?),
                "wm" => wm_field = Some(input.parse()?),
                "wn" => wn_field = Some(input.parse()?),
                "blocks" => blocks_field = Some(input.parse()?),
                "pass2_kernel" => pass2_kernel_field = Some(input.parse()?),
                "batch_q" => batch_q_field = Some(input.parse()?),
                "variant" => {
                    let id: Ident = input.parse()?;
                    let var = match id.to_string().as_str() {
                        "Decode" => BatchedDecodeVariantArg::Decode,
                        "PrefillTile" => BatchedDecodeVariantArg::PrefillTile,
                        o =>
                            return Err(syn::Error::new(
                                id.span(),
                                format!("variant must be Decode|PrefillTile, got `{o}`"),
                            )),
                    };
                    variant_field = Some(var);
                },
                "bm" => bm_field = Some(input.parse()?),
                "bn" => bn_field = Some(input.parse()?),
                o => {
                    return Err(syn::Error::new(key.span(), format!("unknown bench arg: `{o}`")));
                },
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(BenchArgs {
            op: op.ok_or_else(|| input.error("missing `op`"))?,
            subop: subop.ok_or_else(|| input.error("missing `subop`"))?,
            class: class.ok_or_else(|| input.error("missing `class`"))?,
            tol: tol.ok_or_else(|| input.error("missing `tol`"))?,
            start,
            step,
            input: inp.unwrap_or(InputKind::Half),
            input_a: inp_a.unwrap_or(InputKind::Half),
            input_b: inp_b.unwrap_or(InputKind::Half),
            mlx,
            dtypes,
            metal_file,
            kernel_mode: km_field.unwrap_or(KernelModeArg::None),
            shapes,
            reads,
            out_elements,
            tpg: tpg_field,
            pre_weight: pre_weight_field,
            pre_bias: pre_bias_field,
            post_eps: post_eps_field,
            n: n_field,
            check_n: check_n_field,
            b: b_field,
            group_size: group_size_field,
            h: h_field,
            l: l_field,
            d: d_field,
            n_per_group: n_per_group_field,
            m: m_field,
            pad: pad_field,
            bits: bits_field,
            n_groups: n_groups_field,
            batch: batch_field,
            n_kv: n_kv_field,
            n_heads: n_heads_field,
            gqa_factor: gqa_factor_field,
            q_len: q_len_field,
            k_len: k_len_field,
            bq: bq_field,
            bk: bk_field,
            wm: wm_field,
            wn: wn_field,
            blocks: blocks_field,
            pass2_kernel: pass2_kernel_field,
            batch_q: batch_q_field,
            variant: variant_field,
            bm: bm_field,
            bn: bn_field,
        })
    }
}

fn input_buf_init_ts(k: &InputKind) -> TokenStream {
    match k {
        InputKind::Signed => quote! { crate::spec::BufInit::Signed },
        InputKind::Positive => quote! { crate::spec::BufInit::Positive },
        InputKind::Half => quote! { crate::spec::BufInit::Half },
        InputKind::Unit => quote! { crate::spec::BufInit::Unit },
    }
}

fn opt_str(v: &Option<LitStr>) -> TokenStream {
    match v {
        Some(s) => quote! {Some(#s)},
        None => quote! {None},
    }
}

fn kernel_mode_ts(km: &KernelModeArg) -> TokenStream {
    match km {
        KernelModeArg::None => quote! { None },
        KernelModeArg::Elementwise => quote! { Some(metaltile_core::ir::KernelMode::Elementwise) },
        KernelModeArg::Reduction => quote! { Some(metaltile_core::ir::KernelMode::Reduction) },
        KernelModeArg::Grid3D => quote! { Some(metaltile_core::ir::KernelMode::Grid3D) },
        KernelModeArg::SimdGroup2D => quote! { Some(metaltile_core::ir::KernelMode::SimdGroup2D) },
    }
}
pub fn generate_submit(fn_name: &syn::Ident, a: &BenchArgs, is_generic: bool) -> TokenStream {
    let op = &a.op;
    let subop = &a.subop;
    let fn_str = fn_name.to_string();
    let tol = &a.tol;
    let mlx_pat = opt_str(&a.mlx);
    let mlx_src = match &a.metal_file {
        Some(f) => {
            let path = f.value();
            quote! { Some(include_str!(concat!(env!("OUT_DIR"), "/metal/", #path))) }
        },
        None => quote! { None },
    };
    let dtypes = match &a.dtypes {
        Some(e) => quote! {#e},
        None => quote! {crate::bench_types::FLOAT_DTYPES},
    };

    // Non-generic kernels have `kernel_ir_for() -> Kernel` (no args).
    // BenchSpec.kernel_ir is `fn(DType) -> Kernel`, so we wrap in a lambda.
    let kernel_ir_expr = if is_generic {
        quote! { #fn_name::kernel_ir_for }
    } else {
        quote! { |_: metaltile_core::dtype::DType| #fn_name::kernel_ir_for() }
    };

    // For Generic dispatch: produce (shapes_ts, BenchDispatch::Generic).
    // For complex dispatch: produce (&[], BenchDispatch::Foo{...}).
    let (shapes_ts, dispatch_ts) = match &a.class {
        // ── Generic: Unary ───────────────────────────────────────────────
        ClassKind::Unary => {
            let inp = input_buf_init_ts(&a.input);
            let sh = quote! { &[crate::spec::ShapeSpec {
                label: "N=64M",
                n: crate::spec::ELEMENTWISE_N_BENCH, b: 1usize,
                check_n: crate::spec::ELEMENTWISE_N_CHECK, check_b: 1usize,
                mode: metaltile_core::ir::KernelMode::Elementwise,
                tpg: crate::spec::ELEMENTWISE_TPG,
                grid: crate::spec::DispatchGrid::DivCeilN,
                tensor_bufs: &[
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: #inp, dtype_override: None },
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::Zeros, dtype_override: None },
                ],
                scalar_bufs: &[crate::spec::ScalarBufSpec::U32N],
                cexprs: &[("n", crate::spec::Dim::N)],
                out_elems: crate::spec::Dim::N,
                reads: 1usize,
                bytes_fn: crate::spec::bytes_elementwise,
                mlx_args: Some(&[
                    crate::spec::MlxArg::TensorBuf(0),
                    crate::spec::MlxArg::FreshOut(0),
                    crate::spec::MlxArg::U32N,
                ]),
                mlx_grid: None,
                mlx_tpg: 0usize,
            }] };
            (sh, quote! { crate::spec::BenchDispatch::Generic })
        },
        // ── Generic: Binary ──────────────────────────────────────────────
        ClassKind::Binary => {
            let ia = input_buf_init_ts(&a.input_a);
            let ib = input_buf_init_ts(&a.input_b);
            let sh = quote! { &[crate::spec::ShapeSpec {
                label: "N=64M",
                n: crate::spec::ELEMENTWISE_N_BENCH, b: 1usize,
                check_n: crate::spec::ELEMENTWISE_N_CHECK, check_b: 1usize,
                mode: metaltile_core::ir::KernelMode::Elementwise,
                tpg: crate::spec::BINARY_TPG,
                grid: crate::spec::DispatchGrid::DivCeilN,
                tensor_bufs: &[
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: #ia, dtype_override: None },
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: #ib, dtype_override: None },
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::Zeros, dtype_override: None },
                ],
                scalar_bufs: &[crate::spec::ScalarBufSpec::U64N],
                cexprs: &[("n", crate::spec::Dim::N)],
                out_elems: crate::spec::Dim::N,
                reads: 2usize,
                bytes_fn: crate::spec::bytes_elementwise,
                mlx_args: Some(&[
                    crate::spec::MlxArg::TensorBuf(0),
                    crate::spec::MlxArg::TensorBuf(1),
                    crate::spec::MlxArg::FreshOut(0),
                    crate::spec::MlxArg::U64N,
                    crate::spec::MlxArg::Zeros8,
                ]),
                mlx_grid: Some(crate::spec::DispatchGrid::DivCeilN2),
                mlx_tpg: 0usize,
            }] };
            (sh, quote! { crate::spec::BenchDispatch::Generic })
        },
        // ── Generic: AllReduce ───────────────────────────────────────────
        ClassKind::AllReduce => {
            let sh = quote! { &[crate::spec::ShapeSpec {
                label: "N=64M",
                n: crate::spec::ALL_REDUCE_N, b: 1usize,
                check_n: crate::spec::ALL_REDUCE_N_CHECK, check_b: 1usize,
                mode: metaltile_core::ir::KernelMode::Reduction,
                tpg: crate::spec::ALL_REDUCE_TPG,
                grid: crate::spec::DispatchGrid::Single,
                tensor_bufs: &[
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::Signed, dtype_override: None },
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::One, init: crate::spec::BufInit::Zeros, dtype_override: None },
                ],
                scalar_bufs: &[crate::spec::ScalarBufSpec::U32N],
                cexprs: &[("n", crate::spec::Dim::N)],
                out_elems: crate::spec::Dim::One,
                reads: 1usize,
                bytes_fn: crate::spec::bytes_row_op,
                mlx_args: Some(&[
                    crate::spec::MlxArg::TensorBuf(0),
                    crate::spec::MlxArg::FreshOut(1),
                    crate::spec::MlxArg::U64N,
                    crate::spec::MlxArg::U64N,
                ]),
                mlx_grid: Some(crate::spec::DispatchGrid::Single),
                mlx_tpg: 0usize,
            }] };
            (sh, quote! { crate::spec::BenchDispatch::Generic })
        },
        // ── Generic: RowReduce ───────────────────────────────────────────
        ClassKind::RowReduce => {
            let sh = quote! { &[crate::spec::ShapeSpec {
                label: "B=1024 N=4096",
                n: crate::spec::ROW_REDUCE_SHAPES[0].1, b: crate::spec::ROW_REDUCE_SHAPES[0].0,
                check_n: crate::spec::ROW_REDUCE_CHECK_N, check_b: crate::spec::ROW_REDUCE_CHECK_B,
                mode: metaltile_core::ir::KernelMode::Reduction,
                tpg: crate::spec::ROW_REDUCE_TPG,
                grid: crate::spec::DispatchGrid::RowsB,
                tensor_bufs: &[
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::BxN, init: crate::spec::BufInit::Signed, dtype_override: None },
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::B, init: crate::spec::BufInit::Zeros, dtype_override: None },
                ],
                scalar_bufs: &[crate::spec::ScalarBufSpec::U32N],
                cexprs: &[("n", crate::spec::Dim::N)],
                out_elems: crate::spec::Dim::B,
                reads: 1usize,
                bytes_fn: crate::spec::bytes_row_op,
                mlx_args: Some(&[
                    crate::spec::MlxArg::TensorBuf(0),
                    crate::spec::MlxArg::FreshOut(1),
                    crate::spec::MlxArg::U64N,
                    crate::spec::MlxArg::I64B,
                ]),
                mlx_grid: Some(crate::spec::DispatchGrid::RowsBY),
                mlx_tpg: 0usize,
            }] };
            (sh, quote! { crate::spec::BenchDispatch::Generic })
        },
        // ── Generic: Arange ──────────────────────────────────────────────
        ClassKind::Arange => {
            let s_val = a.start.as_ref().map(|f| quote! {#f as f32}).unwrap_or(quote! {0.0f32});
            let st_val = a.step.as_ref().map(|f| quote! {#f as f32}).unwrap_or(quote! {1.0f32});
            let sh = quote! { &[crate::spec::ShapeSpec {
                label: "N=64M",
                n: crate::spec::ARANGE_N, b: 1usize,
                check_n: crate::spec::ARANGE_N_CHECK, check_b: 1usize,
                mode: metaltile_core::ir::KernelMode::Elementwise,
                tpg: crate::spec::ARANGE_TPG,
                grid: crate::spec::DispatchGrid::DivCeilN,
                tensor_bufs: &[
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::Zeros, dtype_override: None },
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::One, init: crate::spec::BufInit::Fill(#s_val), dtype_override: None },
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::One, init: crate::spec::BufInit::Fill(#st_val), dtype_override: None },
                ],
                scalar_bufs: &[crate::spec::ScalarBufSpec::U32N],
                cexprs: &[("n", crate::spec::Dim::N)],
                out_elems: crate::spec::Dim::N,
                reads: 0usize,
                bytes_fn: crate::spec::bytes_elementwise,
                mlx_args: Some(&[
                    crate::spec::MlxArg::TensorBuf(1),
                    crate::spec::MlxArg::TensorBuf(2),
                    crate::spec::MlxArg::FreshOut(0),
                ]),
                mlx_grid: None,
                mlx_tpg: 0usize,
            }] };
            (sh, quote! { crate::spec::BenchDispatch::Generic })
        },
        // ── Generic: BinaryTwo ───────────────────────────────────────────
        ClassKind::BinaryTwo => {
            let ia = input_buf_init_ts(&a.input_a);
            let ib = input_buf_init_ts(&a.input_b);
            let sh = quote! { &[crate::spec::ShapeSpec {
                label: "N=64M",
                n: crate::spec::ELEMENTWISE_N_BENCH, b: 1usize,
                check_n: crate::spec::ELEMENTWISE_N_CHECK, check_b: 1usize,
                mode: metaltile_core::ir::KernelMode::Elementwise,
                tpg: crate::spec::BINARY_TWO_TPG,
                grid: crate::spec::DispatchGrid::DivCeilN,
                tensor_bufs: &[
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: #ia, dtype_override: None },
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: #ib, dtype_override: None },
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::Zeros, dtype_override: None },
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::Zeros, dtype_override: None },
                ],
                scalar_bufs: &[],
                cexprs: &[],
                out_elems: crate::spec::Dim::N,
                reads: 3usize,
                bytes_fn: crate::spec::bytes_elementwise,
                mlx_args: None,
                mlx_grid: None,
                mlx_tpg: 0usize,
            }] };
            (sh, quote! { crate::spec::BenchDispatch::Generic })
        },
        // ── Generic: Select ──────────────────────────────────────────────
        ClassKind::Select => {
            let sh = quote! { &[crate::spec::ShapeSpec {
                label: "N=64M",
                n: crate::spec::ELEMENTWISE_N_BENCH, b: 1usize,
                check_n: crate::spec::ELEMENTWISE_N_CHECK, check_b: 1usize,
                mode: metaltile_core::ir::KernelMode::Elementwise,
                tpg: crate::spec::SELECT_TPG,
                grid: crate::spec::DispatchGrid::DivCeilN,
                tensor_bufs: &[
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::AltZeroOne, dtype_override: Some(metaltile_core::DType::U8) },
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::Half, dtype_override: None },
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::Half, dtype_override: None },
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::Zeros, dtype_override: None },
                ],
                scalar_bufs: &[],
                cexprs: &[],
                out_elems: crate::spec::Dim::N,
                reads: 3usize,
                bytes_fn: crate::spec::bytes_select,
                mlx_args: Some(&[
                    crate::spec::MlxArg::BoolAltN,
                    crate::spec::MlxArg::TensorBuf(1),
                    crate::spec::MlxArg::TensorBuf(2),
                    crate::spec::MlxArg::FreshOut(3),
                    crate::spec::MlxArg::U32N,
                ]),
                mlx_grid: None,
                mlx_tpg: 0usize,
            }] };
            (sh, quote! { crate::spec::BenchDispatch::Generic })
        },
        // ── Generic: RowNorm (softmax, rms_norm, layer_norm, logsumexp) ──
        ClassKind::RowNorm => {
            let b_val = a.b.as_ref().expect("RowNorm requires b=");
            let n_val = a.n.as_ref().expect("RowNorm requires n=");
            let tpg_val = a.tpg.as_ref().expect("RowNorm requires tpg=");
            let reads_val = a.reads.as_ref().expect("RowNorm requires reads=");
            let b_lit: usize = b_val.base10_parse().unwrap_or(1024);
            let n_lit: usize = n_val.base10_parse().unwrap_or(4096);
            let label = format!("B={b_lit} N={n_lit}");
            let inp_init = input_buf_init_ts(&a.input);

            // out_elements=1 → per-row scalar output (logsumexp), else full BxN
            let out_per_row = a
                .out_elements
                .as_ref()
                .and_then(|e| e.base10_parse::<usize>().ok())
                .map(|v| v == 1)
                .unwrap_or(false);
            let out_dim_ts = if out_per_row {
                quote! { crate::spec::Dim::B }
            } else {
                quote! { crate::spec::Dim::BxN }
            };

            let has_weight = a.pre_weight.is_some();
            let has_bias = a.pre_bias.is_some();
            let has_eps = a.post_eps.is_some();
            let pre_count = has_weight as usize + has_bias as usize;
            let out_idx = 1usize + pre_count;
            let eps_idx = out_idx + 1usize;

            // Build tensor_bufs token list
            let mut tbufs: Vec<TokenStream> = vec![
                quote! { crate::spec::TensorBufSpec { count: crate::spec::Dim::BxN, init: #inp_init, dtype_override: None } },
            ];
            if let Some(wv) = &a.pre_weight {
                tbufs.push(quote! { crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::Fill(#wv as f32), dtype_override: None } });
            }
            if let Some(bv) = &a.pre_bias {
                tbufs.push(quote! { crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::Fill(#bv as f32), dtype_override: None } });
            }
            tbufs.push(quote! { crate::spec::TensorBufSpec { count: #out_dim_ts, init: crate::spec::BufInit::Zeros, dtype_override: None } });
            if let Some(ev) = &a.post_eps {
                tbufs.push(quote! { crate::spec::TensorBufSpec { count: crate::spec::Dim::One, init: crate::spec::BufInit::Fill(#ev as f32), dtype_override: Some(metaltile_core::dtype::DType::F32) } });
            }

            // Build mlx_args token list (only if mlx pattern is set)
            let mlx_args_ts = if a.mlx.is_some() {
                let mut margs: Vec<TokenStream> =
                    vec![quote! { crate::spec::MlxArg::TensorBuf(0) }];
                for i in 1..=pre_count {
                    margs.push(quote! { crate::spec::MlxArg::TensorBuf(#i) });
                }
                margs.push(quote! { crate::spec::MlxArg::FreshOut(#out_idx) });
                if has_eps {
                    margs.push(quote! { crate::spec::MlxArg::TensorBuf(#eps_idx) });
                }
                margs.push(quote! { crate::spec::MlxArg::U32N });
                for _ in 0..pre_count {
                    margs.push(quote! { crate::spec::MlxArg::U32V(1u32) });
                }
                quote! { Some(&[ #(#margs),* ]) }
            } else {
                quote! { None }
            };

            let sh = quote! { &[crate::spec::ShapeSpec {
                label: #label,
                n: #n_val as usize, b: #b_val as usize,
                check_n: #n_val as usize, check_b: 4usize,
                mode: metaltile_core::ir::KernelMode::Reduction,
                tpg: #tpg_val as usize,
                grid: crate::spec::DispatchGrid::RowsB,
                tensor_bufs: &[ #(#tbufs),* ],
                scalar_bufs: &[crate::spec::ScalarBufSpec::U32N],
                cexprs: &[("n", crate::spec::Dim::N)],
                out_elems: #out_dim_ts,
                reads: #reads_val as usize,
                bytes_fn: crate::spec::bytes_row_op,
                mlx_args: #mlx_args_ts,
                mlx_grid: None,
                // MLX `looped_softmax_*` and `looped_logsumexp_*` use
                // `threadgroup AccT local_max[SIMD_SIZE=32]` (and
                // `local_normalizer[32]`) without zero-initialising the
                // slots past `simd_group_id`. The subsequent
                // `simd_max(local_max[simd_lane_id])` reads all 32 slots
                // — when MLX dispatches these it does so at
                // `kernel->maxTotalThreadsPerThreadgroup() == 1024`, so
                // `n_simd == 32` and every slot is initialised. We
                // dispatch with `tpg=256` for MT-side cooperative
                // reductions, which would leave 24 slots holding
                // garbage and produces NaN outputs. Pin MLX dispatch to
                // 1024 to sidestep it; `rms_*` and `layer_norm_looped*`
                // zero-init their threadgroup arrays explicitly and
                // are unaffected by a larger MLX tpg.
                mlx_tpg: 1024usize,
            }] };
            (sh, quote! { crate::spec::BenchDispatch::Generic })
        },
        // ── Generic: MatVec ──────────────────────────────────────────────
        ClassKind::MatVec => {
            let b_val = a.b.as_ref().expect("MatVec requires b=");
            let n_val = a.n.as_ref().expect("MatVec requires n=");
            let tpg_val = a.tpg.as_ref().expect("MatVec requires tpg=");
            let b_lit: usize = b_val.base10_parse().unwrap_or(4096);
            let n_lit: usize = n_val.base10_parse().unwrap_or(4096);
            let label = format!("B={b_lit} N={n_lit}");
            let sh = quote! { &[crate::spec::ShapeSpec {
                label: #label,
                n: #n_val as usize, b: #b_val as usize,
                check_n: 128usize, check_b: 4usize,
                mode: metaltile_core::ir::KernelMode::Reduction,
                tpg: #tpg_val as usize,
                grid: crate::spec::DispatchGrid::RowsB,
                tensor_bufs: &[
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::BxN, init: crate::spec::BufInit::Signed, dtype_override: None },
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::Signed, dtype_override: None },
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::B, init: crate::spec::BufInit::Zeros, dtype_override: None },
                ],
                scalar_bufs: &[crate::spec::ScalarBufSpec::U32N],
                cexprs: &[("k", crate::spec::Dim::N)],
                out_elems: crate::spec::Dim::B,
                reads: 1usize,
                bytes_fn: crate::spec::bytes_mat_vec,
                mlx_args: None,
                mlx_grid: None,
                mlx_tpg: 0usize,
            }] };
            (sh, quote! { crate::spec::BenchDispatch::Generic })
        },
        // ── Generic: MatVecMasked ────────────────────────────────────────
        ClassKind::MatVecMasked => {
            let b_val = a.b.as_ref().expect("MatVecMasked requires b=");
            let n_val = a.n.as_ref().expect("MatVecMasked requires n=");
            let tpg_val = a.tpg.as_ref().expect("MatVecMasked requires tpg=");
            let b_lit: usize = b_val.base10_parse().unwrap_or(4096);
            let n_lit: usize = n_val.base10_parse().unwrap_or(4096);
            let label = format!("B={b_lit} N={n_lit}");
            let sh = quote! { &[crate::spec::ShapeSpec {
                label: #label,
                n: #n_val as usize, b: #b_val as usize,
                check_n: 128usize, check_b: 4usize,
                mode: metaltile_core::ir::KernelMode::Reduction,
                tpg: #tpg_val as usize,
                grid: crate::spec::DispatchGrid::RowsB,
                tensor_bufs: &[
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::BxN, init: crate::spec::BufInit::Signed, dtype_override: None },
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::Signed, dtype_override: None },
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::AltZeroOne, dtype_override: None },
                    crate::spec::TensorBufSpec { count: crate::spec::Dim::B, init: crate::spec::BufInit::Zeros, dtype_override: None },
                ],
                scalar_bufs: &[crate::spec::ScalarBufSpec::U32N],
                cexprs: &[("k", crate::spec::Dim::N)],
                out_elems: crate::spec::Dim::B,
                reads: 1usize,
                bytes_fn: crate::spec::bytes_mat_vec_masked,
                mlx_args: None,
                mlx_grid: None,
                mlx_tpg: 0usize,
            }] };
            (sh, quote! { crate::spec::BenchDispatch::Generic })
        },
        // ── Complex: Sort ────────────────────────────────────────────────
        ClassKind::Sort => {
            let b_val = a.b.as_ref().expect("Sort requires b");
            let n_val = a.n.as_ref().expect("Sort requires n");
            let tpg_val = a.tpg.as_ref().expect("Sort requires tpg");
            (quote! { &[] }, quote! {
                crate::spec::BenchDispatch::Sort {
                    b: #b_val as usize, n: #n_val as usize, tpg: #tpg_val as usize,
                }
            })
        },
        // ── Complex: Scan ────────────────────────────────────────────────
        ClassKind::Scan => {
            let sh = a.shapes.as_ref().expect("Scan requires shapes");
            let tpg_val = a.tpg.as_ref().expect("Scan requires tpg");
            (quote! { &[] }, quote! {
                crate::spec::BenchDispatch::Scan { shapes: #sh, tpg: #tpg_val as usize }
            })
        },
        // ── Complex: ArgReduce ───────────────────────────────────────────
        ClassKind::ArgReduce => {
            let n_val = a.n.as_ref().expect("ArgReduce requires n");
            let cn_val = a.check_n.as_ref().expect("ArgReduce requires check_n");
            let tpg_val = a.tpg.as_ref().expect("ArgReduce requires tpg");
            (quote! { &[] }, quote! {
                crate::spec::BenchDispatch::ArgReduce {
                    n: #n_val as usize, check_n: #cn_val as usize, tpg: #tpg_val as usize,
                }
            })
        },
        // ── Complex: Random ──────────────────────────────────────────────
        ClassKind::Random => {
            let n_val = a.n.as_ref().expect("Random requires n");
            let tpg_val = a.tpg.as_ref().expect("Random requires tpg");
            (quote! { &[] }, quote! {
                crate::spec::BenchDispatch::Random { n: #n_val as usize, tpg: #tpg_val as usize }
            })
        },
        // ── Complex: FpQuantized ─────────────────────────────────────────
        ClassKind::FpQuantized => {
            let n_val = a.n.as_ref().expect("FpQuantized requires n");
            let tpg_val = a.tpg.as_ref().expect("FpQuantized requires tpg");
            (quote! { &[] }, quote! {
                crate::spec::BenchDispatch::FpQuantized {
                    n: #n_val as usize, tpg: #tpg_val as usize,
                }
            })
        },
        // ── Complex: QuantizedMatVec ─────────────────────────────────────
        ClassKind::QuantizedMatVec => {
            let sh = a.shapes.as_ref().expect("QuantizedMatVec requires shapes");
            let gs = a.group_size.as_ref().expect("QuantizedMatVec requires group_size");
            let tpg_val = a.tpg.as_ref().expect("QuantizedMatVec requires tpg");
            // `bits` defaults to 4 (the int4 contract `run_quantized_mat_vec`
            // used to hard-code) so every existing `QuantizedMatVec` kernel
            // keeps its current behaviour. int8 perf kernels opt in with
            // `bits=8`.
            let bits_ts = match a.bits.as_ref() {
                Some(b) => quote! { #b as u32 },
                None => quote! { 4u32 },
            };
            (quote! { &[] }, quote! {
                crate::spec::BenchDispatch::QuantizedMatVec {
                    shapes: #sh,
                    group_size: #gs as usize,
                    tpg: #tpg_val as usize,
                    bits: #bits_ts,
                }
            })
        },
        // ── Complex: QuantizedMatMul (B>1 / prefill) ─────────────────────
        ClassKind::QuantizedMatMul => {
            let sh = a.shapes.as_ref().expect("QuantizedMatMul requires shapes");
            let gs = a.group_size.as_ref().expect("QuantizedMatMul requires group_size");
            let tpg_val = a.tpg.as_ref().expect("QuantizedMatMul requires tpg");
            let m_val = a.m.as_ref().expect("QuantizedMatMul requires m (token count)");
            let bits_ts = match a.bits.as_ref() {
                Some(b) => quote! { #b as u32 },
                None => quote! { 4u32 },
            };
            (quote! { &[] }, quote! {
                crate::spec::BenchDispatch::QuantizedMatMul {
                    shapes: #sh,
                    m: #m_val as usize,
                    group_size: #gs as usize,
                    tpg: #tpg_val as usize,
                    bits: #bits_ts,
                }
            })
        },
        // ── Complex: Rope ────────────────────────────────────────────────
        ClassKind::Rope => {
            let b_val = a.b.as_ref().expect("Rope requires b");
            let h_val = a.h.as_ref().expect("Rope requires h");
            let l_val = a.l.as_ref().expect("Rope requires l");
            let d_val = a.d.as_ref().expect("Rope requires d");
            let npg = a.n_per_group.as_ref().expect("Rope requires n_per_group");
            (quote! { &[] }, quote! {
                crate::spec::BenchDispatch::Rope {
                    b: #b_val as usize, h: #h_val as usize,
                    l: #l_val as usize, d: #d_val as usize,
                    n_per_group: #npg as usize,
                }
            })
        },
        // ── Complex: Attention ───────────────────────────────────────────
        ClassKind::Attention => {
            let sh = a.shapes.as_ref().expect("Attention requires shapes");
            let tpg_val = a.tpg.as_ref().expect("Attention requires tpg");
            (quote! { &[] }, quote! {
                crate::spec::BenchDispatch::Attention { shapes: #sh, tpg: #tpg_val as usize }
            })
        },
        // ── Complex: StridedCopy ─────────────────────────────────────────
        ClassKind::StridedCopy => {
            let m_val = a.m.as_ref().expect("StridedCopy requires m");
            let n_val = a.n.as_ref().expect("StridedCopy requires n");
            let pad_val = a.pad.as_ref().expect("StridedCopy requires pad");
            (quote! { &[] }, quote! {
                crate::spec::BenchDispatch::StridedCopy {
                    m: #m_val as usize, n: #n_val as usize, pad: #pad_val as usize,
                }
            })
        },
        // ── Complex: AffineDequantize ────────────────────────────────────
        ClassKind::AffineDequantize => {
            let bits_val = a.bits.as_ref().expect("AffineDequantize requires bits");
            let gs = a.group_size.as_ref().expect("AffineDequantize requires group_size");
            let ng = a.n_groups.as_ref().expect("AffineDequantize requires n_groups");
            let batch = a.batch.as_ref().expect("AffineDequantize requires batch");
            let tpg_val = a.tpg.as_ref().expect("AffineDequantize requires tpg");
            (quote! { &[] }, quote! {
                crate::spec::BenchDispatch::AffineDequantize {
                    bits: #bits_val as usize,
                    group_size: #gs as usize,
                    n_groups: #ng as usize,
                    batch: #batch as usize,
                    tpg: #tpg_val as usize,
                }
            })
        },
        // ── Complex: AffineQuantize ──────────────────────────────────────
        ClassKind::AffineQuantize => {
            let bits_val = a.bits.as_ref().expect("AffineQuantize requires bits");
            let gs = a.group_size.as_ref().expect("AffineQuantize requires group_size");
            let ng = a.n_groups.as_ref().expect("AffineQuantize requires n_groups");
            let batch = a.batch.as_ref().expect("AffineQuantize requires batch");
            let tpg_val = a.tpg.as_ref().expect("AffineQuantize requires tpg");
            (quote! { &[] }, quote! {
                crate::spec::BenchDispatch::AffineQuantize {
                    bits: #bits_val as usize,
                    group_size: #gs as usize,
                    n_groups: #ng as usize,
                    batch: #batch as usize,
                    tpg: #tpg_val as usize,
                }
            })
        },
        // ── Complex: SdpaVector (decode-form SDPA) ───────────────────────
        ClassKind::SdpaVector => {
            let hd = a.h.as_ref().expect("SdpaVector requires h (head_dim)");
            let nkv = a.n_kv.as_ref().expect("SdpaVector requires n_kv");
            let nh = a.n_heads.as_ref().expect("SdpaVector requires n_heads (Q heads)");
            let gqa = a.gqa_factor.as_ref().expect("SdpaVector requires gqa_factor (Q-per-KV)");
            let batch = a.batch.as_ref().expect("SdpaVector requires batch");
            let tpg_val = a.tpg.as_ref().expect("SdpaVector requires tpg");
            (quote! { &[] }, quote! {
                crate::spec::BenchDispatch::SdpaVector {
                    head_dim: #hd as usize,
                    n_kv: #nkv as usize,
                    n_q_heads: #nh as usize,
                    gqa_factor: #gqa as usize,
                    batch: #batch as usize,
                    tpg: #tpg_val as usize,
                }
            })
        },
        // ── Complex: SdpaPrefill (steel_attention Flash-Attention 2 tile) ─
        ClassKind::SdpaPrefill => {
            let hd = a.h.as_ref().expect("SdpaPrefill requires h (head_dim)");
            let nh = a.n_heads.as_ref().expect("SdpaPrefill requires n_heads (Q heads)");
            let gqa = a.gqa_factor.as_ref().expect("SdpaPrefill requires gqa_factor (Q-per-KV)");
            let batch = a.batch.as_ref().expect("SdpaPrefill requires batch");
            let qlen = a.q_len.as_ref().expect("SdpaPrefill requires q_len");
            let klen = a.k_len.as_ref().expect("SdpaPrefill requires k_len");
            let bq_val = a.bq.as_ref().expect("SdpaPrefill requires bq");
            let bk_val = a.bk.as_ref().expect("SdpaPrefill requires bk");
            let wm_val = a.wm.as_ref().expect("SdpaPrefill requires wm");
            let wn_val = a.wn.as_ref().expect("SdpaPrefill requires wn");
            let shapes_ts = a.shapes.as_ref().map(|s| quote! { #s }).unwrap_or(quote! { &[] });
            let tpg_val = a.tpg.as_ref().expect("SdpaPrefill requires tpg");
            (quote! { #shapes_ts }, quote! {
                crate::spec::BenchDispatch::SdpaPrefill {
                    head_dim: #hd as usize,
                    n_q_heads: #nh as usize,
                    gqa_factor: #gqa as usize,
                    batch: #batch as usize,
                    q_len: #qlen as usize,
                    k_len: #klen as usize,
                    bq: #bq_val as usize,
                    bk: #bk_val as usize,
                    wm: #wm_val as usize,
                    wn: #wn_val as usize,
                    tpg: #tpg_val as usize,
                }
            })
        },
        // ── GenericEmpty: Generic dispatch with empty shapes ────────────
        ClassKind::GenericEmpty => (quote! { &[] }, quote! { crate::spec::BenchDispatch::Generic }),
        // ── Complex: SdpaVector2Pass (two-pass SDPA decode) ────────────
        ClassKind::SdpaVector2Pass => {
            let hd = a.h.as_ref().expect("SdpaVector2Pass requires h (head_dim)");
            let nkv = a.n_kv.as_ref().expect("SdpaVector2Pass requires n_kv");
            let nh = a.n_heads.as_ref().expect("SdpaVector2Pass requires n_heads (Q heads)");
            let gqa = a.gqa_factor.as_ref().expect("SdpaVector2Pass requires gqa_factor");
            let batch = a.batch.as_ref().expect("SdpaVector2Pass requires batch");
            let blocks_val = a.blocks.as_ref().expect("SdpaVector2Pass requires blocks");
            let p2k = a.pass2_kernel.as_ref().expect("SdpaVector2Pass requires pass2_kernel");
            let p2_name = p2k.to_string();
            (quote! { &[] }, quote! {
                crate::spec::BenchDispatch::SdpaVector2Pass {
                    head_dim: #hd as usize,
                    n_kv: #nkv as usize,
                    n_q_heads: #nh as usize,
                    gqa_factor: #gqa as usize,
                    batch: #batch as usize,
                    blocks: #blocks_val as usize,
                    pass2_kernel_name: #p2_name,
                    pass2_kernel_ir: #p2k::kernel_ir_for,
                }
            })
        },
        // ── Complex: SdpaBatchedDecode ──────────────────────────────────
        ClassKind::SdpaBatchedDecode => {
            let hd = a.h.as_ref().expect("SdpaBatchedDecode requires h (head_dim)");
            let nkv = a.n_kv.as_ref().expect("SdpaBatchedDecode requires n_kv");
            let nh = a.n_heads.as_ref().expect("SdpaBatchedDecode requires n_heads (Q heads)");
            let gqa = a.gqa_factor.as_ref().expect("SdpaBatchedDecode requires gqa_factor");
            let bq = a.batch_q.as_ref().expect("SdpaBatchedDecode requires batch_q");
            let tpg_val = a.tpg.as_ref().expect("SdpaBatchedDecode requires tpg");
            let variant_dispatch =
                match a.variant.as_ref().expect("SdpaBatchedDecode requires variant") {
                    BatchedDecodeVariantArg::Decode => {
                        quote! { crate::spec::BatchedDecodeVariant::Decode }
                    },
                    BatchedDecodeVariantArg::PrefillTile => {
                        let bq_tile = a.bq.as_ref().expect("PrefillTile requires bq");
                        let bk_tile = a.bk.as_ref().expect("PrefillTile requires bk");
                        let wm_tile = a.wm.as_ref().expect("PrefillTile requires wm");
                        let wn_tile = a.wn.as_ref().expect("PrefillTile requires wn");
                        quote! {
                            crate::spec::BatchedDecodeVariant::PrefillTile {
                                bq: #bq_tile as usize,
                                bk: #bk_tile as usize,
                                wm: #wm_tile as usize,
                                wn: #wn_tile as usize,
                            }
                        }
                    },
                };
            (quote! { &[] }, quote! {
                crate::spec::BenchDispatch::SdpaBatchedDecode {
                    head_dim: #hd as usize,
                    n_kv: #nkv as usize,
                    n_q_heads: #nh as usize,
                    gqa_factor: #gqa as usize,
                    batch_q: #bq as usize,
                    variant: #variant_dispatch,
                    tpg: #tpg_val as usize,
                }
            })
        },
        // ── Complex: SteelGemm ────────────────────────────────
        ClassKind::SteelGemm => {
            let bm_val = a.bm.as_ref().expect("SteelGemm requires bm");
            let bn_val = a.bn.as_ref().expect("SteelGemm requires bn");
            let tpg_val = a.tpg.as_ref().expect("SteelGemm requires tpg");
            (quote! { &[] }, quote! {
                crate::spec::BenchDispatch::SteelGemm {
                    m: 4096usize,
                    n: 4096usize,
                    k: 4096usize,
                    check_m: #bm_val as usize,
                    check_n: #bn_val as usize,
                    check_k: 16usize,
                    bm: #bm_val as usize,
                    bn: #bn_val as usize,
                    tpg: #tpg_val as usize,
                }
            })
        },
    };

    let km_ts = kernel_mode_ts(&a.kernel_mode);

    quote! {
        const _: () = {
            #[allow(unused_imports)]
            use crate::bench_types::DType;
            ::inventory::submit! {
                crate::spec::BenchSpec {
                    op:          #op,
                    subop:       #subop,
                    kernel_name: #fn_str,
                    kernel_ir:   #kernel_ir_expr,
                    dtypes:      #dtypes,
                    tol:         #tol as f32,
                    mlx_src:     #mlx_src,
                    mlx_pattern: #mlx_pat,
                    shapes:      #shapes_ts,
                    dispatch:    #dispatch_ts,
                    kernel_mode: #km_ts,
                }
            }
        };
    }
}
