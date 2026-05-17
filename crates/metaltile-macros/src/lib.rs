//! MetalTile proc macros: `#[kernel]`, `shape!`, `tile!`, `#[autotune]`.
//!
//! These macros parse user-written Rust functions and transform them
//! into MetalTile IR and host-side launch code.

mod body_parser;

use std::collections::BTreeSet;

use body_parser::DslBodyParser;
use darling::FromMeta;
use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemFn, parse_macro_input};

// ---------------------------------------------------------------------------
// #[kernel] — the main macro
// ---------------------------------------------------------------------------

/// Marks a function as a MetalTile kernel.
///
/// The function body uses the MetalTile DSL (load, store, dot, etc.) and
/// is parsed into IR at compile time. A host-side `launch` method is
/// also generated.
///
/// # Attributes
///
/// - `#[autotune(configs = [...], key = [M, N, K])]` — enable autotuning
///   with the given configs and bucketing keys.
///
/// # Example
///
/// ```ignore
/// #[kernel]
/// pub fn vector_add(
///     a: Tensor<f16>,
///     b: Tensor<f16>,
///     c: Tensor<f16>,
/// ) {
///     let idx = program_id::<0>();
///     let x = load(a[idx]);
///     let y = load(b[idx]);
///     store(c[idx], x + y);
/// }
/// ```
#[proc_macro_attribute]
pub fn constexpr(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Pass-through: just marks a parameter as a constexpr for #[kernel] to detect
    item
}

#[proc_macro_attribute]
pub fn scalar(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Pass-through: marks a Tensor param as a scalar (constant T& in MSL)
    item
}

#[proc_macro_attribute]
pub fn strided(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Pass-through: marks a Tensor param as strided (emits shape/strides arrays)
    item
}

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    expand_kernel(input_fn)
}

fn expand_kernel(input_fn: ItemFn) -> TokenStream {
    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    let vis = &input_fn.vis;

    // Extract type parameters from generics: <T>, <T, U>, etc.
    // Each type param gets a corresponding DType arg variable: T→_t, U→_u, V→_v, W→_w.
    let type_param_names: Vec<String> = input_fn
        .sig
        .generics
        .params
        .iter()
        .filter_map(|p| {
            if let syn::GenericParam::Type(tp) = p { Some(tp.ident.to_string()) } else { None }
        })
        .collect();
    let arg_var_names: Vec<String> = type_param_names
        .iter()
        .enumerate()
        .map(|(i, _)| format!("_{}", ['t', 'u', 'v', 'w'].get(i).copied().unwrap_or('x')))
        .collect();
    // Map from type-param name ("T") to the DType arg ident token (_t).
    let type_var_map: std::collections::HashMap<String, proc_macro2::TokenStream> =
        type_param_names
            .iter()
            .zip(arg_var_names.iter())
            .map(|(tp, av)| {
                let ident = syn::Ident::new(av, proc_macro2::Span::call_site());
                (tp.clone(), quote! { #ident })
            })
            .collect();

    // Parse function signature for tensor parameters and constexprs
    let param_decls = parse_kernel_params_generic(&input_fn.sig, &type_var_map);
    let param_names: Vec<String> = extract_param_names(&input_fn.sig);
    let constexpr_info = extract_constexprs_typed(&input_fn.sig);
    let constexpr_names: Vec<String> = constexpr_info.iter().map(|(n, _)| n.clone()).collect();

    // Parse the DSL body into IR-building token stream
    let body_ir = DslBodyParser::parse_with_type_vars(
        &input_fn.block,
        &param_names,
        &constexpr_names,
        &type_var_map,
    );

    let constexpr_idents: Vec<_> = constexpr_names
        .iter()
        .map(|n| syn::Ident::new(n, proc_macro2::Span::call_site()))
        .collect();
    let constexpr_dtypes: Vec<proc_macro2::TokenStream> =
        constexpr_info.iter().map(|(_, d)| d.clone()).collect();

    // Build kernel_ir_for signature and kernel_ir() default call.
    // For non-generic kernels, kernel_ir_for takes no args (same as today).
    let arg_var_idents: Vec<_> =
        arg_var_names.iter().map(|n| syn::Ident::new(n, proc_macro2::Span::call_site())).collect();
    let kernel_ir_for_sig = quote! { pub fn kernel_ir_for(#(#arg_var_idents: DType),*) -> Kernel };
    // kernel_ir() calls kernel_ir_for with F32 defaults for each type param.
    let f32_defaults = arg_var_idents.iter().map(|_| quote! { DType::F32 });

    // Generate the expanded output: both the IR constant and the launch builder.
    let expanded = quote! {
        #vis mod #fn_name {
            use super::*;
            use metaltile_core::ir::{Kernel, Block, Op, ValueId, BlockId, VarId, Param, ParamKind, TypedSlot, ConstExprDecl, BinOpKind, ReduceKind, AttnParams, IndexExpr, UnaryOpKind, ActKind};
            use metaltile_core::shape::{Shape, Dim};
            use metaltile_core::dtype::DType;
            use metaltile_core::constexpr::ConstExpr;

            /// Build the kernel IR for specific dtype(s).
            /// For non-generic kernels this takes no arguments.
            /// For generic kernels (e.g. `fn foo<T>`) call `kernel_ir_for(DType::F16)`.
            #kernel_ir_for_sig {
                let mut kernel = Kernel::new(#fn_name_str);

                // Constexpr declarations
                #(
                    kernel.constexprs.push(ConstExprDecl {
                        name: ConstExpr::new(stringify!(#constexpr_idents)),
                        dtype: #constexpr_dtypes,
                        value: None,
                    });
                )*

                // Tensor parameters parsed from the signature
                #param_decls

                // DSL body translated to IR ops
                #body_ir

                kernel
            }

            /// The kernel IR, defaulting all type params to f32.
            pub fn kernel_ir() -> Kernel {
                kernel_ir_for(#(#f32_defaults),*)
            }

            /// Host-side launch builder.
            /// Accepts a context and named input buffers.
            pub struct LaunchBuilder<'a> {
                ctx: &'a metaltile_runtime::Context,
                /// Named input buffers.
                buffers: std::collections::BTreeMap<String, Vec<u8>>,
            }

            impl<'a> LaunchBuilder<'a> {
                pub fn new(ctx: &'a metaltile_runtime::Context) -> Self {
                    LaunchBuilder {
                        ctx,
                        buffers: std::collections::BTreeMap::new(),
                    }
                }

                /// Bind a named input buffer.
                pub fn input(mut self, name: &str, data: Vec<u8>) -> Self {
                    self.buffers.insert(name.to_string(), data);
                    self
                }

                /// Dispatch the kernel.
                pub fn dispatch(self) -> std::result::Result<metaltile_runtime::DispatchResult, metaltile_runtime::MetalTileError> {
                    let kernel = kernel_ir();
                    self.ctx.dispatch_with_buffers(&kernel, &self.buffers)
                }
            }

            /// Launch method on the module.
            pub fn launch(ctx: &metaltile_runtime::Context) -> LaunchBuilder<'_> {
                LaunchBuilder::new(ctx)
            }
        }
    };

    TokenStream::from(expanded)
}

/// Parse tensor parameters from function signature into IR param declarations.
/// `type_vars` maps type-param names (e.g. "T") to their DType arg-variable tokens (e.g. `_t`).
fn parse_kernel_params_generic(
    sig: &syn::Signature,
    type_vars: &std::collections::HashMap<String, proc_macro2::TokenStream>,
) -> proc_macro2::TokenStream {
    let mut param_builders = Vec::new();
    let use_explicit_outputs = sig.inputs.iter().any(has_mutable_tensor_param);

    for input in &sig.inputs {
        if let syn::FnArg::Typed(pat_type) = input
            && let syn::Pat::Ident(pat_ident) = &*pat_type.pat
        {
            let param_name = pat_ident.ident.to_string();
            let ty = &pat_type.ty;

            if !is_tensor_type(ty) {
                continue;
            }

            let is_output = if use_explicit_outputs {
                pat_ident.mutability.is_some()
            } else {
                is_legacy_output_name(&param_name)
            };
            let (dtype, shape, _shape_ces) = parse_tensor_type_generic(ty, type_vars);

            let kind = if has_attr(pat_type, "scalar") {
                quote! { ParamKind::Scalar }
            } else if has_attr(pat_type, "strided") {
                quote! { ParamKind::Strided }
            } else {
                quote! { Default::default() }
            };

            param_builders.push(quote! {
                kernel.params.push(Param {
                    name: #param_name.to_string(),
                    dtype: #dtype,
                    shape: #shape,
                    is_output: #is_output,
                    kind: #kind,
                });
            });
        }
    }

    if param_builders.is_empty() {
        quote! {}
    } else {
        quote! { #(#param_builders)* }
    }
}

fn has_mutable_tensor_param(input: &syn::FnArg) -> bool {
    if let syn::FnArg::Typed(pat_type) = input {
        return matches!(&*pat_type.pat, syn::Pat::Ident(pat_ident) if pat_ident.mutability.is_some())
            && is_tensor_type(&pat_type.ty);
    }
    false
}

fn is_legacy_output_name(name: &str) -> bool { matches!(name, "out" | "c" | "output") }

/// Extract parameter names from the signature.
fn extract_param_names(sig: &syn::Signature) -> Vec<String> {
    let mut names = Vec::new();
    for input in &sig.inputs {
        if let syn::FnArg::Typed(pat_type) = input
            && let syn::Pat::Ident(pat_ident) = &*pat_type.pat
        {
            names.push(pat_ident.ident.to_string());
        }
    }
    names
}

/// Check if a typed parameter has a given attribute by name.
fn has_attr(pat_type: &syn::PatType, attr_name: &str) -> bool {
    pat_type.attrs.iter().any(|a| a.path().is_ident(attr_name))
}

/// Check if a type looks like a Tensor (contains "Tensor" in its path).
fn is_tensor_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        type_path.path.segments.iter().any(|seg| seg.ident == "Tensor")
    } else {
        false
    }
}

/// Returns (dtype_tokens, shape_tokens, constexpr_names_from_shape).
/// `type_vars` maps type-param names to their runtime DType arg tokens.
fn parse_tensor_type_generic(
    ty: &syn::Type,
    type_vars: &std::collections::HashMap<String, proc_macro2::TokenStream>,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream, Vec<String>) {
    let mut dtype_tokens = quote! { DType::F32 };
    let mut shape_tokens = quote! { Shape::scalar() };
    let mut shape_ces = Vec::new();

    if let syn::Type::Path(type_path) = ty {
        for seg in &type_path.path.segments {
            if seg.ident == "Tensor"
                && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
            {
                let mut iter = args.args.iter();
                if let Some(syn::GenericArgument::Type(dtype_ty)) = iter.next() {
                    dtype_tokens = parse_dtype_generic(dtype_ty, type_vars);
                }
                if let Some(arg) = iter.next() {
                    let (tokens, ces) = parse_shape_arg(arg);
                    shape_tokens = tokens;
                    shape_ces = ces;
                }
            }
        }
    }
    (dtype_tokens, shape_tokens, shape_ces)
}

/// Backwards-compat wrapper with empty type_vars map.
fn parse_tensor_type(
    ty: &syn::Type,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream, Vec<String>) {
    parse_tensor_type_generic(ty, &std::collections::HashMap::new())
}

/// Returns (shape_tokens, constexpr_names_from_dims).
fn parse_shape_arg(arg: &syn::GenericArgument) -> (proc_macro2::TokenStream, Vec<String>) {
    let str = quote! { #arg }.to_string();
    let inner = str.trim();
    let mut ces = Vec::new();

    if let Some(start) = inner.find('(')
        && let Some(end) = inner.rfind(')')
    {
        let content = &inner[start + 1..end];
        let dims: Vec<_> =
            content.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();

        if !dims.is_empty() {
            let dim_tokens: Vec<_> = dims
                .iter()
                .map(|d| {
                    if let Ok(n) = d.parse::<usize>() {
                        quote! { Dim::Known(#n) }
                    } else {
                        ces.push(d.clone());
                        let ident = syn::Ident::new(d, proc_macro2::Span::call_site());
                        quote! { Dim::ConstExpr(ConstExpr::new(stringify!(#ident))) }
                    }
                })
                .collect();

            return (
                quote! {
                    { use metaltile_core::shape::Shape; use metaltile_core::shape::Dim; use metaltile_core::constexpr::ConstExpr;
                    Shape::new([#(#dim_tokens),*]) }
                },
                ces,
            );
        }
    }
    (quote! { Shape::scalar() }, ces)
}

fn parse_dtype_generic(
    ty: &syn::Type,
    type_vars: &std::collections::HashMap<String, proc_macro2::TokenStream>,
) -> proc_macro2::TokenStream {
    if let syn::Type::Path(type_path) = ty {
        let ident = &type_path.path.segments.last().unwrap().ident;
        let name = ident.to_string();
        // If this ident is a known type parameter (T, U, ...), emit the runtime arg variable.
        if let Some(arg_tok) = type_vars.get(&name) {
            return arg_tok.clone();
        }
        return match name.as_str() {
            "f32" => quote! { DType::F32 },
            "f16" => quote! { DType::F16 },
            "bf16" => quote! { DType::BF16 },
            "i32" => quote! { DType::I32 },
            "u32" => quote! { DType::U32 },
            "i8" => quote! { DType::I8 },
            "bool" => quote! { DType::Bool },
            _ => quote! { DType::F32 },
        };
    }
    quote! { DType::F32 }
}

/// Extract constexpr names from `#[constexpr]` params and tensor shape dims.
#[cfg(test)]
fn extract_constexprs(sig: &syn::Signature) -> Vec<String> {
    extract_constexprs_typed(sig).into_iter().map(|(n, _)| n).collect()
}

/// Extract constexpr names with their DType tokens from `#[constexpr]` params and tensor shape dims.
fn extract_constexprs_typed(sig: &syn::Signature) -> Vec<(String, proc_macro2::TokenStream)> {
    let mut entries: Vec<(String, proc_macro2::TokenStream)> = Vec::new();
    let mut seen = BTreeSet::new();

    for input in &sig.inputs {
        if let syn::FnArg::Typed(pat_type) = input {
            if pat_type.attrs.iter().any(|a| a.path().is_ident("constexpr"))
                && let syn::Pat::Ident(pat_ident) = &*pat_type.pat
            {
                let name = pat_ident.ident.to_string();
                let dtype = rust_type_to_dtype_tokens(&pat_type.ty);
                push_unique_typed(&mut entries, &mut seen, name, dtype);
            }

            if is_tensor_type(&pat_type.ty) {
                let (_, _, shape_ces) = parse_tensor_type(&pat_type.ty);
                for ce_name in shape_ces {
                    push_unique_typed(&mut entries, &mut seen, ce_name, quote! { DType::U32 });
                }
            }
        }
    }

    entries
}

/// Map a Rust scalar type path to a `DType::*` token stream.
fn rust_type_to_dtype_tokens(ty: &syn::Type) -> proc_macro2::TokenStream {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last()
    {
        return match seg.ident.to_string().as_str() {
            "f32" => quote! { DType::F32 },
            "f16" | "half" => quote! { DType::F16 },
            "f64" => quote! { DType::F64 },
            "i32" => quote! { DType::I32 },
            "i64" => quote! { DType::I64 },
            "u64" => quote! { DType::U64 },
            _ => quote! { DType::U32 },
        };
    }
    quote! { DType::U32 }
}

fn push_unique_typed(
    entries: &mut Vec<(String, proc_macro2::TokenStream)>,
    seen: &mut BTreeSet<String>,
    name: String,
    dtype: proc_macro2::TokenStream,
) {
    if seen.insert(name.clone()) {
        entries.push((name, dtype));
    }
}

// ---------------------------------------------------------------------------
// shape! macro
// ---------------------------------------------------------------------------

/// Construct a [`Shape`] from dimension expressions.
///
/// ```ignore
/// shape!(M, K)       // 2D shape with constexpr dims M and K
/// shape!(32, 64)     // 2D shape with known dims
/// shape!()           // scalar
/// ```
#[proc_macro]
pub fn shape(input: TokenStream) -> TokenStream {
    let dims: Vec<_> = input
        .to_string()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let dim_exprs: Vec<_> = dims
        .iter()
        .map(|d| {
            if let Ok(n) = d.parse::<usize>() {
                quote! { Dim::Known(#n) }
            } else {
                let ident = syn::Ident::new(d, proc_macro2::Span::call_site());
                quote! { Dim::ConstExpr(ConstExpr::new(stringify!(#ident))) }
            }
        })
        .collect();

    let expanded = quote! {
        {
            use metaltile_core::shape::Shape;
            use metaltile_core::shape::Dim;
            use metaltile_core::constexpr::ConstExpr;
            Shape::new([#(#dim_exprs),*])
        }
    };

    TokenStream::from(expanded)
}

// ---------------------------------------------------------------------------
// tile! macro
// ---------------------------------------------------------------------------

/// Construct a 2D tile shape.
///
/// ```ignore
/// tile!(TILE_M, TILE_N)  // 2D tile of constexpr dimensions
/// tile!(32, 64)           // 2D tile of known dimensions
/// ```
#[proc_macro]
pub fn tile(input: TokenStream) -> TokenStream {
    let parts: Vec<_> = input.to_string().split(',').map(|s| s.trim().to_string()).collect();

    let rows = parse_dim_expr(&parts[0]);
    let cols = parse_dim_expr(parts.get(1).map_or("1", |s| s.as_str()));

    let expanded = quote! {
        {
            use metaltile_core::shape::tile;
            use metaltile_core::shape::Dim;
            use metaltile_core::constexpr::ConstExpr;
            tile(#rows, #cols)
        }
    };

    TokenStream::from(expanded)
}

fn parse_dim_expr(s: &str) -> proc_macro2::TokenStream {
    if let Ok(n) = s.parse::<usize>() {
        quote! { Dim::Known(#n) }
    } else {
        let ident = syn::Ident::new(s, proc_macro2::Span::call_site());
        quote! { Dim::ConstExpr(ConstExpr::new(stringify!(#ident))) }
    }
}

// ---------------------------------------------------------------------------
// #[autotune] attribute — parsed by #[kernel]
// ---------------------------------------------------------------------------

#[derive(Debug, FromMeta)]
#[allow(dead_code)]
struct AutotuneArgs {
    configs: Option<String>,
    key: Option<String>,
}

#[cfg(test)]
mod tests {
    use syn::parse_quote;

    use super::*;

    #[test]
    fn mutable_tensor_outputs_override_legacy_name_heuristics() {
        let item: ItemFn = parse_quote! {
            fn kernel(a: Tensor<f32>, mut result: Tensor<f32>, c: Tensor<f32>) {}
        };

        let tokens =
            parse_kernel_params_generic(&item.sig, &std::collections::HashMap::new()).to_string();

        assert_param_output(&tokens, "a", false);
        assert_param_output(&tokens, "result", true);
        assert_param_output(&tokens, "c", false);
    }

    #[test]
    fn legacy_output_names_still_work_without_mutable_tensor_params() {
        let item: ItemFn = parse_quote! {
            fn kernel(a: Tensor<f32>, c: Tensor<f32>, output: Tensor<f32>) {}
        };

        let tokens =
            parse_kernel_params_generic(&item.sig, &std::collections::HashMap::new()).to_string();

        assert_param_output(&tokens, "a", false);
        assert_param_output(&tokens, "c", true);
        assert_param_output(&tokens, "output", true);
    }

    #[test]
    fn extract_constexprs_deduplicates_shape_dims() {
        let item: ItemFn = parse_quote! {
            fn kernel(
                a: Tensor<f32, shape!(M, N)>,
                b: Tensor<f32, shape!(M, N)>,
                #[constexpr] K: u32,
                out: Tensor<f32, shape!(K, N)>,
            ) {}
        };

        assert_eq!(extract_constexprs(&item.sig), vec!["M", "N", "K"]);
    }

    fn assert_param_output(tokens: &str, name: &str, expected: bool) {
        let needle = format!(
            "name : \"{name}\" . to_string () , dtype : DType :: F32 , shape : Shape :: scalar () , is_output : {expected}"
        );
        assert!(tokens.contains(&needle), "missing `{needle}` in `{tokens}`");
    }
}

// ---------------------------------------------------------------------------
// #[bench_kernel] — declarative benchmark registration
// ---------------------------------------------------------------------------

/// Registers a `#[kernel]` function for automatic benchmarking.
///
/// Must be placed **before** `#[kernel]` so it sees the original function
/// signature. Generates an `inventory::submit! { BenchSpec { ... } }` alongside
/// the kernel, which the bench suite collects via `inventory::iter::<BenchSpec>`.
///
/// # Required args
/// - `op    = "group"` — bench table group (e.g. `"unary"`)
/// - `subop = "name"`  — sub-operation label (e.g. `"exp"`)
/// - `class = Unary | Binary | AllReduce | RowReduce`
/// - `cpu   = fn_ptr`  — CPU reference (named fn, not closure)
/// - `tol   = 1e-4`    — maximum absolute correctness error
///
/// # Optional args
/// - `input = Signed|Positive|Half|Unit` (Unary default: `Half`)
/// - `input_a / input_b` (Binary, default: `Half`)
/// - `metal_file = "foo.metal"` — MLX reference source file (loaded via `include_str!` at compile time)
/// - `mlx = "pattern"` — kernel name pattern; `{tn}` → MLX type name
/// - `dtypes = IDENT`  — `&'static [DType]` (default: `FLOAT_DTYPES`)
///
/// # Example
/// ```ignore
/// fn cpu_exp(x: f32) -> f32 { x.exp() }
///
/// #[bench_kernel(op="unary", subop="exp", class=Unary, cpu=cpu_exp,
///                input=Signed, tol=1e-4, metal_file="unary.metal", mlx="v_Exp{tn}{tn}")]
/// #[kernel]
/// pub fn mt_exp<T>(a: Tensor<T>, out: Tensor<T>) { … }
/// ```
#[proc_macro_attribute]
pub fn bench_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    use bench_impl::{BenchArgs, generate_submit};

    let args = match syn::parse::<BenchArgs>(attr) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };

    let (fn_name, is_generic) = {
        let f = match syn::parse::<syn::ItemFn>(item.clone()) {
            Ok(f) => f,
            Err(e) => return e.to_compile_error().into(),
        };
        let generic = !f.sig.generics.params.is_empty();
        (f.sig.ident.clone(), generic)
    };

    let submit = generate_submit(&fn_name, &args, is_generic);
    let item_ts: proc_macro2::TokenStream = item.into();
    quote! { #item_ts  #submit }.into()
}

mod bench_impl {
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
        Rope,
        Attention,
        StridedCopy,
    }

    pub enum InputKind {
        Signed,
        Positive,
        Half,
        Unit,
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
                            "Rope" => ClassKind::Rope,
                            "Attention" => ClassKind::Attention,
                            "StridedCopy" => ClassKind::StridedCopy,
                            o =>
                                return Err(syn::Error::new(
                                    id.span(),
                                    format!("unknown class `{o}`"),
                                )),
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
                    o =>
                        return Err(syn::Error::new(
                            key.span(),
                            format!("unknown bench_kernel arg: `{o}`"),
                        )),
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
            None => quote! {crate::ops::FLOAT_DTYPES},
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
                        crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::AltZeroOne, dtype_override: None },
                        crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::Half, dtype_override: None },
                        crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::Half, dtype_override: None },
                        crate::spec::TensorBufSpec { count: crate::spec::Dim::N, init: crate::spec::BufInit::Zeros, dtype_override: None },
                    ],
                    scalar_bufs: &[],
                    cexprs: &[],
                    out_elems: crate::spec::Dim::N,
                    reads: 3usize,
                    bytes_fn: crate::spec::bytes_elementwise,
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
                    check_n: crate::spec::ROW_REDUCE_CHECK_N, check_b: 4usize,
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
                    mlx_tpg: 0usize,
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
                (quote! { &[] }, quote! {
                    crate::spec::BenchDispatch::QuantizedMatVec {
                        shapes: #sh, group_size: #gs as usize, tpg: #tpg_val as usize,
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
        };

        quote! {
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
                    kernel_mode: None,
                }
            }
        }
    }
}
