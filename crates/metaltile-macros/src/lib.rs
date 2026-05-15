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
        if let syn::FnArg::Typed(pat_type) = input {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
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
        if let syn::FnArg::Typed(pat_type) = input {
            if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                names.push(pat_ident.ident.to_string());
            }
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
            if seg.ident == "Tensor" {
                if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
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

    if let Some(start) = inner.find('(') {
        if let Some(end) = inner.rfind(')') {
            let content = &inner[start + 1..end];
            let dims: Vec<_> = content
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

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
            if pat_type.attrs.iter().any(|a| a.path().is_ident("constexpr")) {
                if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                    let name = pat_ident.ident.to_string();
                    let dtype = rust_type_to_dtype_tokens(&pat_type.ty);
                    push_unique_typed(&mut entries, &mut seen, name, dtype);
                }
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
    if let syn::Type::Path(tp) = ty {
        if let Some(seg) = tp.path.segments.last() {
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
