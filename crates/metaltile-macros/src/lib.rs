//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! MetalTile proc macros: `#[kernel]`, `#[kernel(bench(...))]`, `shape!`, `tile!`.
//!
//! These macros parse user-written Rust functions and transform them
//! into MetalTile IR and host-side launch code.

mod bench_impl;
mod body_parser;
mod derive_op;
mod sig_parser;

use std::collections::HashMap;

use bench_impl::{BenchArgs, generate_submit};
use body_parser::DslBodyParser;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use sig_parser::{extract_constexprs_typed, extract_param_names, parse_kernel_params_generic};
use syn::{ItemFn, parse_macro_input};

// ---------------------------------------------------------------------------
// Attribute parsing — unified #[kernel(bench(...))]
// ---------------------------------------------------------------------------

/// Arguments parsed from the `#[kernel(...)]` attribute.
///
/// Supports two forms:
/// - `#[kernel]` — no bench args, just kernel expansion
/// - `#[kernel(bench(op="...", subop="...", ...))]` — kernel expansion + BenchSpec submission
struct KernelAttr {
    bench: Option<BenchArgs>,
}

impl syn::parse::Parse for KernelAttr {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(KernelAttr { bench: None });
        }

        let ident: syn::Ident = input.parse()?;
        if ident != "bench" {
            return Err(syn::Error::new(
                ident.span(),
                "expected `bench(...)` as the only #[kernel] argument",
            ));
        }

        let content;
        syn::parenthesized!(content in input);
        let bench_args = content.parse::<BenchArgs>()?;

        if !input.is_empty() {
            return Err(syn::Error::new(input.span(), "unexpected tokens after `bench(...)`"));
        }

        Ok(KernelAttr { bench: Some(bench_args) })
    }
}

// ---------------------------------------------------------------------------
// KernelMacroBuilder — object-oriented expansion orchestrator
// ---------------------------------------------------------------------------

/// Builder that orchestrates the full `#[kernel]` macro expansion.
///
/// Owns the parsed function and optional bench configuration, then
/// generates all output token streams: the kernel-IR module, launch
/// builder, inventory registration, and (optionally) the BenchSpec
/// submission for `tile bench`.
struct KernelMacroBuilder {
    /// The parsed kernel function.
    input_fn: ItemFn,
    /// Optional bench args for automatic benchmark registration.
    bench_args: Option<BenchArgs>,
}

impl KernelMacroBuilder {
    /// Create a new builder from the parsed function and optional bench args.
    fn new(input_fn: ItemFn, bench_args: Option<BenchArgs>) -> Self {
        KernelMacroBuilder { input_fn, bench_args }
    }

    /// Run the full expansion pipeline and return the generated token stream.
    fn expand(self) -> TokenStream2 {
        let fn_name = &self.input_fn.sig.ident;
        let fn_name_str = fn_name.to_string();
        let vis = &self.input_fn.vis;
        let is_generic = !self.input_fn.sig.generics.params.is_empty();

        // ── 1. Extract type-param → DType-arg mapping ──────────────────
        let type_param_names = self.extract_type_param_names();
        let arg_var_names = self.build_arg_var_names(&type_param_names);
        let type_var_map = self.build_type_var_map(&type_param_names, &arg_var_names);

        // ── 2. Parse signature: tensor params + constexprs ─────────────
        let param_decls = parse_kernel_params_generic(&self.input_fn.sig, &type_var_map);
        let constexpr_info = extract_constexprs_typed(&self.input_fn.sig);
        let constexpr_names: Vec<String> = constexpr_info.iter().map(|(n, _)| n.clone()).collect();
        let param_names = extract_param_names(&self.input_fn.sig);

        // ── 3. Parse the DSL body into IR-building token streams ──────
        let body_ir = DslBodyParser::parse_with_type_vars(
            &self.input_fn.block,
            &param_names,
            &constexpr_names,
            &type_var_map,
        );

        // ── 4. Build kernel_ir_for / kernel_ir signatures ──────────────
        let constexpr_idents: Vec<_> = constexpr_names
            .iter()
            .map(|n| syn::Ident::new(n, proc_macro2::Span::call_site()))
            .collect();
        let constexpr_dtypes: Vec<TokenStream2> =
            constexpr_info.iter().map(|(_, d)| d.clone()).collect();

        let arg_var_idents: Vec<_> = arg_var_names
            .iter()
            .map(|n| syn::Ident::new(n, proc_macro2::Span::call_site()))
            .collect();
        let kernel_ir_for_sig = quote! {
            pub fn kernel_ir_for(#(#arg_var_idents: DType),*) -> Kernel
        };
        let f32_defaults: Vec<_> = arg_var_idents.iter().map(|_| quote! { DType::F32 }).collect();

        // ── 5. Generate the kernel module ──────────────────────────────
        let kernel_module = self.generate_kernel_module(
            fn_name,
            &fn_name_str,
            vis,
            &arg_var_idents,
            &kernel_ir_for_sig,
            &f32_defaults,
            &constexpr_idents,
            &constexpr_dtypes,
            &param_decls,
            &body_ir,
        );

        // ── 6. Optionally generate the BenchSpec submission ───────────
        let bench_submit = match &self.bench_args {
            Some(args) => generate_submit(fn_name, args, is_generic),
            None => TokenStream2::new(),
        };

        quote! {
            #kernel_module
            #bench_submit
        }
    }

    /// Extract the list of type parameter names from generics (e.g. `["T", "U"]`).
    fn extract_type_param_names(&self) -> Vec<String> {
        self.input_fn
            .sig
            .generics
            .params
            .iter()
            .filter_map(|p| {
                if let syn::GenericParam::Type(tp) = p { Some(tp.ident.to_string()) } else { None }
            })
            .collect()
    }

    /// Map each type param to its DType arg variable name: `T → _t`, `U → _u`, etc.
    fn build_arg_var_names(&self, type_param_names: &[String]) -> Vec<String> {
        type_param_names
            .iter()
            .enumerate()
            .map(|(i, _)| format!("_{}", ['t', 'u', 'v', 'w'].get(i).copied().unwrap_or('x')))
            .collect()
    }

    /// Build a name→TokenStream map from type-param names to their DType arg idents.
    fn build_type_var_map(
        &self,
        type_param_names: &[String],
        arg_var_names: &[String],
    ) -> HashMap<String, TokenStream2> {
        type_param_names
            .iter()
            .zip(arg_var_names.iter())
            .map(|(tp, av)| {
                let ident = syn::Ident::new(av, proc_macro2::Span::call_site());
                (tp.clone(), quote! { #ident })
            })
            .collect()
    }

    /// Generate the kernel module containing IR, launch builder, and inventory entry.
    #[allow(clippy::too_many_arguments)]
    fn generate_kernel_module(
        &self,
        fn_name: &syn::Ident,
        fn_name_str: &str,
        vis: &syn::Visibility,
        arg_var_idents: &[syn::Ident],
        kernel_ir_for_sig: &TokenStream2,
        f32_defaults: &[TokenStream2],
        constexpr_idents: &[syn::Ident],
        constexpr_dtypes: &[TokenStream2],
        param_decls: &TokenStream2,
        body_ir: &TokenStream2,
    ) -> TokenStream2 {
        quote! {
            #vis mod #fn_name {
                use super::*;
                use metaltile_core::ir::{
                    Kernel, Block, Op, ValueId, BlockId, VarId, Param, ParamKind,
                    TypedSlot, ConstExprDecl, BinOpKind, ReduceKind, AtomicKind,
                    AtomicScope, IndexExpr, UnaryOpKind, ActKind, KernelCallArg,
                    CoopTileAccMode, CoopTileScope,
                };
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
                    pub fn dispatch(
                        self,
                    ) -> std::result::Result<
                        metaltile_runtime::DispatchResult,
                        metaltile_runtime::MetalTileError,
                    > {
                        let kernel = kernel_ir();
                        self.ctx.dispatch_with_buffers(&kernel, &self.buffers)
                    }
                }

                /// Launch method on the module.
                pub fn launch(ctx: &metaltile_runtime::Context) -> LaunchBuilder<'_> {
                    LaunchBuilder::new(ctx)
                }

                // Use `const _: ()` hygiene scope so `__build_for_inline` does not
                // leak into the enclosing module's namespace.
                const _: () = {
                    fn __build_for_inline(
                        dtypes: &[metaltile_core::dtype::DType],
                    ) -> metaltile_core::ir::Kernel {
                        #[allow(unused_variables)]
                        let _t = dtypes
                            .first()
                            .copied()
                            .unwrap_or(metaltile_core::dtype::DType::F32);
                        kernel_ir_for(#(#arg_var_idents),*)
                    }

                    metaltile_core::inventory::submit! {
                        metaltile_core::KernelEntry::new(#fn_name_str, __build_for_inline)
                    }
                };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Proc-macro derive helpers (unchanged from upstream)
// ---------------------------------------------------------------------------

/// Derive `Op::value_refs()` and `Op::for_each_value_id_mut()`.
///
/// Annotate `ValueId` fields with `#[vid]`, `#[vid_opt]`, `#[vid_vec]`,
/// `#[vid_exprs]`, or `#[vid_recursive]`. See `derive_op` module docs for details.
#[proc_macro_derive(ValueRefs, attributes(vid, vid_opt, vid_vec, vid_exprs, vid_recursive))]
pub fn derive_value_refs(input: TokenStream) -> TokenStream { derive_op::derive_value_refs(input) }

/// Derive op-flag predicates (`is_elementwise`, `has_side_effects`, etc.).
///
/// Annotate variants with `#[elementwise]`, `#[side_effect]`, `#[unpredictable]`,
/// `#[cheap_alu]`, or `#[op_load]`. See `derive_op` module docs for details.
#[proc_macro_derive(
    OpFlags,
    attributes(
        elementwise,
        side_effect,
        unpredictable,
        cheap_alu,
        op_load,
        op_store,
        barrier,
        op_loop,
        op_if,
        op_fused,
        op_const,
        shape_op,
        needs_simd_lane,
        needs_simd_group,
        needs_simdgroup_matrix,
        needs_simd_product,
        no_result,
        result_u32,
        result_i32,
        result_f32_scalar,
        result_f16_scalar,
        result_same_type,
        result_custom
    )
)]
pub fn derive_op_flags(input: TokenStream) -> TokenStream { derive_op::derive_op_flags(input) }

/// Derive `Op::variant_name()` — returns the variant identifier as a &'static str.
///
/// Supports `#[variant_name("CustomName")]` on variants that need a display name different
/// from their Rust identifier.
#[proc_macro_derive(VariantName, attributes(variant_name))]
pub fn derive_variant_name(input: TokenStream) -> TokenStream {
    derive_op::derive_variant_name(input)
}

/// Pass-through attribute that marks a parameter as a compile-time constant.
///
/// The `#[kernel]` macro detects this attribute during signature parsing.
#[proc_macro_attribute]
pub fn constexpr(_attr: TokenStream, item: TokenStream) -> TokenStream { item }

/// Pass-through attribute that marks a `Tensor` param as scalar (`constant T&` in MSL).
///
/// The `#[kernel]` macro detects this attribute during signature parsing.
#[proc_macro_attribute]
pub fn scalar(_attr: TokenStream, item: TokenStream) -> TokenStream { item }

/// Pass-through attribute that marks a `Tensor` param as strided (emits shape/strides arrays).
///
/// The `#[kernel]` macro detects this attribute during signature parsing.
#[proc_macro_attribute]
pub fn strided(_attr: TokenStream, item: TokenStream) -> TokenStream { item }

// ---------------------------------------------------------------------------
// #[kernel] — the main macro (unified)
// ---------------------------------------------------------------------------

/// Marks a function as a MetalTile kernel.
///
/// The function body uses the MetalTile DSL (load, store, dot, etc.) and
/// is parsed into IR at compile time. A host-side `launch` method is
/// also generated, and a `KernelEntry` is registered in the inventory.
///
/// # Bench registration
///
/// To automatically register the kernel for `tile bench`, pass `bench(...)`:
///
/// ```ignore
/// #[kernel(
///     bench(
///         op = "unary",
///         subop = "exp",
///         class = Unary,
///         tol = 1e-4,
///         metal_file = "unary.metal",
///         mlx = "v_Exp{tn}{tn}",
///     )
/// )]
/// pub fn mt_exp<T>(a: Tensor<T>, out: Tensor<T>) { … }
/// ```
///
/// This is the unified syntax — bench registration is built into `#[kernel]`.
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
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let kernel_attr = parse_macro_input!(attr as KernelAttr);
    let input_fn = parse_macro_input!(item as ItemFn);
    let builder = KernelMacroBuilder::new(input_fn, kernel_attr.bench);
    TokenStream::from(builder.expand())
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

fn parse_dim_expr(s: &str) -> TokenStream2 {
    if let Ok(n) = s.parse::<usize>() {
        quote! { Dim::Known(#n) }
    } else {
        let ident = syn::Ident::new(s, proc_macro2::Span::call_site());
        quote! { Dim::ConstExpr(ConstExpr::new(stringify!(#ident))) }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use syn::parse_quote;

    use super::*;
    #[cfg(test)]
    use crate::sig_parser::extract_constexprs;

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

    #[test]
    fn kernel_attr_parses_empty() {
        let attr: KernelAttr = syn::parse_quote! {};
        assert!(attr.bench.is_none());
    }

    #[test]
    fn kernel_attr_parses_bench() {
        let attr: KernelAttr = syn::parse_quote! {
            bench(op="unary", subop="exp", class=Unary, tol=1e-4)
        };
        assert!(attr.bench.is_some());
        let bench = attr.bench.unwrap();
        assert_eq!(bench.op.value(), "unary");
        assert_eq!(bench.subop.value(), "exp");
    }

    fn assert_param_output(tokens: &str, name: &str, expected: bool) {
        let needle = format!(
            "name : \"{name}\" . to_string () , dtype : DType :: F32 , shape : Shape :: scalar () , is_output : {expected}"
        );
        assert!(tokens.contains(&needle), "missing `{needle}` in `{tokens}`");
    }
}
