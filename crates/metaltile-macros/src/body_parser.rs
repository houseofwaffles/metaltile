//! DSL body parser: walks `syn::Expr` trees and translates DSL function calls
//! into MetalTile IR-building token streams.
//!
//! ## How it works
//!
//! The parser walks the function body statement by statement. For each:
//! - `let x = <expr>`: evaluates `<expr>`, binds result ValueId to `x`
//! - `<expr>` (bare statement, e.g. `store(...)`): evaluates and discards
//!
//! Recognized DSL calls:
//! - `program_id::<N>()`              → Op::ProgramId { axis: N }
//! - `arange::<N>()`                  → Op::Arange { len: ConstExpr(N) }
//! - `load(tensor[index])`            → Op::Load { src, indices }
//! - `store(tensor[index], val)`      → Op::Store { dst, indices, value }
//! - `zeros::<dtype, tile!(M,N)>()`   → Op::Zeros { dtype, shape }
//! - `dot(a, b)`                      → Op::Dot { a, b }
//! - `exp(x)`, `log(x)`, `sqrt(x)`, `rsqrt(x)`, `abs(x)`, `recip(x)`, ...
//! - `reduce_max(acc)`, `reduce_sum(acc)`, `reduce_min(acc)`
//! - `strided_reduce(src, off, end, max|sum)`
//! - `strided_reduce_exp_sub(src, off, end, sub_val)` -- sum(exp(x-sub))
//! - `strided_store(src, dst, off, end, scalar[, aux_src])`
//! - `x + y`, `x - y`, `x * y`, `x / y`  -- Op::BinOp
//! - `for v in range(start, end, step) { ... }` or `for v in start..end { ... }` -- Op::Loop

use std::collections::{BTreeMap, BTreeSet};

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Block,
    Expr,
    ExprAssign,
    ExprBinary,
    ExprCall,
    ExprForLoop,
    ExprIf,
    ExprIndex,
    ExprPath,
    ExprRange,
    Local,
    Pat,
    RangeLimits,
    Stmt,
};

/// State maintained while parsing a DSL body.
pub struct DslBodyParser {
    /// Next available ValueId.
    next_vid: u32,
    /// Next available BlockId.
    next_bid: u32,
    /// Next available VarId (loop variables).
    next_var: u32,
    /// Map from Rust variable names to their ValueIds.
    bindings: BTreeMap<String, u32>,
    /// Names of `let mut` variables (mutable locals backed by `__ml_*` in MSL).
    mut_locals: BTreeSet<String>,
    /// Accumulated IR-building statements (token streams).
    ir_stmts: Vec<TokenStream>,
    /// Accumulated block definitions for loops.
    blocks: Vec<(u32, TokenStream)>,
    /// Names of tensor params (for load/store).
    #[allow(dead_code)]
    param_names: Vec<String>,
    /// Names of constexpr params.
    constexpr_names: Vec<String>,
    /// Current block target: "kernel.body" in main body, "block_N" inside a loop.
    current_target: String,
    /// Map from type parameter names (e.g. "T") to their DType arg idents (e.g. `_t`).
    /// Used so `.cast::<T>()` emits `dtype: _t` instead of defaulting to F32.
    type_vars: std::collections::HashMap<String, TokenStream>,
}

impl DslBodyParser {
    /// Parse a function body and return token streams for IR construction.
    #[allow(dead_code)]
    pub fn parse(body: &Block, param_names: &[String], constexpr_names: &[String]) -> TokenStream {
        Self::parse_with_type_vars(body, param_names, constexpr_names, &Default::default())
    }

    /// Like `parse` but also maps generic type-param names → DType arg TokenStreams.
    pub fn parse_with_type_vars(
        body: &Block,
        param_names: &[String],
        constexpr_names: &[String],
        type_vars: &std::collections::HashMap<String, TokenStream>,
    ) -> TokenStream {
        let mut parser = DslBodyParser {
            next_vid: 0,
            next_bid: 1, // block 0 is the body
            next_var: 0,
            bindings: BTreeMap::new(),
            mut_locals: BTreeSet::new(),
            ir_stmts: Vec::new(),
            blocks: Vec::new(),
            param_names: param_names.to_vec(),
            constexpr_names: constexpr_names.to_vec(),
            current_target: "kernel.body".into(),
            type_vars: type_vars.clone(),
        };

        for stmt in &body.stmts {
            parser.parse_stmt(stmt);
        }

        let main_stmts = &parser.ir_stmts;
        let block_defs = &parser.blocks;

        let block_defs_tokens = block_defs.iter().map(|(bid, body_tokens)| {
            let block_var = format_ident!("block_{bid}");
            let bid_val = *bid;
            quote! {
                let mut #block_var = Block::new(BlockId::new(#bid_val));
                #body_tokens
                kernel.add_block(#block_var);
            }
        });

        quote! {
            #(#main_stmts)*
            #(#block_defs_tokens)*
        }
    }

    fn alloc_vid(&mut self) -> u32 {
        let id = self.next_vid;
        self.next_vid += 1;
        id
    }

    fn alloc_bid(&mut self) -> u32 {
        let id = self.next_bid;
        self.next_bid += 1;
        id
    }

    fn alloc_var(&mut self) -> u32 {
        let id = self.next_var;
        self.next_var += 1;
        id
    }

    fn push_const(&mut self, value: i64) -> u32 {
        let result = self.alloc_vid();
        self.push_op(quote! { Op::Const { value: #value } }, result);
        result
    }

    /// Push `<target>.push_op(<op>, ValueId::new(<result>));`
    fn push_op(&mut self, op_ts: TokenStream, result: u32) {
        let tgt: TokenStream = self.current_target.parse().expect("valid target");
        self.ir_stmts.push(quote! { #tgt.push_op(#op_ts, ValueId::new(#result)); });
    }

    /// Push `<target>.push_op_no_result(<op>);`
    fn push_op_no_result(&mut self, op_ts: TokenStream) {
        let tgt: TokenStream = self.current_target.parse().expect("valid target");
        self.ir_stmts.push(quote! { #tgt.push_op_no_result(#op_ts); });
    }

    /// Push `<target>.name_value(ValueId::new(<vid>), <name>);`
    fn push_name_value(&mut self, vid: u32, name: &str) {
        let tgt: TokenStream = self.current_target.parse().expect("valid target");
        self.ir_stmts.push(quote! { #tgt.name_value(ValueId::new(#vid), #name); });
    }

    fn push_error(&mut self, error: syn::Error) { self.ir_stmts.push(error.to_compile_error()); }

    fn push_error_value(&mut self, error: syn::Error) -> u32 {
        self.push_error(error);
        self.alloc_vid()
    }

    // ---- Statement parsing --------------------------------------------------

    fn parse_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Local(local) => self.parse_let(local),
            Stmt::Expr(expr, _semi) => self.parse_expr_stmt(expr),
            Stmt::Macro(mac) => {
                // The proc-macro does NOT expand declarative macros inside the
                // kernel body — they're seen as opaque tokens and would
                // silently produce no IR. Fail loudly so future contributors
                // can't ship a kernel with a dropped body (PR #19 shipped 25+
                // such kernels before this guard existed).
                //
                // Workarounds: (1) wrap the entire `#[kernel] fn …` declaration
                // in a `macro_rules!` so declarative expansion happens before
                // the proc-macro runs; (2) inline the macro body; (3) replace
                // an unrolled tree with a DSL `for` loop.
                self.push_error(syn::Error::new_spanned(
                    &mac.mac.path,
                    "macro_rules! invocations inside #[kernel] bodies are not \
                     expanded by the metaltile proc-macro and would silently \
                     drop their body. Wrap the entire `#[kernel] fn` in the \
                     macro instead, or inline / replace with a DSL loop.",
                ));
            },
            _ => {},
        }
    }

    fn parse_let(&mut self, local: &Local) {
        if let Pat::Ident(pat_ident) = &local.pat {
            let var_name = pat_ident.ident.to_string();
            let is_mut = pat_ident.mutability.is_some();
            if let Some(init) = &local.init {
                let init_vid = self.parse_expr(&init.expr);
                if is_mut {
                    // Mutable local: declare a C-level mutable variable via DeclareLocal.
                    // Reads go through Op::Load { src: "__ml_{name}" }.
                    let n = var_name.clone();
                    self.push_op_no_result(quote! {
                        Op::DeclareLocal {
                            name: #n.to_string(),
                            value: ValueId::new(#init_vid),
                        }
                    });
                    self.mut_locals.insert(var_name.clone());
                    // Don't bind to a ValueId; reads use GetLocal pattern via parse_path.
                } else {
                    self.bindings.insert(var_name.clone(), init_vid);
                    self.push_name_value(init_vid, &var_name);
                }
            }
        }
    }

    fn parse_expr_stmt(&mut self, expr: &Expr) {
        if let Expr::ForLoop(for_loop) = expr {
            self.parse_for_loop(for_loop);
            return;
        }
        if let Expr::If(if_expr) = expr {
            self.parse_if(if_expr);
            return;
        }
        if let Expr::Assign(assign) = expr {
            self.parse_assign(assign);
            return;
        }
        if let Expr::Block(block_expr) = expr {
            for stmt in &block_expr.block.stmts {
                self.parse_stmt(stmt);
            }
            return;
        }
        self.parse_expr(expr);
    }

    /// Handle `name = expr` assignment to a mutable local variable.
    fn parse_assign(&mut self, assign: &ExprAssign) {
        // Only handle simple path targets (variable names).
        if let Expr::Path(path) = &*assign.left {
            let name = path.path.get_ident().map(|i| i.to_string()).unwrap_or_default();
            if self.mut_locals.contains(&name) {
                let val_vid = self.parse_expr(&assign.right);
                let n = name.clone();
                self.push_op_no_result(quote! {
                    Op::SetLocal {
                        name: #n.to_string(),
                        value: ValueId::new(#val_vid),
                    }
                });
                return;
            }
        }
        // Unknown assignment target — parse as expression (no-op result).
        self.parse_expr(&assign.right);
    }

    // ---- For loop -----------------------------------------------------------

    fn parse_for_loop(&mut self, for_loop: &ExprForLoop) {
        let loop_var_name = if let Pat::Ident(pat_ident) = &*for_loop.pat {
            pat_ident.ident.to_string()
        } else {
            return;
        };

        let Some((start_vid, end_vid, step_vid)) = self.parse_loop_range(&for_loop.expr) else {
            return;
        };

        let var_id = self.alloc_var();
        let loop_body_bid = self.alloc_bid();

        // Save outer scope; add loop variable with the +1000 legacy key that
        // the msl.rs emitter also registers.
        let prev_bindings = self.bindings.clone();
        self.bindings.insert(loop_var_name.clone(), var_id + 0x4000_0000);

        // Redirect IR emission to the loop body block.
        let prev_target =
            std::mem::replace(&mut self.current_target, format!("block_{loop_body_bid}"));
        let prev_stmts = std::mem::take(&mut self.ir_stmts);

        for stmt in &for_loop.body.stmts {
            self.parse_stmt(stmt);
        }

        let loop_body_tokens = std::mem::replace(&mut self.ir_stmts, prev_stmts);
        self.current_target = prev_target;
        self.bindings = prev_bindings;

        // Emit the Loop op into the parent block.
        self.push_op_no_result(quote! {
            Op::Loop {
                var: VarId::new(#var_id),
                start: ValueId::new(#start_vid),
                end: ValueId::new(#end_vid),
                step: ValueId::new(#step_vid),
                body: BlockId::new(#loop_body_bid),
            }
        });

        self.blocks.push((loop_body_bid, quote! { #(#loop_body_tokens)* }));
    }

    // ---- If / else ----------------------------------------------------------

    /// Parse `if cond { ... } [else { ... }]` into `Op::If`.
    fn parse_if(&mut self, if_expr: &ExprIf) -> u32 {
        let cond = self.parse_expr(&if_expr.cond);
        let then_bid = self.alloc_bid();

        // Collect then-block IR.
        let prev_target = std::mem::replace(&mut self.current_target, format!("block_{then_bid}"));
        let prev_stmts = std::mem::take(&mut self.ir_stmts);
        let prev_bindings = self.bindings.clone();
        for stmt in &if_expr.then_branch.stmts {
            self.parse_stmt(stmt);
        }
        let then_tokens = std::mem::replace(&mut self.ir_stmts, prev_stmts);
        self.current_target = prev_target;
        self.bindings = prev_bindings;

        // Collect else-block IR (if present).
        let else_bid_tokens = if let Some((_, else_expr)) = &if_expr.else_branch {
            let else_bid = self.alloc_bid();
            let prev_target =
                std::mem::replace(&mut self.current_target, format!("block_{else_bid}"));
            let prev_stmts = std::mem::take(&mut self.ir_stmts);
            let prev_bindings = self.bindings.clone();
            match else_expr.as_ref() {
                Expr::Block(else_block) =>
                    for stmt in &else_block.block.stmts {
                        self.parse_stmt(stmt);
                    },
                // else if — recurse
                Expr::If(nested_if) => {
                    self.parse_if(nested_if);
                },
                _ => {},
            }
            let else_tokens = std::mem::replace(&mut self.ir_stmts, prev_stmts);
            self.current_target = prev_target;
            self.bindings = prev_bindings;
            self.blocks.push((else_bid, quote! { #(#else_tokens)* }));
            quote! { Some(BlockId::new(#else_bid)) }
        } else {
            quote! { None }
        };

        self.push_op_no_result(quote! {
            Op::If {
                cond: ValueId::new(#cond),
                then_block: BlockId::new(#then_bid),
                else_block: #else_bid_tokens,
            }
        });
        self.blocks.push((then_bid, quote! { #(#then_tokens)* }));
        0 // Op::If produces no value
    }

    fn parse_loop_range(&mut self, expr: &Expr) -> Option<(u32, u32, u32)> {
        match expr {
            Expr::Call(call) if expr_to_path_string(&call.func) == "range" =>
                Some(self.parse_range_call(call)),
            Expr::Range(range) => self.parse_range_expr(range),
            _ => {
                self.push_error(syn::Error::new_spanned(
                    expr,
                    "unsupported loop iterator; use `range(start, end[, step])` or `start..end`",
                ));
                None
            },
        }
    }

    /// Parse `range(start, end[, step])` -- step defaults to literal 1.
    fn parse_range_call(&mut self, call: &ExprCall) -> (u32, u32, u32) {
        let path = expr_to_path_string(&call.func);
        if path != "range" {
            return (0, 0, 0);
        }
        let args: Vec<_> = call.args.iter().collect();
        let start = args.first().map(|a| self.parse_expr(a)).unwrap_or(0);
        let end = args.get(1).map(|a| self.parse_expr(a)).unwrap_or(0);
        let step = if let Some(a) = args.get(2) { self.parse_expr(a) } else { self.push_const(1) };
        (start, end, step)
    }

    /// Parse `start..end` Rust range syntax for `for` loops.
    fn parse_range_expr(&mut self, range: &ExprRange) -> Option<(u32, u32, u32)> {
        if matches!(range.limits, RangeLimits::Closed(_)) {
            self.push_error(syn::Error::new_spanned(
                range,
                "inclusive ranges are not supported in MetalTile loops; use `start..end`",
            ));
            return None;
        }

        let start = range
            .start
            .as_ref()
            .map(|expr| self.parse_expr(expr))
            .unwrap_or_else(|| self.push_const(0));
        let Some(end_expr) = range.end.as_ref() else {
            self.push_error(syn::Error::new_spanned(
                range,
                "open-ended ranges are not supported in MetalTile loops",
            ));
            return None;
        };
        let end = self.parse_expr(end_expr);
        let step = self.push_const(1);
        Some((start, end, step))
    }

    // ---- Expression parsing -------------------------------------------------

    fn parse_expr(&mut self, expr: &Expr) -> u32 {
        match expr {
            Expr::Binary(binary) => self.parse_binary(binary),
            Expr::Call(call) => self.parse_call(call),
            Expr::Index(index) => self.parse_index(index),
            Expr::Path(path) => self.resolve_path(path),
            Expr::Lit(lit) => self.parse_literal(lit),
            Expr::MethodCall(method) => self.parse_method_call(method),
            Expr::Unary(unary) => self.parse_unary(unary),
            Expr::Paren(paren) => self.parse_expr(&paren.expr),
            // `macro_rules!` captures (e.g. `$bits:literal`) are substituted
            // wrapped in `Delimiter::None` invisible groups, which syn parses
            // as `Expr::Group`. Unwrap so the inner literal / expression
            // reaches its real arm — otherwise the catch-all below allocates
            // a VID without pushing an Op and downstream consumers reference
            // an undeclared SSA value.
            Expr::Group(group) => self.parse_expr(&group.expr),
            Expr::If(if_expr) => self.parse_if(if_expr),
            Expr::ForLoop(_) => self.alloc_vid(),
            Expr::Macro(mac) => {
                // Same hazard as Stmt::Macro — fail loudly so silent-drop
                // regressions cannot recur.
                self.push_error(syn::Error::new_spanned(
                    &mac.mac.path,
                    "macro_rules! invocations inside #[kernel] bodies are not \
                     expanded by the metaltile proc-macro and would silently \
                     drop their body. Wrap the entire `#[kernel] fn` in the \
                     macro instead, or inline / replace with a DSL loop.",
                ));
                self.alloc_vid()
            },
            // Control-flow constructs the body parser does not lower. Without
            // these guards they fall into the `_` catch-all below, which
            // allocates a ValueId but emits NO IR — the construct silently
            // vanishes from the generated kernel and it ships wrong behavior
            // (a `while` reduction loop that does zero steps; a `return` that
            // falls through). Fail loudly instead, the same way `Expr::Macro`
            // does, and point at the DSL construct that *is* supported.
            Expr::While(w) => {
                self.push_error(syn::Error::new_spanned(
                    w,
                    "`while` loops are not supported inside #[kernel] bodies — \
                     the body parser would silently drop the loop. Use a DSL \
                     `for _ in range(start, end, step)` loop instead.",
                ));
                self.alloc_vid()
            },
            Expr::Loop(l) => {
                self.push_error(syn::Error::new_spanned(
                    l,
                    "`loop` is not supported inside #[kernel] bodies — the body \
                     parser would silently drop it. Use a bounded DSL \
                     `for _ in range(start, end, step)` loop instead.",
                ));
                self.alloc_vid()
            },
            Expr::Return(r) => {
                self.push_error(syn::Error::new_spanned(
                    r,
                    "`return` is not supported inside #[kernel] bodies — the \
                     body parser would silently drop it and execution would \
                     fall through. Use `if` / `else` branching instead.",
                ));
                self.alloc_vid()
            },
            Expr::Match(m) => {
                self.push_error(syn::Error::new_spanned(
                    m,
                    "`match` is not supported inside #[kernel] bodies — the \
                     body parser would silently drop it. Use `if` / `else` \
                     branching, or `select(cond, a, b)`.",
                ));
                self.alloc_vid()
            },
            Expr::Closure(c) => {
                self.push_error(syn::Error::new_spanned(
                    c,
                    "closures are not supported inside #[kernel] bodies — the \
                     body parser would silently drop them.",
                ));
                self.alloc_vid()
            },
            _ => self.alloc_vid(),
        }
    }

    fn parse_binary(&mut self, binary: &ExprBinary) -> u32 {
        let lhs = self.parse_expr(&binary.left);
        let rhs = self.parse_expr(&binary.right);
        let result = self.alloc_vid();
        let op_tokens = match binary.op {
            syn::BinOp::Add(_) => quote! { BinOpKind::Add },
            syn::BinOp::Sub(_) => quote! { BinOpKind::Sub },
            syn::BinOp::Mul(_) => quote! { BinOpKind::Mul },
            syn::BinOp::Div(_) => quote! { BinOpKind::Div },
            syn::BinOp::Lt(_) => quote! { BinOpKind::CmpLt },
            syn::BinOp::Gt(_) => quote! { BinOpKind::CmpGt },
            syn::BinOp::Le(_) => quote! { BinOpKind::CmpLe },
            syn::BinOp::Ge(_) => quote! { BinOpKind::CmpGe },
            syn::BinOp::Eq(_) => quote! { BinOpKind::CmpEq },
            syn::BinOp::Ne(_) => quote! { BinOpKind::CmpNe },
            syn::BinOp::BitAnd(_) => quote! { BinOpKind::BitAnd },
            syn::BinOp::BitOr(_) => quote! { BinOpKind::BitOr },
            syn::BinOp::BitXor(_) => quote! { BinOpKind::BitXor },
            syn::BinOp::Shl(_) => quote! { BinOpKind::Shl },
            syn::BinOp::Shr(_) => quote! { BinOpKind::Shr },
            syn::BinOp::Rem(_) => quote! { BinOpKind::Mod },
            syn::BinOp::And(_) => quote! { BinOpKind::And },
            syn::BinOp::Or(_) => quote! { BinOpKind::Or },
            _ => quote! { BinOpKind::Add },
        };
        self.push_op(
            quote! {
                Op::BinOp { op: #op_tokens, lhs: ValueId::new(#lhs), rhs: ValueId::new(#rhs) }
            },
            result,
        );
        result
    }

    fn parse_call(&mut self, call: &ExprCall) -> u32 {
        let path = expr_to_path_string(&call.func);
        match path.as_str() {
            "program_id" => self.parse_program_id(call),
            "arange" => self.parse_arange(call),
            "load" => self.parse_load(call),
            "store" => self.parse_store(call),
            "zeros" => self.parse_zeros(call),
            "dot" => self.parse_dot(call),
            "exp" => self.parse_unary_call(call, "exp"),
            "exp2" => self.parse_unary_call(call, "exp2"),
            "log" => self.parse_unary_call(call, "log"),
            "log2" => self.parse_unary_call(call, "log2"),
            "sqrt" => self.parse_unary_call(call, "sqrt"),
            "rsqrt" => self.parse_unary_call(call, "rsqrt"),
            "recip" => self.parse_unary_call(call, "recip"),
            "abs" => self.parse_unary_call(call, "abs"),
            "sin" => self.parse_unary_call(call, "sin"),
            "cos" => self.parse_unary_call(call, "cos"),
            "ceil" => self.parse_unary_call(call, "ceil"),
            "floor" => self.parse_unary_call(call, "floor"),
            "silu" => self.parse_unary_call(call, "silu"),
            "gelu" => self.parse_unary_call(call, "gelu"),
            "relu" => self.parse_unary_call(call, "relu"),
            "tanh" => self.parse_unary_call(call, "tanh"),
            "sigmoid" => self.parse_unary_call(call, "sigmoid"),
            "reduce_max" => self.parse_unary_call(call, "reduce_max"),
            "reduce_sum" => self.parse_unary_call(call, "reduce_sum"),
            "reduce_min" => self.parse_unary_call(call, "reduce_min"),
            "reduce_product" => self.parse_unary_call(call, "reduce_product"),
            "erf" => self.parse_unary_call(call, "erf"),
            "sign" => self.parse_unary_call(call, "sign"),
            "round" => self.parse_unary_call(call, "round"),
            "trunc" => self.parse_unary_call(call, "trunc"),
            "sinh" => self.parse_unary_call(call, "sinh"),
            "cosh" => self.parse_unary_call(call, "cosh"),
            "tan" => self.parse_unary_call(call, "tan"),
            "asin" => self.parse_unary_call(call, "asin"),
            "atan" => self.parse_unary_call(call, "atan"),
            "asinh" => self.parse_unary_call(call, "asinh"),
            "acos" => self.parse_unary_call(call, "acos"),
            "acosh" => self.parse_unary_call(call, "acosh"),
            "atanh" => self.parse_unary_call(call, "atanh"),
            "expm1" => self.parse_unary_call(call, "expm1"),
            "log10" => self.parse_unary_call(call, "log10"),
            "erfinv" => self.parse_unary_call(call, "erfinv"),
            "atan2" => self.parse_binary_call(call, "ATan2"),
            "remainder" => self.parse_binary_call(call, "Rem"),
            "max" => self.parse_binary_call(call, "Max"),
            "min" => self.parse_binary_call(call, "Min"),
            "cast" => self.parse_cast_call(call),
            "select" => self.parse_select_call(call),
            "pow" => self.parse_binary_call(call, "Pow"),
            "simd_sum" => self.parse_simd_reduce(call, "Sum"),
            "simd_max" => self.parse_simd_reduce(call, "Max"),
            "simd_min" => self.parse_simd_reduce(call, "Min"),
            "simd_shuffle_xor" => self.parse_simd_shuffle_xor(call),
            "simd_broadcast" => self.parse_simd_broadcast(call),
            // Device-scope atomics — target a kernel buffer parameter.
            "atomic_add" => self.parse_atomic(call, "Add", "Device"),
            "atomic_max" => self.parse_atomic(call, "Max", "Device"),
            "atomic_min" => self.parse_atomic(call, "Min", "Device"),
            "atomic_and" => self.parse_atomic(call, "And", "Device"),
            "atomic_or" => self.parse_atomic(call, "Or", "Device"),
            "atomic_xor" => self.parse_atomic(call, "Xor", "Device"),
            // Threadgroup-scope atomics — target a `threadgroup_alloc`'d
            // uint array.  Codegen reinterprets each slot as
            // `threadgroup atomic_uint*` for the call.  AURA encode's
            // pack stage races on shared u32 words with these.
            "atomic_add_tg" => self.parse_atomic(call, "Add", "Threadgroup"),
            "atomic_max_tg" => self.parse_atomic(call, "Max", "Threadgroup"),
            "atomic_min_tg" => self.parse_atomic(call, "Min", "Threadgroup"),
            "atomic_and_tg" => self.parse_atomic(call, "And", "Threadgroup"),
            "atomic_or_tg" => self.parse_atomic(call, "Or", "Threadgroup"),
            "atomic_xor_tg" => self.parse_atomic(call, "Xor", "Threadgroup"),
            "threadgroup_barrier" => self.parse_barrier(call),
            "simdgroup_barrier_mem_none" => self.parse_simdgroup_barrier(call),
            "threadgroup_alloc" => self.parse_threadgroup_alloc(call),
            "threadgroup_load" => self.parse_threadgroup_load(call),
            "threadgroup_store" => self.parse_threadgroup_store(call),
            // Per-thread stack arrays.  Mirror the threadgroup_* shape but
            // emit an unqualified `T name[size];` so Metal places the
            // array in per-thread registers / thread-local memory.  AURA
            // flash kernels need this for `q_vals[DIMS_PER_LANE]`,
            // `o[DIMS_PER_LANE]`, and the per-thread codebook cache.
            "stack_alloc" => self.parse_stack_alloc(call),
            "stack_load" => self.parse_stack_load(call),
            "stack_store" => self.parse_stack_store(call),
            "simd_scan_inclusive" => self.parse_simd_scan(call, false),
            "simd_scan_exclusive" => self.parse_simd_scan(call, true),
            "simdgroup_alloc" => self.parse_simdgroup_alloc(call),
            "simdgroup_elem_load" => self.parse_simdgroup_elem_load(call),
            "simdgroup_elem_store" => self.parse_simdgroup_elem_store(call),
            "simdgroup_load" => self.parse_simdgroup_load(call),
            "simdgroup_matmul" => self.parse_simdgroup_matmul(call),
            "simd_lane_id" => self.parse_simd_lane_id(call),
            "simd_group_id" => self.parse_simd_group_id(call),
            "neg_infinity" => self.parse_special_const(call, "-INFINITY"),
            "infinity" => self.parse_special_const(call, "INFINITY"),
            "strided_reduce" => self.parse_strided_reduce(call),
            "strided_reduce_exp_sub" => self.parse_strided_reduce_exp_sub(call),
            "strided_reduce_dot" => self.parse_strided_reduce_dot(call),
            "strided_store" => self.parse_strided_store(call),
            "strided_scan" => self.parse_strided_scan(call),
            "strided_argmax" => self.parse_strided_argreduce(call, "Max"),
            "strided_argmin" => self.parse_strided_argreduce(call, "Min"),
            "range" => 0,
            _ => {
                let callee = if path.is_empty() { "<expr>".to_string() } else { path };
                self.push_error_value(syn::Error::new_spanned(
                    &call.func,
                    format!("unrecognized MetalTile DSL call `{callee}`"),
                ))
            },
        }
    }

    fn parse_program_id(&mut self, call: &ExprCall) -> u32 {
        let axis = extract_turbofish_axis(&call.func).unwrap_or(0);
        let result = self.alloc_vid();
        self.push_op(quote! { Op::ProgramId { axis: #axis } }, result);
        result
    }

    fn parse_arange(&mut self, call: &ExprCall) -> u32 {
        let len_name = extract_turbofish_name(&call.func).unwrap_or_else(|| "1".to_string());
        let result = self.alloc_vid();
        self.push_op(
            quote! { Op::Arange { start: None, step: None, len: ConstExpr::new(#len_name) } },
            result,
        );
        result
    }

    fn parse_load(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        if args.is_empty() {
            return self.alloc_vid();
        }
        let (src_name, idx_tokens) = self.parse_tensor_index(args[0]);
        let result = self.alloc_vid();
        self.push_op(
            quote! {
                Op::Load {
                    src: #src_name.to_string(),
                    mask: None,
                    other: None,
                    indices: vec![#(#idx_tokens),*],
                }
            },
            result,
        );
        result
    }

    fn parse_store(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        if args.len() < 2 {
            return 0;
        }
        let (dst_name, idx_tokens) = self.parse_tensor_index(args[0]);
        let value_vid = self.parse_expr(args[1]);
        self.push_op_no_result(quote! {
            Op::Store {
                dst: #dst_name.to_string(),
                mask: None,
                indices: vec![#(#idx_tokens),*],
                value: ValueId::new(#value_vid),
            }
        });
        0
    }

    fn parse_zeros(&mut self, _call: &ExprCall) -> u32 {
        let result = self.alloc_vid();
        self.push_op(
            quote! {
                Op::Zeros { dtype: DType::F32, shape: Shape::scalar() }
            },
            result,
        );
        result
    }

    fn parse_dot(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        let a = args.first().map(|a| self.parse_expr(a)).unwrap_or_else(|| self.alloc_vid());
        let b = args.get(1).map(|b| self.parse_expr(b)).unwrap_or_else(|| self.alloc_vid());
        let result = self.alloc_vid();
        self.push_op(quote! { Op::Dot { a: ValueId::new(#a), b: ValueId::new(#b) } }, result);
        result
    }

    fn parse_unary_call(&mut self, call: &ExprCall, fn_name: &str) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        let val = args.first().map(|a| self.parse_expr(a)).unwrap_or_else(|| self.alloc_vid());
        let result = self.alloc_vid();
        match fn_name {
            "exp" | "exp2" | "log" | "log2" | "sqrt" | "rsqrt" | "abs" | "sin" | "cos" | "ceil"
            | "floor" | "recip" | "erf" | "sign" | "round" | "trunc" | "sinh" | "cosh" | "tan"
            | "asin" | "acos" | "atan" | "asinh" | "acosh" | "atanh" | "expm1" | "log10"
            | "erfinv" => {
                let op_tokens = match fn_name {
                    "exp" => quote! { UnaryOpKind::Exp },
                    "exp2" => quote! { UnaryOpKind::Exp2 },
                    "log" => quote! { UnaryOpKind::Log },
                    "log2" => quote! { UnaryOpKind::Log2 },
                    "sqrt" => quote! { UnaryOpKind::Sqrt },
                    "rsqrt" => quote! { UnaryOpKind::Rsqrt },
                    "abs" => quote! { UnaryOpKind::Abs },
                    "sin" => quote! { UnaryOpKind::Sin },
                    "cos" => quote! { UnaryOpKind::Cos },
                    "ceil" => quote! { UnaryOpKind::Ceil },
                    "floor" => quote! { UnaryOpKind::Floor },
                    "recip" => quote! { UnaryOpKind::Recip },
                    "erf" => quote! { UnaryOpKind::Erf },
                    "sign" => quote! { UnaryOpKind::Sign },
                    "round" => quote! { UnaryOpKind::Round },
                    "sinh" => quote! { UnaryOpKind::Sinh },
                    "cosh" => quote! { UnaryOpKind::Cosh },
                    "tan" => quote! { UnaryOpKind::Tan },
                    "asin" => quote! { UnaryOpKind::Asin },
                    "atan" => quote! { UnaryOpKind::Atan },
                    "asinh" => quote! { UnaryOpKind::Asinh },
                    "trunc" => quote! { UnaryOpKind::Trunc },
                    "acos" => quote! { UnaryOpKind::Acos },
                    "acosh" => quote! { UnaryOpKind::Acosh },
                    "atanh" => quote! { UnaryOpKind::Atanh },
                    "expm1" => quote! { UnaryOpKind::Expm1 },
                    "log10" => quote! { UnaryOpKind::Log10 },
                    "erfinv" => quote! { UnaryOpKind::ErfInv },
                    _ => quote! { UnaryOpKind::Exp },
                };
                self.push_op(
                    quote! {
                        Op::UnaryOp { op: #op_tokens, value: ValueId::new(#val) }
                    },
                    result,
                );
            },
            "silu" | "gelu" | "relu" | "tanh" | "sigmoid" => {
                let kind = match fn_name {
                    "silu" => quote! { ActKind::Silu },
                    "gelu" => quote! { ActKind::Gelu },
                    "relu" => quote! { ActKind::Relu },
                    "tanh" => quote! { ActKind::Tanh },
                    "sigmoid" => quote! { ActKind::Sigmoid },
                    _ => quote! { ActKind::Silu },
                };
                self.push_op(
                    quote! {
                        Op::Activation { kind: #kind, value: ValueId::new(#val) }
                    },
                    result,
                );
            },
            "reduce_sum" | "reduce_max" | "reduce_min" | "reduce_product" => {
                let op = match fn_name {
                    "reduce_sum" => quote! { ReduceKind::Sum },
                    "reduce_max" => quote! { ReduceKind::Max },
                    "reduce_min" => quote! { ReduceKind::Min },
                    "reduce_product" => quote! { ReduceKind::Product },
                    _ => quote! { ReduceKind::Sum },
                };
                self.push_op(
                    quote! {
                        Op::Reduce { value: ValueId::new(#val), axis: 0, op: #op }
                    },
                    result,
                );
            },
            _ => {
                self.push_op(
                    quote! {
                        Op::Reduce { value: ValueId::new(#val), axis: 0, op: ReduceKind::Sum }
                    },
                    result,
                );
            },
        }
        result
    }

    fn parse_binary_call(&mut self, call: &ExprCall, kind: &str) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        let lhs = args.first().map(|a| self.parse_expr(a)).unwrap_or_else(|| self.alloc_vid());
        let rhs = args.get(1).map(|a| self.parse_expr(a)).unwrap_or_else(|| self.alloc_vid());
        let result = self.alloc_vid();
        let op_tokens = match kind {
            "Max" => quote! { BinOpKind::Max },
            "Min" => quote! { BinOpKind::Min },
            "Pow" => quote! { BinOpKind::Pow },
            "ATan2" => quote! { BinOpKind::ATan2 },
            "Rem" => quote! { BinOpKind::Rem },
            _ => quote! { BinOpKind::Add },
        };
        self.push_op(
            quote! {
                Op::BinOp { op: #op_tokens, lhs: ValueId::new(#lhs), rhs: ValueId::new(#rhs) }
            },
            result,
        );
        result
    }

    fn parse_cast_call(&mut self, call: &ExprCall) -> u32 {
        let value_id = if call.args.is_empty() { 0u32 } else { self.parse_expr(&call.args[0]) };
        let dtype_tokens = extract_turbofish_name(&call.func)
            .map(|s| {
                // Check if it's a generic type var (e.g. "T" → _t arg variable).
                if let Some(ts) = self.type_vars.get(&s) {
                    ts.clone()
                } else {
                    dtype_tokens_for_name(&s)
                }
            })
            .unwrap_or_else(|| quote! { DType::F32 });
        let result = self.alloc_vid();
        self.push_op(
            quote! { Op::Cast { value: ValueId::new(#value_id), dtype: #dtype_tokens } },
            result,
        );
        result
    }

    /// `strided_reduce(src, offset, end, op)` or `strided_reduce(src, offset, stride, end, op)` -> Op::StrideReduce
    /// 4-arg form: stride is implicit (ignored in Reduction mode; resolves to 1 on CPU).
    fn parse_strided_reduce(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        if args.len() < 4 {
            return self.alloc_vid();
        }
        let src_name = expr_to_path_string(args[0]);
        if src_name.is_empty() {
            return self.alloc_vid();
        }
        // 4-arg: (src, offset, end, op)   — implicit stride
        // 5-arg: (src, offset, stride, end, op)
        let (offset, stride, end, op_name) = if args.len() >= 5 {
            let off = self.parse_expr(args[1]);
            let st = self.parse_expr(args[2]);
            let en = self.parse_expr(args[3]);
            let op = expr_to_path_string(args[4]);
            (off, st, en, op)
        } else {
            let off = self.parse_expr(args[1]);
            let en = self.parse_expr(args[2]);
            let op = expr_to_path_string(args[3]);
            (off, 0u32, en, op) // stride=0 → ignored in Reduction mode / resolves to 1 on CPU
        };
        let op = match op_name.as_str() {
            "sum" => quote! { ReduceKind::Sum },
            "max" => quote! { ReduceKind::Max },
            "min" => quote! { ReduceKind::Min },
            "product" => quote! { ReduceKind::Product },
            _ => quote! { ReduceKind::Sum },
        };
        let result = self.alloc_vid();
        self.push_op(
            quote! {
                Op::StrideReduce {
                    src: #src_name.to_string(),
                    offset: ValueId::new(#offset),
                    stride: ValueId::new(#stride),
                    end: ValueId::new(#end),
                    op: #op,
                    dtype: DType::F32,
                    transform: None,
                    secondary_src: None,
                    secondary_base: None,
                }
            },
            result,
        );
        result
    }

    /// `strided_reduce_exp_sub(src, offset, end, sub_val)` or `(src, offset, stride, end, sub_val)`
    fn parse_strided_reduce_exp_sub(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        if args.len() < 4 {
            return self.alloc_vid();
        }
        let src_name = expr_to_path_string(args[0]);
        if src_name.is_empty() {
            return self.alloc_vid();
        }
        // 4-arg: (src, offset, end, sub_val)   5-arg: (src, offset, stride, end, sub_val)
        let (offset, stride, end, sub_val) = if args.len() >= 5 {
            (
                self.parse_expr(args[1]),
                self.parse_expr(args[2]),
                self.parse_expr(args[3]),
                self.parse_expr(args[4]),
            )
        } else {
            let off = self.parse_expr(args[1]);
            let en = self.parse_expr(args[2]);
            let sv = self.parse_expr(args[3]);
            (off, 0u32, en, sv)
        };
        let result = self.alloc_vid();
        self.push_op(
            quote! {
                Op::StrideReduce {
                    src: #src_name.to_string(),
                    offset: ValueId::new(#offset),
                    stride: ValueId::new(#stride),
                    end: ValueId::new(#end),
                    op: ReduceKind::Sum,
                    dtype: DType::F32,
                    transform: Some(vec![
                        Op::BinOp {
                            op: BinOpKind::Sub,
                            lhs: ValueId::new(0),
                            rhs: ValueId::new(#sub_val),
                        },
                        Op::UnaryOp {
                            op: UnaryOpKind::Exp,
                            value: ValueId::new(0),
                        },
                    ]),
                    secondary_src: None,
                    secondary_base: None,
                }
            },
            result,
        );
        result
    }

    /// `strided_reduce_dot(a, b, offset, base, end)` — dot product sum(a[i] * b[i - base])
    fn parse_strided_reduce_dot(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        if args.len() < 5 {
            return self.alloc_vid();
        }
        let src_a = expr_to_path_string(args[0]);
        let src_b = expr_to_path_string(args[1]);
        if src_a.is_empty() || src_b.is_empty() {
            return self.alloc_vid();
        }
        let offset = self.parse_expr(args[2]);
        let base = self.parse_expr(args[3]);
        let end = self.parse_expr(args[4]);
        let result = self.alloc_vid();
        self.push_op(
            quote! {
                Op::StrideReduce {
                    src: #src_a.to_string(),
                    offset: ValueId::new(#offset),
                    stride: ValueId::new(0),
                    end: ValueId::new(#end),
                    op: ReduceKind::Sum,
                    dtype: DType::F32,
                    transform: None,
                    secondary_src: Some(#src_b.to_string()),
                    secondary_base: Some(ValueId::new(#base)),
                }
            },
            result,
        );
        result
    }

    /// `strided_store(src, dst, offset, end, scalar[, aux_src])` -> Op::StrideStore
    fn parse_strided_store(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        if args.len() < 5 {
            return 0;
        }
        let src_name = expr_to_path_string(args[0]);
        let dst_name = expr_to_path_string(args[1]);
        if src_name.is_empty() || dst_name.is_empty() {
            return 0;
        }
        let offset = self.parse_expr(args[2]);
        let end = self.parse_expr(args[3]);
        let scalar = self.parse_expr(args[4]);
        let aux_src_tokens = if args.len() >= 6 {
            let aux_name = expr_to_path_string(args[5]);
            quote! { Some(#aux_name.to_string()) }
        } else {
            quote! { None }
        };
        self.push_op_no_result(quote! {
            Op::StrideStore {
                src: #src_name.to_string(),
                dst: #dst_name.to_string(),
                offset: ValueId::new(#offset),
                end: ValueId::new(#end),
                scalar: ValueId::new(#scalar),
                aux_src: #aux_src_tokens,
            }
        });
        0
    }

    /// `strided_scan(src, dst, start, end)` — serial inclusive prefix sum of src[start..end]
    /// into dst[start..end].  Single-threaded: dispatch [B,1,1]×[1,1,1].
    fn parse_strided_scan(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        if args.len() < 4 {
            return 0;
        }
        let src_name = expr_to_path_string(args[0]);
        let dst_name = expr_to_path_string(args[1]);
        if src_name.is_empty() || dst_name.is_empty() {
            return 0;
        }
        let offset = self.parse_expr(args[2]);
        let end = self.parse_expr(args[3]);
        self.push_op_no_result(quote! {
            Op::StrideScan {
                src: #src_name.to_string(),
                dst: #dst_name.to_string(),
                offset: ValueId::new(#offset),
                end: ValueId::new(#end),
                op: ReduceKind::Sum,
            }
        });
        0
    }

    /// `strided_argmax(src, start, end)` or `strided_argmin(src, start, end)`
    /// — serial argmax/argmin of src[start..end], returns the flat index.
    fn parse_strided_argreduce(&mut self, call: &ExprCall, op_str: &str) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        if args.len() < 3 {
            return self.alloc_vid();
        }
        let src_name = expr_to_path_string(args[0]);
        if src_name.is_empty() {
            return self.alloc_vid();
        }
        let offset = self.parse_expr(args[1]);
        let end = self.parse_expr(args[2]);
        let op = match op_str {
            "Max" => quote! { ReduceKind::Max },
            _ => quote! { ReduceKind::Min },
        };
        let result = self.alloc_vid();
        self.push_op(
            quote! {
                Op::StrideArgReduce {
                    src: #src_name.to_string(),
                    offset: ValueId::new(#offset),
                    end: ValueId::new(#end),
                    op: #op,
                }
            },
            result,
        );
        result
    }

    // ---- Method calls (.cast::<T>(), .t(), .slice()) ------------------------

    fn parse_method_call(&mut self, method: &syn::ExprMethodCall) -> u32 {
        let method_name = method.method.to_string();

        if method_name == "cast" {
            let receiver_vid = self.parse_expr(&method.receiver);
            let result = self.alloc_vid();
            let dtype = method
                .turbofish
                .as_ref()
                .and_then(|args| {
                    args.args.first().and_then(|arg| {
                        if let syn::GenericArgument::Type(syn::Type::Path(tp)) = arg {
                            tp.path.segments.last().map(|s| s.ident.to_string())
                        } else {
                            None
                        }
                    })
                })
                .map(|n| match n.as_str() {
                    "f16" => quote! { DType::F16 },
                    "f32" => quote! { DType::F32 },
                    "bf16" => quote! { DType::BF16 },
                    "i32" => quote! { DType::I32 },
                    "u32" => quote! { DType::U32 },
                    other =>
                        if let Some(ts) = self.type_vars.get(other) {
                            ts.clone()
                        } else {
                            quote! { DType::F32 }
                        },
                })
                .unwrap_or(quote! { DType::F16 });
            self.push_op(
                quote! {
                    Op::Cast { value: ValueId::new(#receiver_vid), dtype: #dtype }
                },
                result,
            );
            return result;
        }

        if method_name == "t" {
            let receiver_vid = self.parse_expr(&method.receiver);
            let result = self.alloc_vid();
            self.push_op(
                quote! {
                    Op::Transpose { value: ValueId::new(#receiver_vid) }
                },
                result,
            );
            return result;
        }

        if method_name == "slice" {
            return self.parse_expr(&method.receiver);
        }

        self.alloc_vid()
    }

    // ---- Indexing / path / literal ------------------------------------------

    fn parse_tensor_index(&mut self, expr: &Expr) -> (String, Vec<TokenStream>) {
        if let Expr::Index(index) = expr {
            let src_name = expr_to_path_string(&index.expr);
            let idx_tokens = self.index_as_tokens(&index.index);
            (src_name, idx_tokens)
        } else {
            (expr_to_path_string(expr), vec![quote! { IndexExpr::Value(ValueId::new(0)) }])
        }
    }

    fn index_as_tokens(&mut self, idx: &Expr) -> Vec<TokenStream> {
        match idx {
            Expr::Tuple(tuple) => tuple.elems.iter().map(|e| self.single_index_token(e)).collect(),
            _ => vec![self.single_index_token(idx)],
        }
    }

    fn single_index_token(&mut self, idx: &Expr) -> TokenStream {
        match idx {
            Expr::Lit(lit) => {
                let val: i64 = match &lit.lit {
                    syn::Lit::Int(n) => n.base10_parse::<i64>().unwrap_or(0),
                    syn::Lit::Float(f) => f.base10_parse::<f64>().map(|v| v as i64).unwrap_or(0),
                    _ => 0,
                };
                quote! { IndexExpr::Const(#val) }
            },
            _ => {
                let vid = self.parse_expr(idx);
                quote! { IndexExpr::Value(ValueId::new(#vid)) }
            },
        }
    }

    fn parse_index(&mut self, index: &ExprIndex) -> u32 { self.parse_expr(&index.index) }

    /// Resolve a path to a ValueId. Constexprs are auto-loaded on first use
    /// via `Op::Load { src: name, indices: [] }` and cached.
    fn resolve_path(&mut self, path: &ExprPath) -> u32 {
        let name = path_to_string(&path.path);

        if let Some(&vid) = self.bindings.get(&name) {
            return vid;
        }

        if self.constexpr_names.contains(&name) {
            let result = self.alloc_vid();
            let n = name.clone();
            self.push_op(
                quote! {
                    Op::Load { src: #n.to_string(), mask: None, other: None, indices: vec![] }
                },
                result,
            );
            self.bindings.insert(name, result);
            return result;
        }

        // Mutable local variable read: emit Op::Load { src: "__ml_{name}" }.
        if self.mut_locals.contains(&name) {
            let result = self.alloc_vid();
            let src = format!("__ml_{name}");
            self.push_op(
                quote! {
                    Op::Load { src: #src.to_string(), mask: None, other: None, indices: vec![] }
                },
                result,
            );
            return result;
        }

        // GPU built-in scalars available in every kernel preamble.
        // Emitted as Op::Load { src: "<name>", indices: [] } so the MSL emitter
        // outputs `auto vN = tid;` (or lsize, tgid_x/y/z, simd_lane, simd_id, n_simd).
        if matches!(
            name.as_str(),
            "tid" | "lsize" | "tgid_x" | "tgid_y" | "tgid_z" | "simd_lane" | "simd_id" | "n_simd"
        ) {
            let result = self.alloc_vid();
            let n = name.clone();
            self.push_op(
                quote! {
                    Op::Load { src: #n.to_string(), mask: None, other: None, indices: vec![] }
                },
                result,
            );
            self.bindings.insert(name, result);
            return result;
        }

        0
    }

    fn parse_literal(&mut self, lit: &syn::ExprLit) -> u32 {
        match &lit.lit {
            // Float literals: emit as Op::Load { src: "<val>f" } so that MSL
            // deduces the correct `float` (not `int`) type via `auto`.
            syn::Lit::Float(f) => {
                let fval: f64 = f.base10_parse::<f64>().unwrap_or(0.0);
                // Format as a Metal float literal (suffix 'f'). Must include a
                // decimal point — "0f" is not valid MSL, "0.0f" is.
                let s = format!("{fval}");
                let src = if s.contains('.') { format!("{s}f") } else { format!("{s}.0f") };
                let result = self.alloc_vid();
                self.push_op(
                    quote! {
                        Op::Load {
                            src: #src.to_string(),
                            indices: Vec::new(),
                            mask: None,
                            other: None,
                        }
                    },
                    result,
                );
                result
            },
            // Integer literals: keep existing Op::Const path.
            syn::Lit::Int(n) => {
                let val: i64 = n.base10_parse::<i64>().unwrap_or(0);
                let result = self.alloc_vid();
                self.push_op(quote! { Op::Const { value: #val } }, result);
                result
            },
            _ => {
                let result = self.alloc_vid();
                self.push_op(quote! { Op::Const { value: 0i64 } }, result);
                result
            },
        }
    }

    fn parse_unary(&mut self, unary: &syn::ExprUnary) -> u32 {
        match unary.op {
            syn::UnOp::Neg(_) => {
                let val = self.parse_expr(&unary.expr);
                let result = self.alloc_vid();
                self.push_op(
                    quote! {
                        Op::UnaryOp { op: UnaryOpKind::Neg, value: ValueId::new(#val) }
                    },
                    result,
                );
                result
            },
            _ => self.parse_expr(&unary.expr),
        }
    }

    /// `select(cond, on_true, on_false)` → Op::Select
    fn parse_select_call(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        if args.len() < 3 {
            return self.alloc_vid();
        }
        let cond = self.parse_expr(args[0]);
        let on_true = self.parse_expr(args[1]);
        let on_false = self.parse_expr(args[2]);
        let result = self.alloc_vid();
        self.push_op(
            quote! {
                Op::Select {
                    cond: ValueId::new(#cond),
                    on_true: ValueId::new(#on_true),
                    on_false: ValueId::new(#on_false),
                }
            },
            result,
        );
        result
    }

    /// `simd_sum/max/min(val)` → Op::SimdReduce
    fn parse_simd_reduce(&mut self, call: &ExprCall, op_str: &str) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        let val = args.first().map(|a| self.parse_expr(a)).unwrap_or_else(|| self.alloc_vid());
        let result = self.alloc_vid();
        let op_tokens = match op_str {
            "Max" => quote! { ReduceKind::Max },
            "Min" => quote! { ReduceKind::Min },
            _ => quote! { ReduceKind::Sum },
        };
        self.push_op(
            quote! {
                Op::SimdReduce { value: ValueId::new(#val), op: #op_tokens }
            },
            result,
        );
        result
    }

    /// `simd_shuffle_xor(val, mask)` → Op::SimdShuffleXor.
    /// Each lane receives the value held by lane `lane_id ^ mask`. `mask` is a
    /// compile-time u32 literal (typically `1, 2, 4, 8, …` from a butterfly
    /// stride loop — AURA's FWHT and Steel attention row reductions).
    fn parse_simd_shuffle_xor(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        let val = args.first().map(|a| self.parse_expr(a)).unwrap_or_else(|| self.alloc_vid());
        let mask = args.get(1).map(|a| literal_u32(a)).unwrap_or(0);
        let result = self.alloc_vid();
        self.push_op(
            quote! {
                Op::SimdShuffleXor {
                    value: ValueId::new(#val),
                    mask: #mask,
                }
            },
            result,
        );
        result
    }

    /// `simd_broadcast(value, lane)` → Op::SimdBroadcast.
    /// Broadcasts `lane`'s value to every lane in the simdgroup. Used by
    /// AURA's cooperative codebook hoist.
    fn parse_simd_broadcast(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        let val = args.first().map(|a| self.parse_expr(a)).unwrap_or_else(|| self.alloc_vid());
        let lane = args.get(1).map(|a| self.parse_expr(a)).unwrap_or_else(|| self.alloc_vid());
        let result = self.alloc_vid();
        self.push_op(
            quote! {
                Op::SimdBroadcast {
                    value: ValueId::new(#val),
                    lane: ValueId::new(#lane),
                }
            },
            result,
        );
        result
    }

    /// `atomic_<op>(dst, index, value)` → `Op::Atomic`.  `dst` must be a
    /// string literal:
    ///   * Device scope (default `atomic_<op>(…)`): a kernel buffer
    ///     parameter name.
    ///   * Threadgroup scope (`atomic_<op>_tg(…)`): a name that was
    ///     declared via `threadgroup_alloc(name, size, "u32")` earlier in
    ///     the kernel.
    ///
    /// `index` and `value` are SSA expressions.  No result — atomics are
    /// side-effecting stores.
    fn parse_atomic(&mut self, call: &ExprCall, op_str: &str, scope_str: &str) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        let dst = if let Some(syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(s), .. })) =
            args.first().map(|a| match a {
                syn::Expr::Group(g) => &*g.expr,
                other => other,
            }) {
            s.value()
        } else {
            self.push_error(syn::Error::new_spanned(
                &call.func,
                format!("atomic_{} expects a string literal as first argument (the destination param name)", op_str.to_lowercase()),
            ));
            String::new()
        };
        let index = args.get(1).map(|a| self.parse_expr(a)).unwrap_or_else(|| self.alloc_vid());
        let value = args.get(2).map(|a| self.parse_expr(a)).unwrap_or_else(|| self.alloc_vid());
        let op_tokens = match op_str {
            "Add" => quote! { AtomicKind::Add },
            "Max" => quote! { AtomicKind::Max },
            "Min" => quote! { AtomicKind::Min },
            "And" => quote! { AtomicKind::And },
            "Or" => quote! { AtomicKind::Or },
            "Xor" => quote! { AtomicKind::Xor },
            _ => quote! { AtomicKind::Add },
        };
        let scope_tokens = match scope_str {
            "Threadgroup" => quote! { AtomicScope::Threadgroup },
            _ => quote! { AtomicScope::Device },
        };
        self.push_op_no_result(quote! {
            Op::Atomic {
                op: #op_tokens,
                scope: #scope_tokens,
                dst: #dst.to_string(),
                index: ValueId::new(#index),
                value: ValueId::new(#value),
            }
        });
        0
    }

    /// `threadgroup_barrier()` → Op::Barrier (no result — DCE keeps no-result ops)
    fn parse_barrier(&mut self, _call: &ExprCall) -> u32 {
        self.push_op_no_result(quote! { Op::Barrier });
        0
    }

    /// `simdgroup_barrier_mem_none()` → Op::SimdgroupBarrier — compiler-only
    /// reordering hint, zero runtime cost. Emits `simdgroup_barrier(mem_flags::mem_none)`.
    fn parse_simdgroup_barrier(&mut self, _call: &ExprCall) -> u32 {
        self.push_op_no_result(quote! { Op::SimdgroupBarrier });
        0
    }

    /// `threadgroup_alloc("name", size [, dtype])` → Op::ThreadgroupAlloc.
    ///
    /// `dtype` is an optional 3rd argument and accepts either form:
    ///   * **Type-path form**: `T` (resolves to the kernel's generic
    ///     type), or `f32` / `f16` / `bf16` / `u32` / `i32` (specific
    ///     dtype). Used by `mlx/sort.rs` etc.
    ///   * **String-literal form**: `"f32"` / `"f16"` / `"bf16"` /
    ///     `"u32"` / `"i32"`. Used by AURA encode for `"u32"` so the
    ///     threadgroup pack buffer can be reinterpreted as `atomic_uint`
    ///     by the threadgroup-scoped atomic ops.
    ///
    /// Defaults to F32 if omitted.
    fn parse_threadgroup_alloc(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        let name = string_lit_from_expr(args.first().unwrap_or(&&*call.func));
        let size: usize = usize_lit_from_expr(args.get(1).copied());
        let size_u32 = size as u32;
        let dtype_ts = if let Some(arg) = args.get(2) {
            // Try string-literal form first (`"u32"`, `"f32"`, ...).
            // macro_rules!-substituted args show up wrapped in
            // Expr::Group; unwrap to peek through.
            let unwrapped: &syn::Expr = match arg {
                syn::Expr::Group(g) => &g.expr,
                other => other,
            };
            if let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(s), .. }) = unwrapped {
                match s.value().as_str() {
                    "f32" => quote! { DType::F32 },
                    "f16" => quote! { DType::F16 },
                    "bf16" => quote! { DType::BF16 },
                    "u32" => quote! { DType::U32 },
                    "i32" => quote! { DType::I32 },
                    other => {
                        self.push_error(syn::Error::new_spanned(
                            arg,
                            format!(
                                "threadgroup_alloc dtype must be one of \
                                 f32/f16/bf16/u32/i32 (got {other:?})"
                            ),
                        ));
                        quote! { DType::F32 }
                    },
                }
            } else {
                // Fall back to type-path form (`T`, `f32`, `f16`, ...).
                let ty_str = quote! { #arg }.to_string();
                match ty_str.trim() {
                    "T" =>
                        if let Some(tok) = self.type_vars.get("T") {
                            tok.clone()
                        } else {
                            quote! { DType::F32 }
                        },
                    "f32" => quote! { DType::F32 },
                    "f16" => quote! { DType::F16 },
                    "bf16" => quote! { DType::BF16 },
                    "i32" => quote! { DType::I32 },
                    "u32" => quote! { DType::U32 },
                    _ => quote! { DType::F32 },
                }
            }
        } else {
            quote! { DType::F32 }
        };
        self.push_op_no_result(quote! {
            Op::ThreadgroupAlloc {
                dtype: #dtype_ts,
                size: #size_u32,
                name: #name.to_string(),
            }
        });
        0
    }

    /// `threadgroup_load("name", idx)` → Op::ThreadgroupLoad
    fn parse_threadgroup_load(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        let name = string_lit_from_expr(args.first().unwrap_or(&&*call.func));
        let idx_vid = args.get(1).map(|a| self.parse_expr(a)).unwrap_or(0);
        let result = self.alloc_vid();
        self.push_op(
            quote! {
                Op::ThreadgroupLoad {
                    name: #name.to_string(),
                    index: ValueId::new(#idx_vid),
                }
            },
            result,
        );
        result
    }

    /// `threadgroup_store("name", idx, val)` → Op::ThreadgroupStore (no result).
    fn parse_threadgroup_store(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        let name = string_lit_from_expr(args.first().unwrap_or(&&*call.func));
        let idx_vid = args.get(1).map(|a| self.parse_expr(a)).unwrap_or(0);
        let val_vid = args.get(2).map(|a| self.parse_expr(a)).unwrap_or(0);
        self.push_op_no_result(quote! {
            Op::ThreadgroupStore {
                name: #name.to_string(),
                index: ValueId::new(#idx_vid),
                value: ValueId::new(#val_vid),
            }
        });
        0
    }

    /// `stack_alloc("name", size [, dtype])` → Op::StackAlloc.
    ///
    /// Per-thread stack-resident array.  `dtype` is an optional 3rd
    /// argument as a string literal (`"f32"` / `"f16"` / `"bf16"` /
    /// `"u32"` / `"i32"`); defaults to `f32`.  Metal places small
    /// fixed-size local arrays in registers; AURA flash kernels use
    /// this for the per-lane `q_vals[]`, `o[]`, and codebook caches.
    fn parse_stack_alloc(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        let name = string_lit_from_expr(args.first().unwrap_or(&&*call.func));
        let size: usize = usize_lit_from_expr(args.get(1).copied());
        let size_u32 = size as u32;

        let dtype_tokens = if let Some(arg) = args.get(2) {
            let unwrapped: &syn::Expr = match arg {
                syn::Expr::Group(g) => &g.expr,
                other => other,
            };
            if let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(s), .. }) = unwrapped {
                match s.value().as_str() {
                    "f32" => quote! { DType::F32 },
                    "f16" => quote! { DType::F16 },
                    "bf16" => quote! { DType::BF16 },
                    "u32" => quote! { DType::U32 },
                    "i32" => quote! { DType::I32 },
                    other => {
                        self.push_error(syn::Error::new_spanned(
                            arg,
                            format!(
                                "stack_alloc dtype must be one of f32/f16/bf16/u32/i32 (got {other:?})"
                            ),
                        ));
                        quote! { DType::F32 }
                    },
                }
            } else {
                self.push_error(syn::Error::new_spanned(
                    arg,
                    "stack_alloc dtype must be a string literal",
                ));
                quote! { DType::F32 }
            }
        } else {
            quote! { DType::F32 }
        };

        self.push_op_no_result(quote! {
            Op::StackAlloc {
                dtype: #dtype_tokens,
                size: #size_u32,
                name: #name.to_string(),
            }
        });
        0
    }

    /// `stack_load("name", idx)` → Op::StackLoad.
    fn parse_stack_load(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        let name = string_lit_from_expr(args.first().unwrap_or(&&*call.func));
        let idx_vid = args.get(1).map(|a| self.parse_expr(a)).unwrap_or(0);
        let result = self.alloc_vid();
        self.push_op(
            quote! {
                Op::StackLoad {
                    name: #name.to_string(),
                    index: ValueId::new(#idx_vid),
                }
            },
            result,
        );
        result
    }

    /// `stack_store("name", idx, val)` → Op::StackStore (no result).
    fn parse_stack_store(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        let name = string_lit_from_expr(args.first().unwrap_or(&&*call.func));
        let idx_vid = args.get(1).map(|a| self.parse_expr(a)).unwrap_or(0);
        let val_vid = args.get(2).map(|a| self.parse_expr(a)).unwrap_or(0);
        self.push_op_no_result(quote! {
            Op::StackStore {
                name: #name.to_string(),
                index: ValueId::new(#idx_vid),
                value: ValueId::new(#val_vid),
            }
        });
        0
    }

    /// `simd_scan_inclusive(x)` / `simd_scan_exclusive(x)` → Op::SimdScan
    fn parse_simd_scan(&mut self, call: &ExprCall, exclusive: bool) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        let val_vid = args.first().map(|a| self.parse_expr(a)).unwrap_or(0);
        let result = self.alloc_vid();
        self.push_op(
            quote! {
                Op::SimdScan {
                    value: ValueId::new(#val_vid),
                    op: ReduceKind::Sum,
                    exclusive: #exclusive,
                }
            },
            result,
        );
        result
    }

    /// `simdgroup_alloc::<T, M, N>()` → Op::SimdgroupAlloc.
    /// Accepts concrete dtype names (`f32`, `f16`, `bf16`, …) or a kernel
    /// generic type-var name (e.g. `T`) which is resolved against the
    /// instantiated dtype via `type_vars`.
    fn parse_simdgroup_alloc(&mut self, call: &ExprCall) -> u32 {
        let result = self.alloc_vid();
        let raw = extract_turbofish_name_and_mn(&call.func);
        let dtype_tokens = match raw.as_ref().map(|(n, ..)| n.as_str()) {
            Some("f32") | Some("float") => quote! { DType::F32 },
            Some("f16") | Some("half") => quote! { DType::F16 },
            Some("bf16") | Some("bfloat") => quote! { DType::BF16 },
            Some("u32") | Some("uint") => quote! { DType::U32 },
            Some("i32") | Some("int") => quote! { DType::I32 },
            Some(other) =>
                if let Some(ts) = self.type_vars.get(other) {
                    ts.clone()
                } else {
                    quote! { DType::F32 }
                },
            None => quote! { DType::F16 },
        };
        let m_val = raw.as_ref().map(|(_, m, _)| *m).unwrap_or(8u32);
        let n_val = raw.as_ref().map(|(_, _, n)| *n).unwrap_or(8u32);
        self.push_op(
            quote! {
                Op::SimdgroupAlloc { dtype: #dtype_tokens, m: #m_val, n: #n_val }
            },
            result,
        );
        result
    }

    /// `simdgroup_elem_load(sm, index)` → Op::SimdgroupElemLoad
    fn parse_simdgroup_elem_load(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        let sm_vid = args.first().map(|a| self.parse_expr(a)).unwrap_or(0);
        let idx = args.get(1).map(|a| literal_u32(a)).unwrap_or(0);
        let result = self.alloc_vid();
        self.push_op(
            quote! {
                Op::SimdgroupElemLoad { value: ValueId::new(#sm_vid), index: #idx }
            },
            result,
        );
        result
    }

    /// `simdgroup_elem_store(sm, index, data)` → Op::SimdgroupElemStore (no result)
    fn parse_simdgroup_elem_store(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        let sm_vid = args.first().map(|a| self.parse_expr(a)).unwrap_or(0);
        let idx = args.get(1).map(|a| literal_u32(a)).unwrap_or(0);
        let data_vid = args.get(2).map(|a| self.parse_expr(a)).unwrap_or(0);
        self.push_op_no_result(quote! {
            Op::SimdgroupElemStore {
                value: ValueId::new(#sm_vid),
                index: #idx,
                data: ValueId::new(#data_vid),
            }
        });
        0
    }

    /// `simdgroup_load(frag, "tg_name", offset, stride)` /
    /// `simdgroup_load(frag, "tg_name", offset, stride, transpose)` →
    /// Op::SimdgroupLoad (no result). HW-fused 8×8 fragment load from
    /// threadgroup memory — emits a single MSL `simdgroup_load(...)`
    /// instruction that issues a coalesced 32-lane fetch with HW swizzle.
    /// Use in place of the per-lane scatter
    /// `simdgroup_elem_store(frag, idx, threadgroup_load("tg", off))`
    /// to dodge f16 TG-bank conflicts on 8×8 fragment fills.
    /// Args: frag value-id (ssa), tg name (str literal), offset (ssa),
    /// stride (u32 const literal — row stride in elements), optional
    /// `transpose` (bool literal — default `false`; pass `true` to swap
    /// the row/col axes of the loaded fragment, e.g. for a row-major
    /// `[N, K]` B tile read into the MMA's `[K, N]` operand layout).
    fn parse_simdgroup_load(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        let frag_vid = args.first().map(|a| self.parse_expr(a)).unwrap_or(0);
        let tg_name = string_lit_from_expr(args.get(1).unwrap_or(&&*call.func));
        let off_vid = args.get(2).map(|a| self.parse_expr(a)).unwrap_or(0);
        let stride = args.get(3).map(|a| literal_u32(a)).unwrap_or(0);
        let transpose = args
            .get(4)
            .map(|a| {
                if let Expr::Lit(lit) = *a
                    && let syn::Lit::Bool(b) = &lit.lit
                {
                    b.value
                } else {
                    false
                }
            })
            .unwrap_or(false);
        self.push_op_no_result(quote! {
            Op::SimdgroupLoad {
                dest: ValueId::new(#frag_vid),
                tg: #tg_name.to_string(),
                offset: ValueId::new(#off_vid),
                stride: #stride,
                transpose: #transpose,
            }
        });
        0
    }

    /// `simdgroup_matmul(a, b, c)` → Op::SimdgroupMatMul (c = a * b + c, no result)
    fn parse_simdgroup_matmul(&mut self, call: &ExprCall) -> u32 {
        let args: Vec<_> = call.args.iter().collect();
        let a_vid = args.first().map(|a| self.parse_expr(a)).unwrap_or(0);
        let b_vid = args.get(1).map(|a| self.parse_expr(a)).unwrap_or(0);
        let c_vid = args.get(2).map(|a| self.parse_expr(a)).unwrap_or(0);
        self.push_op_no_result(quote! {
            Op::SimdgroupMatMul {
                a: ValueId::new(#a_vid),
                b: ValueId::new(#b_vid),
                c: ValueId::new(#c_vid),
            }
        });
        0
    }

    /// `simd_lane_id()` → Op::SimdLaneId
    fn parse_simd_lane_id(&mut self, _call: &ExprCall) -> u32 {
        let result = self.alloc_vid();
        self.push_op(quote! { Op::SimdLaneId }, result);
        result
    }

    /// `simd_group_id()` → Op::SimdGroupId
    fn parse_simd_group_id(&mut self, _call: &ExprCall) -> u32 {
        let result = self.alloc_vid();
        self.push_op(quote! { Op::SimdGroupId }, result);
        result
    }

    /// `neg_infinity()` / `infinity()` → Op::Load with the MSL constant name.
    fn parse_special_const(&mut self, _call: &ExprCall, src: &str) -> u32 {
        let result = self.alloc_vid();
        let src = src.to_string();
        self.push_op(
            quote! {
                Op::Load { src: #src.to_string(), mask: None, other: None, indices: vec![] }
            },
            result,
        );
        result
    }
}

// ---- Helpers ----------------------------------------------------------------

fn expr_to_path_string(expr: &Expr) -> String {
    if let Expr::Path(path) = expr { path_to_string(&path.path) } else { String::new() }
}

fn path_to_string(path: &syn::Path) -> String {
    path.segments.iter().map(|seg| seg.ident.to_string()).collect::<Vec<_>>().join("::")
}

fn extract_turbofish_axis(expr: &Expr) -> Option<u32> {
    if let Expr::Path(path) = expr {
        for seg in &path.path.segments {
            if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                for arg in &args.args {
                    if let syn::GenericArgument::Type(syn::Type::Path(tp)) = arg
                        && let Some(last) = tp.path.segments.last()
                        && let Ok(n) = last.ident.to_string().parse::<u32>()
                    {
                        return Some(n);
                    }
                    if let syn::GenericArgument::Const(syn::Expr::Lit(lit)) = arg
                        && let syn::Lit::Int(n) = &lit.lit
                        && let Ok(val) = n.base10_parse::<u32>()
                    {
                        return Some(val);
                    }
                }
            }
        }
    }
    None
}

fn extract_turbofish_name(expr: &Expr) -> Option<String> {
    if let Expr::Path(path) = expr {
        for seg in &path.path.segments {
            if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                for arg in &args.args {
                    if let syn::GenericArgument::Type(syn::Type::Path(tp)) = arg
                        && let Some(last) = tp.path.segments.last()
                    {
                        return Some(last.ident.to_string());
                    }
                    if let syn::GenericArgument::Const(syn::Expr::Lit(lit)) = arg {
                        return Some(format!("{}", quote! { #lit }));
                    }
                }
            }
        }
    }
    None
}

/// Convert a type name string to DType tokens for `quote!`.
fn dtype_tokens_for_name(name: &str) -> proc_macro2::TokenStream {
    match name {
        "f32" | "float" => quote! { DType::F32 },
        "f16" | "half" => quote! { DType::F16 },
        "bf16" | "bfloat" => quote! { DType::BF16 },
        "u32" | "uint" => quote! { DType::U32 },
        "i32" | "int" => quote! { DType::I32 },
        _ => quote! { DType::F32 },
    }
}

/// Extract (raw_dtype_name, M, N) from a turbofish like `::<T, 8, 8>`.
/// Used by `simdgroup_alloc::<T, M, N>()` where T may be a concrete dtype
/// keyword (`f32`, `f16`, …) or a kernel generic type-var name resolved
/// at call site via the parser's `type_vars` table.
fn extract_turbofish_name_and_mn(expr: &Expr) -> Option<(String, u32, u32)> {
    if let Expr::Path(path) = expr {
        for seg in &path.path.segments {
            if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                let mut iter = args.args.iter();
                let dtype_name = iter.next().and_then(|arg| {
                    if let syn::GenericArgument::Type(syn::Type::Path(tp)) = arg
                        && let Some(last) = tp.path.segments.last()
                    {
                        Some(last.ident.to_string())
                    } else {
                        None
                    }
                });
                let m = iter.next().and_then(|arg| {
                    if let syn::GenericArgument::Const(syn::Expr::Lit(lit)) = arg
                        && let syn::Lit::Int(n) = &lit.lit
                        && let Ok(val) = n.base10_parse::<u32>()
                    {
                        Some(val)
                    } else {
                        None
                    }
                });
                let n = iter.next().and_then(|arg| {
                    if let syn::GenericArgument::Const(syn::Expr::Lit(lit)) = arg
                        && let syn::Lit::Int(n) = &lit.lit
                        && let Ok(val) = n.base10_parse::<u32>()
                    {
                        Some(val)
                    } else {
                        None
                    }
                });
                if let (Some(name), Some(mm), Some(nn)) = (dtype_name, m, n) {
                    return Some((name, mm, nn));
                }
            }
        }
    }
    None
}

/// Extract (dtype_tokens, M, N) from a turbofish like `::<f16, 8, 8>`.
/// Used by `simdgroup_alloc::<dtype, M, N>()`.
#[allow(dead_code)]
fn extract_turbofish_dtype_and_mn(expr: &Expr) -> Option<(proc_macro2::TokenStream, u32, u32)> {
    if let Expr::Path(path) = expr {
        for seg in &path.path.segments {
            if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                let mut iter = args.args.iter();
                let dtype = iter.next().and_then(|arg| {
                    if let syn::GenericArgument::Type(syn::Type::Path(tp)) = arg
                        && let Some(last) = tp.path.segments.last()
                    {
                        Some(dtype_tokens_for_name(&last.ident.to_string()))
                    } else {
                        None
                    }
                });
                let m = iter.next().and_then(|arg| {
                    if let syn::GenericArgument::Const(syn::Expr::Lit(lit)) = arg
                        && let syn::Lit::Int(n) = &lit.lit
                        && let Ok(val) = n.base10_parse::<u32>()
                    {
                        Some(val)
                    } else {
                        None
                    }
                });
                let n = iter.next().and_then(|arg| {
                    if let syn::GenericArgument::Const(syn::Expr::Lit(lit)) = arg
                        && let syn::Lit::Int(n) = &lit.lit
                        && let Ok(val) = n.base10_parse::<u32>()
                    {
                        Some(val)
                    } else {
                        None
                    }
                });
                if let (Some(dt), Some(mm), Some(nn)) = (dtype, m, n) {
                    return Some((dt, mm, nn));
                }
            }
        }
    }
    None
}

/// Extract a u32 from a literal expression (e.g. `0u32`, `1u32`).
fn literal_u32(expr: &Expr) -> u32 {
    if let Expr::Lit(lit) = expr {
        match &lit.lit {
            syn::Lit::Int(n) => n.base10_parse::<u32>().unwrap_or(0),
            syn::Lit::Float(f) => f.base10_parse::<f64>().map(|v| v as u32).unwrap_or(0),
            _ => 0,
        }
    } else {
        0
    }
}

/// Extract a string literal from an expression like `"my_name"`.
fn string_lit_from_expr(expr: &Expr) -> String {
    if let Expr::Lit(lit) = expr
        && let syn::Lit::Str(s) = &lit.lit
    {
        return s.value();
    }
    String::new()
}

/// Extract a usize literal from an optional expression like `9` or `9u32`.
/// Unwraps `Expr::Group` (the invisible delimiter `macro_rules!` wraps
/// captured fragments in) so callers like `threadgroup_alloc!(..., $size)`
/// see the underlying integer literal rather than the Group wrapping.
fn usize_lit_from_expr(expr: Option<&Expr>) -> usize {
    let Some(expr) = expr else { return 0 };
    let unwrapped: &Expr = match expr {
        Expr::Group(g) => &g.expr,
        other => other,
    };
    if let Expr::Lit(lit) = unwrapped
        && let syn::Lit::Int(n) = &lit.lit
    {
        return n.base10_parse::<usize>().unwrap_or(0);
    }
    0
}

#[cfg(test)]
mod tests {
    use syn::parse_quote;

    use super::*;

    #[test]
    fn parses_rust_range_loops() {
        let body: Block = parse_quote!({
            for i in 0..n {
                store(out[i], i);
            }
        });

        let tokens =
            DslBodyParser::parse(&body, &[String::from("out")], &[String::from("n")]).to_string();

        assert!(tokens.contains("Op :: Loop"), "{tokens}");
        assert!(tokens.contains("Op :: Const { value : 0"), "{tokens}");
        assert!(tokens.contains("Op :: Const { value : 1"), "{tokens}");
        assert!(tokens.contains("src : \"n\" . to_string ()"), "{tokens}");
    }

    #[test]
    fn unknown_calls_emit_compile_errors() {
        let body: Block = parse_quote!({
            let y = sine(x);
        });

        let tokens = DslBodyParser::parse(&body, &[], &[]).to_string();

        assert!(tokens.contains("compile_error"), "{tokens}");
        assert!(tokens.contains("unrecognized MetalTile DSL call"), "{tokens}");
        assert!(tokens.contains("sine"), "{tokens}");
    }

    #[test]
    fn parses_simdgroup_load_basic() {
        // `simdgroup_load(frag, "tg", off, stride)` — default transpose=false.
        let body: Block = parse_quote!({
            let f = simdgroup_alloc::<f16, 8, 8>();
            simdgroup_load(f, "ws", off, 36u32);
        });

        let tokens = DslBodyParser::parse(&body, &[], &[]).to_string();

        assert!(tokens.contains("Op :: SimdgroupLoad"), "{tokens}");
        assert!(tokens.contains("tg : \"ws\" . to_string ()"), "{tokens}");
        assert!(tokens.contains("stride : 36u32"), "{tokens}");
        assert!(tokens.contains("transpose : false"), "{tokens}");
    }

    #[test]
    fn parses_simdgroup_load_with_transpose() {
        // 5-arg form: `simdgroup_load(frag, "tg", off, stride, true)`.
        let body: Block = parse_quote!({
            let f = simdgroup_alloc::<f16, 8, 8>();
            simdgroup_load(f, "ws", off, 36u32, true);
        });

        let tokens = DslBodyParser::parse(&body, &[], &[]).to_string();

        assert!(tokens.contains("Op :: SimdgroupLoad"), "{tokens}");
        assert!(tokens.contains("transpose : true"), "{tokens}");
    }
}
