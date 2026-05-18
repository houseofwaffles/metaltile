//! Fused elementwise chain emission.
//!
//! Handles `Op::FusedElementwise` lowering: walks the fused op tree and emits
//! a single compound MSL expression without intermediate temporaries.

use std::collections::BTreeMap;

use metaltile_core::ir::{BinOpKind, Block, Kernel, Op, ValueId};

use super::MslGenerator;
use crate::passes::{fusion::SUB_OP_FLAG, type_check::TypeEnv};

impl MslGenerator {
    /// Emit a fused expression rooted at `fused_ops[idx]`, returning the MSL expression string.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_fused_expr(
        &self,
        out: &mut String,
        pad: &str,
        fused_ops: &[Op],
        block: &Block,
        kernel: &Kernel,
        type_env: &TypeEnv,
        extra_names: &BTreeMap<ValueId, String>,
        resolved_vid: ValueId,
        idx: usize,
    ) -> String {
        let op = &fused_ops[idx];

        match op {
            Op::BinOp { op: kind, lhs, rhs } => {
                let l = self.fused_operand(
                    out,
                    pad,
                    fused_ops,
                    *lhs,
                    block,
                    kernel,
                    type_env,
                    extra_names,
                    resolved_vid,
                );
                let r = self.fused_operand(
                    out,
                    pad,
                    fused_ops,
                    *rhs,
                    block,
                    kernel,
                    type_env,
                    extra_names,
                    resolved_vid,
                );
                match kind {
                    BinOpKind::Max
                    | BinOpKind::Min
                    | BinOpKind::Pow
                    | BinOpKind::ATan2
                    | BinOpKind::Rem => format!("{}({l}, {r})", kind.msl_symbol()),
                    BinOpKind::And => format!("({l} && {r})"),
                    BinOpKind::Or => format!("({l} || {r})"),
                    BinOpKind::Xor => format!("((bool){l} != (bool){r})"),
                    BinOpKind::BitAnd => format!("({l} & {r})"),
                    BinOpKind::BitOr => format!("({l} | {r})"),
                    BinOpKind::BitXor => format!("({l} ^ {r})"),
                    BinOpKind::Shl => format!("({l} << {r})"),
                    BinOpKind::Shr => format!("({l} >> {r})"),
                    _ => format!("({l} {} {r})", kind.msl_symbol()),
                }
            },
            Op::UnaryOp { op: kind, value } => {
                let v = self.fused_operand(
                    out,
                    pad,
                    fused_ops,
                    *value,
                    block,
                    kernel,
                    type_env,
                    extra_names,
                    resolved_vid,
                );
                kind.msl_emit(&v)
            },
            Op::Activation { kind, value } => {
                let v = self.fused_operand(
                    out,
                    pad,
                    fused_ops,
                    *value,
                    block,
                    kernel,
                    type_env,
                    extra_names,
                    resolved_vid,
                );
                format!("{}({v})", kind.msl_fn())
            },
            Op::Cast { dtype, value } => {
                let v = self.fused_operand(
                    out,
                    pad,
                    fused_ops,
                    *value,
                    block,
                    kernel,
                    type_env,
                    extra_names,
                    resolved_vid,
                );
                self.emit_cast_expr(*dtype, &v)
            },
            Op::Select { cond, on_true, on_false } => {
                let c = self.fused_operand(
                    out,
                    pad,
                    fused_ops,
                    *cond,
                    block,
                    kernel,
                    type_env,
                    extra_names,
                    resolved_vid,
                );
                let t = self.fused_operand(
                    out,
                    pad,
                    fused_ops,
                    *on_true,
                    block,
                    kernel,
                    type_env,
                    extra_names,
                    resolved_vid,
                );
                let f = self.fused_operand(
                    out,
                    pad,
                    fused_ops,
                    *on_false,
                    block,
                    kernel,
                    type_env,
                    extra_names,
                    resolved_vid,
                );
                format!("select({f}, {t}, bool({c}))")
            },
            Op::Broadcast { .. } => {
                // Broadcast in a fused chain is unusual; fall back.
                "0 /* broadcast-in-fused */".into()
            },
            _ => {
                let vid = super::helpers::op_to_vid(op);
                self.vname(Some(vid), block, extra_names)
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fused_operand(
        &self,
        out: &mut String,
        pad: &str,
        fused_ops: &[Op],
        vid: ValueId,
        block: &Block,
        kernel: &Kernel,
        type_env: &TypeEnv,
        extra_names: &BTreeMap<ValueId, String>,
        resolved_vid: ValueId,
    ) -> String {
        if vid.as_u32() & SUB_OP_FLAG != 0 {
            let sub_idx = (vid.as_u32() & !SUB_OP_FLAG) as usize;
            if sub_idx < fused_ops.len() {
                self.emit_fused_expr(
                    out,
                    pad,
                    fused_ops,
                    block,
                    kernel,
                    type_env,
                    extra_names,
                    resolved_vid,
                    sub_idx,
                )
            } else {
                "0 /* bad sub-op ref */".into()
            }
        } else {
            self.vname(Some(vid), block, extra_names)
        }
    }
}
