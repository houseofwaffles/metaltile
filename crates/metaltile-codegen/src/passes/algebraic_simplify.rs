//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Algebraic Simplification — pattern-matching rewrite system.
//!
//! Simplifies IR operations beyond what ConstFold handles: identity absorption,
//! idempotence, canonicalization, and algebraic rewrites.  After earlier passes
//! (CopyProp, ConstFold, CSE) have cleaned the IR, this pass catches residual
//! patterns that emerge from those transformations.
//!
//! ## Patterns
//!
//! ### Zero / One Absorption
//! - `x - x → Const(0)`
//! - `x / x → Const(1)`
//! - `x / Const(1) → x`
//! - `Const(0) - x → Neg(x)`
//! - `(-x) * (-y) → x * y`
//!
//! ### Select / Conditional Simplification
//! - `Select(true, a, b) → a`
//! - `Select(false, a, b) → b`
//! - `Select(cond, a, a) → a`
//! - `Select(Not(cond), a, b) → Select(cond, b, a)`
//!
//! ### Broadcast / Reshape Squashing
//! - `Broadcast(Broadcast(x)) → Broadcast(x)`
//! - `Reshape(Reshape(x)) → Reshape(x)`
//! - `Transpose(Transpose(x)) → x`
//! - `ExpandDims(Reshape(x)) → Reshape(x)`
//!
//! ### Min / Max Canonicalization
//! - `Max(x, x) → x`
//! - `Min(x, x) → x`
//!
//! ### Comparison Canonicalization
//! - `CmpLt(a, b) → CmpGt(b, a)`
//! - `CmpLe(a, b) → CmpGe(b, a)`
//! - `CmpEq(a, a) → Const(1)`
//! - `CmpNe(a, a) → Const(0)`
//!
//! ## Algorithm
//!
//! Iterates to fixpoint over each block.  Each iteration collects rewrites
//! (new ops or ValueId replacements), applies them, and stops when stable.
//! This is a local (single-block) algorithm; inter-block algebraic identities
//! require a global value-numbering framework.
//!
//! ## Limitations
//!
//! - Block-local only: algebraic identities spanning multiple blocks are not caught.
//! - Pattern set is intentionally conservative to avoid IR bloat.
//! - No expression reassociation (e.g. `(a + b) + c → a + (b + c)`); deferred
//!   to a future canonicalization pass.
//!
//! ## References
//! - Cocke & Schwartz (1970), "Programming Languages and their Compilers",
//!   Courant Institute.  Early work on algebraic simplification via value numbering.
//! - Aho, Lam, Sethi & Ullman (2006), "Compilers: Principles, Techniques, and
//!   Tools", 2nd ed., §8.4–8.5.  Classic treatment of algebraic identities and
//!   reduction in strength.

use std::collections::{BTreeMap, BTreeSet};

use metaltile_core::ir::{BinOpKind, Block, BlockId, Kernel, Op, UnaryOpKind, ValueId};

use super::remap;
use crate::error::{Error, Result};

pub struct AlgebraicSimplifyPass;

impl super::Pass for AlgebraicSimplifyPass {
    fn name(&self) -> &str { "algebraic_simplify" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        let block_ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();

        for bid in &block_ids {
            let mut block =
                kernel.blocks.remove(bid).ok_or_else(|| Error::BlockNotFound(bid.as_u32()))?;
            simplify_block_fixpoint(&mut block);
            kernel.blocks.insert(*bid, block);
        }

        simplify_block_fixpoint(&mut kernel.body);

        super::dead_value_elim::eliminate_dead_values(kernel)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Block-level fixpoint
// ---------------------------------------------------------------------------

/// Resolve transitive replacement chains: {v2→v1, v1→v0} becomes {v2→v0, v1→v0}.
fn resolve_transitive(map: &BTreeMap<ValueId, ValueId>) -> BTreeMap<ValueId, ValueId> {
    let mut resolved = BTreeMap::new();
    for (&key, &val) in map.iter() {
        let mut terminal = val;
        let mut visited = BTreeSet::new();
        visited.insert(key);
        while let Some(&next) = map.get(&terminal) {
            if !visited.insert(terminal) {
                break; // cycle detected
            }
            terminal = next;
        }
        resolved.insert(key, terminal);
    }
    resolved
}

fn simplify_block_fixpoint(block: &mut Block) {
    loop {
        if !simplify_block_once(block) {
            break;
        }
    }
}

fn simplify_block_once(block: &mut Block) -> bool {
    let n = block.ops.len();
    let mut const_overwrites: Vec<(usize, i64)> = Vec::new();
    let mut op_replacements: Vec<(usize, Op)> = Vec::new();
    let mut vid_replacements: BTreeMap<ValueId, ValueId> = BTreeMap::new();

    // Build a map for peephole lookups.
    let vid_to_op_pos: BTreeMap<ValueId, usize> = block
        .results
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.and_then(|v| Some((v, i))))
        .collect();

    for i in 0..n {
        if let Some(result) = try_simplify(&block.ops[i], i, block, &vid_to_op_pos) {
            match result {
                SimpResult::Const(v) => const_overwrites.push((i, v)),
                SimpResult::ReplaceWith(op) => op_replacements.push((i, op)),
                SimpResult::ReplaceWithVid(new_vid) => {
                    if let Some(Some(old_vid)) = block.results.get(i) {
                        vid_replacements.insert(*old_vid, new_vid);
                    }
                },
            }
        }
    }

    // Prune vid_replacements that have no actual uses (avoids spurious infinite-loop progress).
    if !vid_replacements.is_empty() {
        let any_used = block
            .ops
            .iter()
            .any(|op| remap::op_value_refs(op).iter().any(|v| vid_replacements.contains_key(v)));
        if !any_used {
            vid_replacements.clear();
        }
    }

    if const_overwrites.is_empty() && op_replacements.is_empty() && vid_replacements.is_empty() {
        return false;
    }

    // Apply op replacements and const overwrites.
    for (idx, new_val) in const_overwrites {
        block.ops[idx] = Op::Const { value: new_val };
    }
    for (idx, new_op) in &op_replacements {
        block.ops[*idx] = new_op.clone();
    }

    // Resolve transitive replacement chains: v2→v1→v0 becomes v2→v0.
    let vid_replacements = resolve_transitive(&vid_replacements);

    // Remap ValueIds in all ops in the block.
    for op in block.ops.iter_mut() {
        remap_values_in_op(op, &vid_replacements);
    }

    // Remove dead ops whose results were redirected via ReplaceWithVid.
    // Without this, the same pattern re-matches on the next iteration,
    // producing the same replacement and causing an infinite fixpoint loop.
    let dead_vids: BTreeSet<ValueId> = vid_replacements.keys().copied().collect();
    if !dead_vids.is_empty() {
        let mut new_ops = Vec::new();
        let mut new_results = Vec::new();
        for (i, op) in block.ops.iter().enumerate() {
            let is_dead =
                block.results.get(i).is_some_and(|r| r.is_some_and(|v| dead_vids.contains(&v)));
            if !is_dead {
                new_ops.push(op.clone());
                new_results.push(block.results[i]);
            }
        }
        block.ops = new_ops;
        block.results = new_results;
    }

    true
}

// ---------------------------------------------------------------------------
// Simplification result
// ---------------------------------------------------------------------------

enum SimpResult {
    Const(i64),
    ReplaceWith(Op),
    ReplaceWithVid(ValueId),
}

// ---------------------------------------------------------------------------
// Main pattern matcher
// ---------------------------------------------------------------------------

fn try_simplify(
    op: &Op,
    pos: usize,
    block: &Block,
    vid_to_pos: &BTreeMap<ValueId, usize>,
) -> Option<SimpResult> {
    match op {
        // ---- BinOp patterns ----
        Op::BinOp { op: kind, lhs, rhs } =>
            simplify_binop(*kind, *lhs, *rhs, pos, block, vid_to_pos),

        // ---- Select patterns ----
        Op::Select { cond, on_true, on_false } =>
            simplify_select(*cond, *on_true, *on_false, block, vid_to_pos),

        // ---- Broadcast squashing ----
        Op::Broadcast { value, shape } => {
            if shape.rank() == 1
                && matches!(shape.dim(0), Some(metaltile_core::shape::Dim::Known(1)))
            {
                // Already a scalar broadcast — skip.
                return None;
            }
            if let Some((inner_pos, Op::Broadcast { .. })) =
                get_defining_op(*value, block, vid_to_pos)
            {
                // Broadcast(Broadcast(x)) → inner Broadcast
                let inner_op = &block.ops[inner_pos];
                Some(SimpResult::ReplaceWith(inner_op.clone()))
            } else {
                None
            }
        },

        // ---- Transpose squashing ----
        Op::Transpose { value } => {
            if let Some((_inner_pos, Op::Transpose { value: inner_val })) =
                get_defining_op(*value, block, vid_to_pos)
            {
                // Transpose(Transpose(x)) → x
                Some(SimpResult::ReplaceWithVid(*inner_val))
            } else {
                None
            }
        },

        // ---- Reshape squashing ----
        Op::Reshape { value, .. } => {
            if let Some((_inner_pos, Op::Reshape { value: inner_val, shape: _ })) =
                get_defining_op(*value, block, vid_to_pos)
            {
                // Reshape(Reshape(x)) → inner Reshape (shape already correct from outer)
                Some(SimpResult::ReplaceWithVid(*inner_val))
            } else {
                None
            }
        },

        // ---- ExpandDims(Reshape(x)) → Reshape(x) ----
        Op::ExpandDims { value, .. } => {
            if let Some((_inner_pos, Op::Reshape { value: inner_val, .. })) =
                get_defining_op(*value, block, vid_to_pos)
            {
                Some(SimpResult::ReplaceWithVid(*inner_val))
            } else {
                None
            }
        },

        _ => None,
    }
}

fn simplify_binop(
    kind: BinOpKind,
    lhs: ValueId,
    rhs: ValueId,
    _pos: usize,
    block: &Block,
    vid_to_pos: &BTreeMap<ValueId, usize>,
) -> Option<SimpResult> {
    let lv = find_const_in_block(block, lhs);
    let rv = find_const_in_block(block, rhs);

    match kind {
        BinOpKind::Sub => {
            if lhs == rhs {
                // x - x → 0
                Some(SimpResult::Const(0))
            } else if lv == Some(0) {
                // 0 - x → Neg(x)
                Some(SimpResult::ReplaceWith(Op::UnaryOp { op: UnaryOpKind::Neg, value: rhs }))
            } else {
                None
            }
        },

        BinOpKind::Div => {
            if lhs == rhs {
                // x / x → 1
                Some(SimpResult::Const(1))
            } else if rv == Some(1) {
                // x / 1 → x
                Some(SimpResult::ReplaceWithVid(lhs))
            } else {
                None
            }
        },

        BinOpKind::Mul => {
            // (-x) * (-y) → x * y  (sign cancellation)
            let neg_x = get_neg_arg(lhs, block, vid_to_pos);
            let neg_y = get_neg_arg(rhs, block, vid_to_pos);
            if let (Some(inner_x), Some(inner_y)) = (neg_x, neg_y) {
                Some(SimpResult::ReplaceWith(Op::BinOp {
                    op: BinOpKind::Mul,
                    lhs: inner_x,
                    rhs: inner_y,
                }))
            } else {
                None
            }
        },

        BinOpKind::Min | BinOpKind::Max => {
            if lhs == rhs {
                // Max(x, x) → x, Min(x, x) → x
                Some(SimpResult::ReplaceWithVid(lhs))
            } else {
                None
            }
        },

        BinOpKind::CmpLt => {
            // CmpLt(a, b) → CmpGt(b, a)
            Some(SimpResult::ReplaceWith(Op::BinOp { op: BinOpKind::CmpGt, lhs: rhs, rhs: lhs }))
        },

        BinOpKind::CmpLe => {
            // CmpLe(a, b) → CmpGe(b, a)
            Some(SimpResult::ReplaceWith(Op::BinOp { op: BinOpKind::CmpGe, lhs: rhs, rhs: lhs }))
        },

        BinOpKind::CmpEq => {
            if lhs == rhs {
                // CmpEq(a, a) → Const(1)
                Some(SimpResult::Const(1))
            } else {
                None
            }
        },

        BinOpKind::CmpNe => {
            if lhs == rhs {
                // CmpNe(a, a) → Const(0)
                Some(SimpResult::Const(0))
            } else {
                None
            }
        },

        _ => None,
    }
}

fn simplify_select(
    cond: ValueId,
    on_true: ValueId,
    on_false: ValueId,
    block: &Block,
    vid_to_pos: &BTreeMap<ValueId, usize>,
) -> Option<SimpResult> {
    // Select(true, a, b) → a
    if let Some(1) = find_const_in_block(block, cond) {
        return Some(SimpResult::ReplaceWithVid(on_true));
    }
    // Select(false, a, b) → b
    if let Some(0) = find_const_in_block(block, cond) {
        return Some(SimpResult::ReplaceWithVid(on_false));
    }
    // Select(cond, a, a) → a
    if on_true == on_false {
        return Some(SimpResult::ReplaceWithVid(on_true));
    }
    // Select(Not(cond), a, b) → Select(cond, b, a)
    if let Some(inner) = get_not_arg(cond, block, vid_to_pos) {
        return Some(SimpResult::ReplaceWith(Op::Select {
            cond: inner,
            on_true: on_false,
            on_false: on_true,
        }));
    }
    None
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn find_const_in_block(block: &Block, vid: ValueId) -> Option<i64> {
    for (i, op) in block.ops.iter().enumerate() {
        if block.results.get(i) == Some(&Some(vid))
            && let Op::Const { value } = op
        {
            return Some(*value);
        }
    }
    None
}

/// Get the defining op for a ValueId. Returns (position, op) if definition is in this block.
fn get_defining_op<'a>(
    vid: ValueId,
    block: &'a Block,
    vid_to_pos: &BTreeMap<ValueId, usize>,
) -> Option<(usize, &'a Op)> {
    let &pos = vid_to_pos.get(&vid)?;
    Some((pos, &block.ops[pos]))
}

/// If `vid` is defined by `UnaryOp(Neg, inner)`, return `Some(inner)`.
fn get_neg_arg(
    vid: ValueId,
    block: &Block,
    vid_to_pos: &BTreeMap<ValueId, usize>,
) -> Option<ValueId> {
    let (_pos, op) = get_defining_op(vid, block, vid_to_pos)?;
    if let Op::UnaryOp { op: UnaryOpKind::Neg, value } = op { Some(*value) } else { None }
}

/// If `vid` is defined by a logical NOT (CmpEq(x, 0) or Xor(x, 1)), return the inner condition.
fn get_not_arg(
    vid: ValueId,
    block: &Block,
    vid_to_pos: &BTreeMap<ValueId, usize>,
) -> Option<ValueId> {
    let (_pos, op) = get_defining_op(vid, block, vid_to_pos)?;
    match op {
        Op::BinOp { op: BinOpKind::CmpEq, lhs, rhs } => {
            // CmpEq(cond, 0) → logical NOT of cond
            if find_const_in_block(block, *rhs) == Some(0) {
                return Some(*lhs);
            }
            if find_const_in_block(block, *lhs) == Some(0) {
                return Some(*rhs);
            }
            None
        },
        Op::BinOp { op: BinOpKind::Xor, lhs, rhs } => {
            // Xor(cond, 1) → logical NOT of cond
            if find_const_in_block(block, *rhs) == Some(1) {
                return Some(*lhs);
            }
            if find_const_in_block(block, *lhs) == Some(1) {
                return Some(*rhs);
            }
            None
        },
        _ => None,
    }
}

/// Remap all ValueId references in an op.
fn remap_values_in_op(op: &mut Op, map: &BTreeMap<ValueId, ValueId>) {
    op.for_each_value_id_mut(&mut |v| {
        if let Some(&nv) = map.get(v) {
            *v = nv;
        }
    });
}

#[cfg(test)]
mod tests {
    use metaltile_core::ir::BinOpKind;

    use super::*;
    use crate::passes::Pass;

    #[test]
    fn x_minus_x_is_zero() {
        let mut k = Kernel::new("x_minus_x");
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Sub, lhs: ValueId::new(0), rhs: ValueId::new(0) },
            ValueId::new(1),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(1),
            mask: None,
        });
        AlgebraicSimplifyPass.run(&mut k).unwrap();
        // x - x → Const(0)
        let has_zero = k.body.ops.iter().any(|op| matches!(op, Op::Const { value: 0 }));
        assert!(has_zero, "x-x should become Const(0)");
    }

    #[test]
    fn x_div_x_is_one() {
        let mut k = Kernel::new("x_div_x");
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Div, lhs: ValueId::new(0), rhs: ValueId::new(0) },
            ValueId::new(1),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(1),
            mask: None,
        });
        AlgebraicSimplifyPass.run(&mut k).unwrap();
        let has_one = k.body.ops.iter().any(|op| matches!(op, Op::Const { value: 1 }));
        assert!(has_one, "x/x should become Const(1)");
    }

    #[test]
    fn x_div_1_is_x() {
        let mut k = Kernel::new("x_div_1");
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(1));
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Div, lhs: ValueId::new(0), rhs: ValueId::new(1) },
            ValueId::new(2),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(2),
            mask: None,
        });
        AlgebraicSimplifyPass.run(&mut k).unwrap();
        // x/1 → x — uses of v2 redirected to v0.
        let stores: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::Store { .. })).collect();
        if let Op::Store { value, .. } = stores[0] {
            assert_eq!(value.as_u32(), 0, "store should reference v0 directly after x/1 fold");
        }
    }

    #[test]
    fn zero_minus_x_is_neg() {
        let mut k = Kernel::new("zero_minus_x");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(1));
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Sub, lhs: ValueId::new(0), rhs: ValueId::new(1) },
            ValueId::new(2),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(2),
            mask: None,
        });
        AlgebraicSimplifyPass.run(&mut k).unwrap();
        // 0 - x → Neg(x)
        let has_neg = k.body.ops.iter().any(|op| {
            matches!(op, Op::UnaryOp { op: UnaryOpKind::Neg, value } if *value == ValueId::new(1))
        });
        assert!(has_neg, "0-x should become Neg(x)");
    }

    #[test]
    fn max_x_x_is_x() {
        let mut k = Kernel::new("max_x_x");
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Max, lhs: ValueId::new(0), rhs: ValueId::new(0) },
            ValueId::new(1),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(1),
            mask: None,
        });
        AlgebraicSimplifyPass.run(&mut k).unwrap();
        let stores: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::Store { .. })).collect();
        if let Op::Store { value, .. } = stores[0] {
            assert_eq!(value.as_u32(), 0, "Max(x,x) should redirect to x");
        }
    }

    #[test]
    fn cmp_lt_canonicalizes_to_cmp_gt() {
        let mut k = Kernel::new("cmp_lt");
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::ProgramId { axis: 1 }, ValueId::new(1));
        k.body.push_op(
            Op::BinOp { op: BinOpKind::CmpLt, lhs: ValueId::new(0), rhs: ValueId::new(1) },
            ValueId::new(2),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(2),
            mask: None,
        });
        AlgebraicSimplifyPass.run(&mut k).unwrap();
        let has_cmp_gt = k.body.ops.iter().any(|op| {
            matches!(op, Op::BinOp { op: BinOpKind::CmpGt, lhs, rhs }
                if *lhs == ValueId::new(1) && *rhs == ValueId::new(0))
        });
        assert!(has_cmp_gt, "CmpLt(a,b) should become CmpGt(b,a)");
    }

    #[test]
    fn cmp_eq_same_is_one() {
        let mut k = Kernel::new("cmp_eq_same");
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(
            Op::BinOp { op: BinOpKind::CmpEq, lhs: ValueId::new(0), rhs: ValueId::new(0) },
            ValueId::new(1),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(1),
            mask: None,
        });
        AlgebraicSimplifyPass.run(&mut k).unwrap();
        let has_one = k.body.ops.iter().any(|op| matches!(op, Op::Const { value: 1 }));
        assert!(has_one, "CmpEq(a,a) should be Const(1)");
    }

    #[test]
    fn select_true_picks_on_true() {
        let mut k = Kernel::new("select_true");
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(0)); // true
        k.body.push_op(Op::Const { value: 42 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 99 }, ValueId::new(2));
        k.body.push_op(
            Op::Select {
                cond: ValueId::new(0),
                on_true: ValueId::new(1),
                on_false: ValueId::new(2),
            },
            ValueId::new(3),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(3),
            mask: None,
        });
        AlgebraicSimplifyPass.run(&mut k).unwrap();
        let stores: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::Store { .. })).collect();
        if let Op::Store { value, .. } = stores[0] {
            assert_eq!(value.as_u32(), 1, "Select(true,a,b) should pick a (v1=42)");
        }
    }

    #[test]
    fn select_same_both_sides_is_identity() {
        let mut k = Kernel::new("select_same");
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 42 }, ValueId::new(1));
        k.body.push_op(
            Op::Select {
                cond: ValueId::new(0),
                on_true: ValueId::new(1),
                on_false: ValueId::new(1),
            },
            ValueId::new(2),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(2),
            mask: None,
        });
        AlgebraicSimplifyPass.run(&mut k).unwrap();
        let stores: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::Store { .. })).collect();
        if let Op::Store { value, .. } = stores[0] {
            assert_eq!(value.as_u32(), 1, "Select(cond,a,a) should redirect to a");
        }
    }

    #[test]
    fn transpose_transpose_is_identity() {
        let mut k = Kernel::new("transpose_transpose");
        k.body.push_op(Op::Const { value: 42 }, ValueId::new(0));
        k.body.push_op(Op::Transpose { value: ValueId::new(0) }, ValueId::new(1));
        k.body.push_op(Op::Transpose { value: ValueId::new(1) }, ValueId::new(2));
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(2),
            mask: None,
        });
        AlgebraicSimplifyPass.run(&mut k).unwrap();
        // Transpose(Transpose(x)) → x
        let stores: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::Store { .. })).collect();
        if let Op::Store { value, .. } = stores[0] {
            assert_eq!(value.as_u32(), 0, "Transpose(Transpose(x)) should point to x");
        }
    }

    #[test]
    fn neg_neg_mul_cancels() {
        let mut k = Kernel::new("neg_neg_mul");
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body
            .push_op(Op::UnaryOp { op: UnaryOpKind::Neg, value: ValueId::new(0) }, ValueId::new(1));
        k.body.push_op(Op::ProgramId { axis: 1 }, ValueId::new(2));
        k.body
            .push_op(Op::UnaryOp { op: UnaryOpKind::Neg, value: ValueId::new(2) }, ValueId::new(3));
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Mul, lhs: ValueId::new(1), rhs: ValueId::new(3) },
            ValueId::new(4),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(4),
            mask: None,
        });
        AlgebraicSimplifyPass.run(&mut k).unwrap();
        // (-x) * (-y) → x * y
        let has_plain_mul = k.body.ops.iter().any(|op| {
            matches!(op, Op::BinOp { op: BinOpKind::Mul, lhs, rhs }
                if *lhs == ValueId::new(0) && *rhs == ValueId::new(2))
        });
        assert!(has_plain_mul, "(-x)*(-y) should become x*y");
    }
}
