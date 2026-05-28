//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Copy Propagation — forward source values through identity operations.
//!
//! Eliminates no-op operations and propagates the underlying source value
//! through chains of copies and identity casts.  Shortens use-def chains so
//! downstream passes (CSE, Fusion, AlgebraicSimplify) see the "real" values.
//!
//! ## Identity Patterns
//! - `Cast(dtype, x)` → `x`  when `x` is already that dtype
//! - `Broadcast(x, [1])` → `x`  when broadcasting a scalar with shape [1]
//! - `Reshape(x, s)` → `x`  when shapes are identical
//! - `Select(cond, x, x)` → `x`  (also in AlgebraicSimplify, but cheap to re-check)
//!
//! ## Copy Forwarding
//! When an op's result is used through a chain of identity operations,
//! forward the source value through. The downstream CSE pass then eliminates the
//! now-dead identity ops.
//!
//! ## Algorithm
//!
//! Iterates to fixpoint.  Each iteration:
//! 1. Find identity ops (result == source).
//! 2. Replace all uses of the identity result with the source ValueId.
//! 3. DCE cleans up the dead identity ops (ran after this pass).
//!
//! ## References
//! - Aho, Lam, Sethi & Ullman (2006), "Compilers: Principles, Techniques, and
//!   Tools", 2nd ed., §9.1.1.  Canonical treatment of copy propagation.
//! - Wegman & Zadeck (1991), "Constant propagation with conditional branches",
//!   ACM TOPLAS 13(2):181–210.  Sparse conditional constant propagation framework
//!   that subsumes copy propagation.

use std::collections::{BTreeMap, BTreeSet};

use metaltile_core::{
    dtype::DType,
    ir::{Block, BlockId, Kernel, Op, ValueId},
};

use super::remap;
use crate::error::{Error, Result};

pub struct CopyPropPass;

impl super::Pass for CopyPropPass {
    fn name(&self) -> &str { "copy_prop" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        let block_ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();

        for bid in &block_ids {
            let mut block =
                kernel.blocks.remove(bid).ok_or_else(|| Error::BlockNotFound(bid.as_u32()))?;
            copy_prop_block_fixpoint(&mut block);
            kernel.blocks.insert(*bid, block);
        }

        copy_prop_block_fixpoint(&mut kernel.body);

        super::dead_value_elim::eliminate_dead_values(kernel)?;
        Ok(())
    }
}

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

fn copy_prop_block_fixpoint(block: &mut Block) {
    loop {
        if !copy_prop_block_once(block) {
            break;
        }
    }
}

fn copy_prop_block_once(block: &mut Block) -> bool {
    let n = block.ops.len();
    let mut vid_replacements: BTreeMap<ValueId, ValueId> = BTreeMap::new();

    for i in 0..n {
        let op = &block.ops[i];
        if let Some(source_vid) = is_identity(op, block)
            && let Some(Some(result_vid)) = block.results.get(i)
        {
            vid_replacements.insert(*result_vid, source_vid);
        }
    }

    if vid_replacements.is_empty() {
        return false;
    }

    // Resolve transitive replacement chains: v2→v1→v0 becomes v2→v0.
    let vid_replacements = resolve_transitive(&vid_replacements);

    // Remap ValueIds in all ops.
    for op in block.ops.iter_mut() {
        remap::remap_value_ids(op, &vid_replacements);
    }

    // Remove dead ops whose results were redirected via identity propagation.
    // Without this, the same identity pattern re-matches on the next iteration,
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

/// Check if an op is an identity (output equals one of its inputs in all cases).
fn is_identity(op: &Op, _block: &Block) -> Option<ValueId> {
    match op {
        // Cast(float, x) → x  when x is already float
        Op::Cast { value, dtype } => {
            let inferred = infer_value_dtype(*value, _block);
            if inferred == Some(*dtype) { Some(*value) } else { None }
        },

        // Broadcast(x, [1]) → x  — broadcasting a scalar by shape [1] is a no-op
        Op::Broadcast { value, shape } => {
            if shape.rank() == 1
                && matches!(shape.dim(0), Some(metaltile_core::shape::Dim::Known(1)))
            {
                Some(*value)
            } else {
                None
            }
        },

        // Reshape(x, s) → x  when shape s has the same total elements and same layout
        // For now: only when the value is already a scalar or single-element tile.
        Op::Reshape { value, shape } => {
            if shape.rank() == 1
                && matches!(shape.dim(0), Some(metaltile_core::shape::Dim::Known(1)))
            {
                // Reshape to [1] is identity for scalars
                Some(*value)
            } else {
                None
            }
        },

        // Select(cond, x, x) → x  — same value both sides
        Op::Select { on_true, on_false, .. } =>
            if on_true == on_false {
                Some(*on_true)
            } else {
                None
            },

        // ExpandDims with shape [1] is effectively an identity for a scalar
        // (handled by Reshape already; but cover base case)
        _ => None,
    }
}

/// Naive dtype inference for a value.  Only detects `Cast` and `Const` patterns.
fn infer_value_dtype(vid: ValueId, block: &Block) -> Option<DType> {
    for (i, op) in block.ops.iter().enumerate() {
        if block.results.get(i) != Some(&Some(vid)) {
            continue;
        }
        match op {
            Op::Cast { dtype, .. } => return Some(*dtype),
            Op::Const { .. } =>
            // Constants are integers; they'll be cast to target dtype at use.
            {
                return None;
            },
            Op::Zeros { dtype, .. } | Op::Splat { dtype, .. } => return Some(*dtype),
            Op::Load { .. } => return None, // dtype comes from param
            _ => return None,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use metaltile_core::{
        dtype::DType,
        shape::{Dim, Shape},
    };

    use super::*;
    use crate::passes::Pass;

    #[test]
    fn eliminates_cast_of_same_dtype() {
        let mut k = Kernel::new("cast_id");
        // Create a float-producing op
        k.body.push_op(Op::Zeros { dtype: DType::F32, shape: Shape::scalar() }, ValueId::new(0));
        k.body.push_op(Op::Cast { value: ValueId::new(0), dtype: DType::F32 }, ValueId::new(1));
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(1),
            mask: None,
        });
        CopyPropPass.run(&mut k).unwrap();
        // Cast(f32, f32_val) → f32_val; uses of v1 redirected to v0.
        let stores: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::Store { .. })).collect();
        if let Op::Store { value, .. } = stores[0] {
            assert_eq!(value.as_u32(), 0, "Cast(f32, f32) should redirect to original");
        }
    }

    #[test]
    fn eliminates_broadcast_scalar_shape1() {
        let mut k = Kernel::new("broadcast_id");
        k.body.push_op(Op::Const { value: 5 }, ValueId::new(0));
        k.body.push_op(
            Op::Broadcast { value: ValueId::new(0), shape: Shape::new([Dim::Known(1)]) },
            ValueId::new(1),
        );
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(1),
            mask: None,
        });
        CopyPropPass.run(&mut k).unwrap();
        let stores: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::Store { .. })).collect();
        if let Op::Store { value, .. } = stores[0] {
            assert_eq!(value.as_u32(), 0, "Broadcast(x, [1]) should redirect to x");
        }
    }

    #[test]
    fn eliminates_select_with_same_branches() {
        let mut k = Kernel::new("select_id");
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
        CopyPropPass.run(&mut k).unwrap();
        let stores: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::Store { .. })).collect();
        if let Op::Store { value, .. } = stores[0] {
            assert_eq!(value.as_u32(), 1, "Select(cond,a,a) should redirect to a");
        }
    }

    #[test]
    fn preserves_non_identity_cast() {
        let mut k = Kernel::new("cast_real");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0)); // i32
        k.body.push_op(Op::Cast { value: ValueId::new(0), dtype: DType::F32 }, ValueId::new(1));
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(1),
            mask: None,
        });
        CopyPropPass.run(&mut k).unwrap();
        // Cast(i32→f32) is NOT an identity (dtype inferred from Const is None), should be kept.
        let has_cast = k.body.ops.iter().any(|op| matches!(op, Op::Cast { .. }));
        assert!(has_cast, "Cast to different dtype should be preserved");
    }

    #[test]
    fn fixpoint_propagates_through_chain() {
        let mut k = Kernel::new("copy_chain");
        k.body.push_op(Op::Zeros { dtype: DType::F32, shape: Shape::scalar() }, ValueId::new(0));
        k.body.push_op(Op::Cast { value: ValueId::new(0), dtype: DType::F32 }, ValueId::new(1));
        k.body.push_op(Op::Cast { value: ValueId::new(1), dtype: DType::F32 }, ValueId::new(2));
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![],
            value: ValueId::new(2),
            mask: None,
        });
        CopyPropPass.run(&mut k).unwrap();
        // Both Casts are identities → all redirect to v0.
        let stores: Vec<_> =
            k.body.ops.iter().filter(|op| matches!(op, Op::Store { .. })).collect();
        if let Op::Store { value, .. } = stores[0] {
            assert_eq!(value.as_u32(), 0, "chain of identity casts should propagate to source");
        }
    }
}
