//! Loop Invariant Code Motion — hoist loop-invariant computations.
//!
//! Identifies computations inside loop bodies whose operands are all defined
//! outside the loop, and hoists them to before the loop.  This eliminates
//! redundant re-computation of index arithmetic and const-buffer loads
//! across loop iterations, reducing both instruction count and register
//! pressure within the loop.
//!
//! ## Algorithm
//!
//! For each `Op::Loop` in every block:
//! 1. Build the initial invariant set: all ValueIds defined in the parent block
//!    before the loop (or in ancestor blocks).
//! 2. Iterate to fixpoint: any op in the loop body whose operands are all
//!    invariant AND which has no side effects is marked as hoistable.
//! 3. Hoist: remove hoistable ops from the loop body and insert them before the
//!    loop in the parent block, respecting topological order among hoisted ops.
//!
//! ## Safety
//!
//! Only pure ops are hoisted. The following are NOT hoisted:
//! - `Store`, `Atomic`, `Barrier`, `ThreadgroupStore` (side effects)
//! - `SetLocal` (writes to mutable loop-carried state)
//! - `DeclareLocal` inside loops (mutable variable declaration)
//! - `Load` from mutable/unknown params
//! - Any op whose operands include the loop induction variable
//!
//! ## References
//! - Allen (1970), "A catalogue of optimizing transformations", in *Design and
//!   Optimization of Compilers* (R. Rustin, ed.), Prentice-Hall.  Earliest
//!   systematic description of loop-invariant code motion.
//! - Aho, Lam, Sethi & Ullman (2006), "Compilers: Principles, Techniques, and
//!   Tools", 2nd ed., §9.4.  Standard treatment of loop optimizations including
//!   code motion.

use std::collections::{BTreeMap, BTreeSet};

use metaltile_core::{
    error::Result,
    ir::{Block, BlockId, Kernel, Op, ParamKind, ValueId},
};

use super::remap;

pub struct LicmPass;

impl super::Pass for LicmPass {
    fn name(&self) -> &str { "licm" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        // Determine which params are read-only (Load-safe for hoisting).
        let read_only: BTreeSet<String> = kernel
            .params
            .iter()
            .filter(|p| !p.is_output && matches!(p.kind, ParamKind::Tensor | ParamKind::Strided))
            .map(|p| p.name.clone())
            .collect();

        // Build a definition map: ValueId -> BlockId where it's defined.
        let mut def_block: BTreeMap<ValueId, BlockId> = BTreeMap::new();
        for vid in kernel.body.results.iter().flatten() {
            def_block.insert(*vid, kernel.body.id);
        }
        for (bid, block) in &kernel.blocks {
            for vid in block.results.iter().flatten() {
                def_block.insert(*vid, *bid);
            }
        }

        // Take all blocks out so we can mutate them freely.
        let mut blocks = std::mem::take(&mut kernel.blocks);

        // Process nested blocks first (inside-out / post-order), then kernel.body last.
        // BlockIds are allocated in order, so higher IDs are deeper-nested children.
        // Sorting descending ensures children are processed before their parents,
        // allowing multi-level hoisting (e.g. b3→b2→b1→b0) in a single pass.
        //
        // Exclude kernel.body.id: kernel.blocks may contain an entry with the same
        // BlockId as kernel.body (when the root block is stored there). Processing it
        // here would operate on the wrong Block object and empty child blocks before
        // the explicit kernel.body call below.
        let body_id = kernel.body.id;
        let mut block_ids: Vec<BlockId> =
            blocks.keys().copied().filter(|&bid| bid != body_id).collect();
        block_ids.sort_by_key(|bid| -(bid.as_u32() as i32));

        for bid in block_ids {
            let mut block = blocks.remove(&bid).unwrap();
            licm_block(&mut block, &mut blocks, &def_block, &read_only);
            blocks.insert(bid, block);
        }

        // kernel.body processed last so it can hoist ops that were themselves
        // hoisted into direct child blocks by the inner-block passes above.
        kernel.blocks = blocks;
        licm_block(&mut kernel.body, &mut kernel.blocks, &def_block, &read_only);
        Ok(())
    }
}

/// Process a single block, hoisting invariants from any `Op::Loop` children.
/// `blocks` is the mutable block map so loop bodies can be modified.
fn licm_block(
    block: &mut Block,
    blocks: &mut BTreeMap<BlockId, Block>,
    def_block: &BTreeMap<ValueId, BlockId>,
    read_only: &BTreeSet<String>,
) {
    let n = block.ops.len();

    // Phase 1: for each Op::Loop, find which ops to hoist.
    // (loop_idx, (hoisted_ops, hoisted_results, body_block_id, removal_indices))
    struct HoistPlan {
        loop_idx: usize,
        hoisted_ops: Vec<Op>,
        hoisted_results: Vec<Option<ValueId>>,
        body_id: BlockId,
        removal_indices: Vec<usize>,
    }

    let mut plans: Vec<HoistPlan> = Vec::new();

    for i in 0..n {
        if let Op::Loop { var, body, .. } = &block.ops[i] {
            let Some(loop_body) = blocks.get(body) else {
                continue;
            };

            // Build the initial invariant set: ValueIds defined before position `i`
            // in the parent block, plus any from ancestor blocks.
            let mut invariant: BTreeSet<ValueId> = BTreeSet::new();
            for j in 0..i {
                if let Some(Some(vid)) = block.results.get(j) {
                    invariant.insert(*vid);
                }
            }
            // Also include values from other blocks (ancestors) referenced by the loop.
            // Exclude values from descendants (inner loops) — those were hoisted
            // here and are still loop-variant at this level.
            let body_id_u32 = body.as_u32();
            for op in &loop_body.ops {
                for vid in remap::op_value_refs(op) {
                    if let Some(&def_bid) = def_block.get(&vid) {
                        let def_u32 = def_bid.as_u32();
                        // Ancestor blocks have lower IDs (allocated before body).
                        // Descendant blocks have higher IDs — exclude them.
                        if def_u32 < body_id_u32 {
                            invariant.insert(vid);
                        }
                    }
                }
            }

            // Mark loop iteration variable as variant (NOT invariant).
            // The loop variable is synthesized with ValueId(var.as_u32() + 1000)
            // or ValueId(0xC000_0000 | var.as_u32()) by the codegen.
            // Anything that depends on it must stay in the loop body.
            let loop_vid_a = ValueId::new(var.as_u32() + 1000);
            let loop_vid_b = ValueId::new(0xC000_0000 | var.as_u32());
            invariant.remove(&loop_vid_a);
            invariant.remove(&loop_vid_b);

            // Fixpoint: find hoistable ops.
            let mut hoist_indices: Vec<usize> = Vec::new();
            let m = loop_body.ops.len();
            loop {
                let mut changed = false;
                for j in 0..m {
                    if hoist_indices.contains(&j) {
                        continue;
                    }
                    let op = &loop_body.ops[j];
                    if !is_pure_op(op, read_only) {
                        continue;
                    }
                    let op_refs = remap::op_value_refs(op);
                    if op_refs.iter().all(|v| invariant.contains(v))
                        && let Some(Some(vid)) = loop_body.results.get(j)
                    {
                        invariant.insert(*vid);
                        hoist_indices.push(j);
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }

            if hoist_indices.is_empty() {
                continue;
            }

            // Sort ascending for topological order.
            hoist_indices.sort();

            let hoisted_ops: Vec<Op> =
                hoist_indices.iter().map(|&j| loop_body.ops[j].clone()).collect();
            let hoisted_results: Vec<Option<ValueId>> =
                hoist_indices.iter().map(|&j| loop_body.results[j]).collect();

            plans.push(HoistPlan {
                loop_idx: i,
                hoisted_ops,
                hoisted_results,
                body_id: *body,
                removal_indices: hoist_indices,
            });
        }
    }

    if plans.is_empty() {
        return;
    }

    // Phase 2: remove hoisted ops from loop bodies.
    for plan in &plans {
        if let Some(loop_body) = blocks.get_mut(&plan.body_id) {
            remove_ops_from_block(loop_body, &plan.removal_indices);
        }
    }

    // Phase 3: rebuild the parent block with hoisted ops inserted before each loop.
    let old_ops = std::mem::take(&mut block.ops);
    let old_results = std::mem::take(&mut block.results);
    let n2 = old_ops.len();

    // Map: loop_idx -> (ops, results) to insert before it.
    let mut insert_at: BTreeMap<usize, (&[Op], &[Option<ValueId>])> = BTreeMap::new();
    for plan in &plans {
        insert_at.insert(plan.loop_idx, (&plan.hoisted_ops, &plan.hoisted_results));
    }

    let mut new_ops: Vec<Op> = Vec::new();
    let mut new_results: Vec<Option<ValueId>> = Vec::new();

    for i in 0..n2 {
        // Insert hoisted ops before position i if any.
        if let Some(&(hoisted_ops, hoisted_results)) = insert_at.get(&i) {
            for (op, result) in hoisted_ops.iter().zip(hoisted_results.iter()) {
                new_ops.push(op.clone());
                new_results.push(*result);
            }
        }

        new_ops.push(old_ops[i].clone());
        new_results.push(old_results[i]);
    }

    block.ops = new_ops;
    block.results = new_results;

    // Transfer names for hoisted values from loop bodies to parent block.
    // Without this, hoisted variables become unnamed (v_a_idx0 vs v82 mismatch)
    // and nested blocks can't resolve them into inner_names.
    for plan in &plans {
        if let Some(loop_body) = blocks.get(&plan.body_id) {
            for vid in plan.hoisted_results.iter().flatten() {
                if let Some(name) = loop_body.names.get(vid) {
                    block.names.insert(*vid, name.clone());
                }
            }
        }
    }
}

/// Remove ops at given indices from a block. Indices must be sorted ascending.
fn remove_ops_from_block(block: &mut Block, indices: &[usize]) {
    let skip: BTreeSet<usize> = indices.iter().copied().collect();
    let old_ops = std::mem::take(&mut block.ops);
    let old_results = std::mem::take(&mut block.results);
    let mut new_ops = Vec::new();
    let mut new_results = Vec::new();
    for (i, op) in old_ops.into_iter().enumerate() {
        if !skip.contains(&i) {
            new_ops.push(op);
            new_results.push(old_results[i]);
        }
    }
    block.ops = new_ops;
    block.results = new_results;
}

/// Return true if the op is pure (no side effects) and safe to hoist.
fn is_pure_op(op: &Op, read_only: &BTreeSet<String>) -> bool {
    match op {
        Op::BinOp { .. }
        | Op::UnaryOp { .. }
        | Op::Cast { .. }
        | Op::Activation { .. }
        | Op::Select { .. }
        | Op::Const { .. }
        | Op::ProgramId { .. }
        | Op::Arange { .. }
        | Op::Zeros { .. }
        | Op::Splat { .. }
        | Op::Broadcast { .. }
        | Op::Transpose { .. }
        | Op::ExpandDims { .. }
        | Op::Reshape { .. }
        | Op::Slice { .. }
        | Op::SimdgroupElemLoad { .. }
        | Op::SimdScan { .. } => true,

        // Load from a read-only (const) param is pure.
        Op::Load { src, .. } => read_only.contains(src.as_str()),

        // NOT pure — side effects or loop-dependent:
        Op::Store { .. }
        | Op::Atomic { .. }
        | Op::Barrier
        | Op::SimdgroupAlloc { .. }
        | Op::ThreadgroupStore { .. }
        | Op::SetLocal { .. }
        | Op::DeclareLocal { .. }
        | Op::Loop { .. }
        | Op::If { .. }
        | Op::InlineMsl { .. }
        | Op::VectorStore { .. }
        | Op::Scatter { .. }
        | Op::ThreadgroupLoad { .. }
        | Op::ThreadgroupAlloc { .. }
        | Op::StrideStore { .. }
        | Op::Dequantize { .. }
        | Op::SimdReduce { .. }
        | Op::ArgReduce { .. }
        | Op::FusedElementwise { .. }
        | Op::VectorLoad { .. }
        | Op::StrideReduce { .. }
        | Op::StrideScan { .. }
        | Op::StrideArgReduce { .. }
        | Op::Cat { .. }
        | Op::Gather { .. }
        | Op::Scan { .. }
        | Op::Reduce { .. }
        | Op::Dot { .. }
        | Op::FlashAttention { .. }
        | Op::SlidingWindowAttention { .. }
        | Op::RmsNorm { .. }
        | Op::GatedMlp { .. }
        | Op::SimdgroupMatMul { .. }
        | Op::SimdgroupElemStore { .. }
        | Op::SimdLaneId
        | Op::SimdGroupId => false,
    }
}

#[cfg(test)]
mod tests {
    use metaltile_core::{
        dtype::DType,
        ir::{BinOpKind, IndexExpr, Param, ParamKind, VarId},
        shape::Shape,
    };

    use super::*;
    use crate::passes::Pass;

    #[test]
    fn hoists_loop_invariant_add() {
        let mut k = Kernel::new("licm_add");
        // Parent block: define invariant values.
        k.body.push_op(Op::Const { value: 10 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 20 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(2)); // loop start
        k.body.push_op(Op::Const { value: 4 }, ValueId::new(3)); // loop end
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(4)); // loop step

        // Loop body: the add is invariant because both operands are outside the loop.
        let mut loop_body = Block::new(BlockId::new(1));
        loop_body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(0), rhs: ValueId::new(1) },
            ValueId::new(100),
        );
        let body_id = k.add_block(loop_body);

        k.body.push_op_no_result(Op::Loop {
            var: VarId::new(0),
            start: ValueId::new(2),
            end: ValueId::new(3),
            step: ValueId::new(4),
            body: body_id,
        });

        LicmPass.run(&mut k).unwrap();

        // The add should be hoisted before the Loop in the parent block.
        let loop_pos = k.body.ops.iter().position(|op| matches!(op, Op::Loop { .. })).unwrap();
        let op_before_loop = &k.body.ops[loop_pos - 1];
        assert!(
            matches!(op_before_loop, Op::BinOp { op: BinOpKind::Add, .. }),
            "invariant add should be hoisted before loop"
        );
    }

    #[test]
    fn does_not_hoist_store() {
        let mut k = Kernel::new("licm_store");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 4 }, ValueId::new(2));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(3));

        let mut loop_body = Block::new(BlockId::new(1));
        loop_body.push_op(Op::Const { value: 42 }, ValueId::new(100));
        // Store has side effects — must NOT be hoisted.
        loop_body.push_op_no_result(Op::Store {
            dst: "buf".into(),
            indices: vec![],
            value: ValueId::new(100),
            mask: None,
        });
        let body_id = k.add_block(loop_body);

        k.body.push_op_no_result(Op::Loop {
            var: VarId::new(0),
            start: ValueId::new(0),
            end: ValueId::new(2),
            step: ValueId::new(3),
            body: body_id,
        });
        LicmPass.run(&mut k).unwrap();

        // Store should still be in the loop body.
        let body = k.blocks.get(&body_id).unwrap();
        let has_store = body.ops.iter().any(|op| matches!(op, Op::Store { .. }));
        assert!(has_store, "Store must not be hoisted from loop");
    }

    #[test]
    fn hoists_const_from_loop() {
        let mut k = Kernel::new("licm_const");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 4 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(2));

        let mut loop_body = Block::new(BlockId::new(1));
        // Const in loop body — pure and invariant.
        loop_body.push_op(Op::Const { value: 7 }, ValueId::new(100));
        let body_id = k.add_block(loop_body);

        k.body.push_op_no_result(Op::Loop {
            var: VarId::new(0),
            start: ValueId::new(0),
            end: ValueId::new(1),
            step: ValueId::new(2),
            body: body_id,
        });
        LicmPass.run(&mut k).unwrap();

        // Const should be hoisted.
        let body = k.blocks.get(&body_id).unwrap();
        let has_const = body.ops.iter().any(|op| matches!(op, Op::Const { .. }));
        assert!(!has_const, "loop-invariant Const should be hoisted");
    }

    #[test]
    fn does_not_hoist_load_from_mutable_param() {
        let mut k = Kernel::new("licm_mutable_load");
        k.params.push(Param {
            name: "buf".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: true, // mutable
            kind: ParamKind::Tensor,
        });
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 4 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(2));

        let mut loop_body = Block::new(BlockId::new(1));
        loop_body.push_op(
            Op::Load {
                src: "buf".into(),
                indices: vec![IndexExpr::Value(ValueId::new(0))],
                mask: None,
                other: None,
            },
            ValueId::new(100),
        );
        let body_id = k.add_block(loop_body);

        k.body.push_op_no_result(Op::Loop {
            var: VarId::new(0),
            start: ValueId::new(0),
            end: ValueId::new(1),
            step: ValueId::new(2),
            body: body_id,
        });
        LicmPass.run(&mut k).unwrap();

        // Load from mutable param should NOT be hoisted.
        let body = k.blocks.get(&body_id).unwrap();
        let has_load = body.ops.iter().any(|op| matches!(op, Op::Load { .. }));
        assert!(has_load, "Load from mutable param must not be hoisted");
    }

    #[test]
    fn hoists_read_only_load() {
        let mut k = Kernel::new("licm_readonly_load");
        k.params.push(Param {
            name: "weights".into(),
            dtype: DType::F32,
            shape: Shape::scalar(),
            is_output: false,
            kind: ParamKind::Tensor, // read-only
        });
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 4 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(2));

        let mut loop_body = Block::new(BlockId::new(1));
        loop_body.push_op(
            Op::Load {
                src: "weights".into(),
                indices: vec![IndexExpr::Value(ValueId::new(0))],
                mask: None,
                other: None,
            },
            ValueId::new(100),
        );
        let body_id = k.add_block(loop_body);

        k.body.push_op_no_result(Op::Loop {
            var: VarId::new(0),
            start: ValueId::new(0),
            end: ValueId::new(1),
            step: ValueId::new(2),
            body: body_id,
        });
        LicmPass.run(&mut k).unwrap();

        // Load from read-only param should be hoisted.
        let body = k.blocks.get(&body_id).unwrap();
        let has_load = body.ops.iter().any(|op| matches!(op, Op::Load { .. }));
        assert!(!has_load, "Load from read-only param should be hoisted");
    }
}
