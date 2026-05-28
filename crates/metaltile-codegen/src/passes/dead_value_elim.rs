//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Dead Value Elimination — remove pure value-producing ops whose result is unused.
//!
//! Closes the loop on a long-standing codegen wart: passes that fuse or
//! eliminate the *consumer* of an SSA value (Vectorize collapsing 4 scalar
//! `Load`s into 1 `VectorLoad`, Unroll inlining a loop body and dropping the
//! start/end/step `Const`s, CopyProp / CSE / ValueSink rewiring operand
//! ValueIds) leave the *producer* in the block.  Those producers are pure
//! arithmetic (`BinOp`, `Const`, `Cast`, etc.) with no remaining uses, but
//! since no pass in the pipeline removed dead value-producing ops, they
//! survived all the way to MSL emission — every one became an
//! `unused-variable` warning when `swift build` compiled the generated
//! `.metal` files.  Emitting `metaltile-std`'s full kernel set produced
//! >5000 such warnings.
//!
//! ## Algorithm
//!
//! Iterate to fixpoint (removing one op can make its operand-producers dead
//! in turn):
//!
//! 1. Walk every op in `kernel.body` plus every op in every nested block
//!    (`kernel.blocks.values()`); collect the set of `ValueId`s referenced
//!    by *any* op anywhere in the kernel.  We must walk all blocks — a
//!    value produced in the outer block can be referenced by an op inside
//!    a nested `If`/`Loop` body, and vice versa.
//! 2. For each block, mark op `i` as dead iff:
//!    - it produces a result (`block.results[i].is_some()`), AND
//!    - it has no side effects (`!op.has_side_effects()`), AND
//!    - it is not an *indexed* load (`!(is_load() && !indices.is_empty())`
//!      — see "Safety" below), AND
//!    - its result is *not* in the use set.
//! 3. Remove dead ops with [`block_util::remove_ops`].
//! 4. Repeat until a pass over all blocks removes nothing.
//!
//! ## Safety
//!
//! - **Side-effecting ops** (`Store`, `Barrier`, `StackStore`,
//!   `ThreadgroupStore`, etc.) are never removed — `has_side_effects()`
//!   filters them out.
//! - **Indexed Loads are conservatively preserved.**  An `Op::Load`
//!   with non-empty `indices` reads from device/threadgroup memory and
//!   synchronises with prior `Op::Store`s under Metal's memory model.
//!   Even if `dead_store_elim` ran before us, eliding such a load near
//!   a barrier could subtly change observable behaviour, so we leave
//!   them alone.
//! - **Scalar Loads are eliminable.**  `Op::Load` with empty `indices`
//!   is a uniform read — a kernel-builtin identifier (`tid`, `n_simd`,
//!   …), a named constant (`0.0f`, `INFINITY`, …), or a constexpr
//!   parameter loaded as a scalar.  None of these participate in
//!   memory-order dependencies, so when const-folded `select`s or
//!   dead-branch elimination orphan the load (e.g. `q_position` in the
//!   non-causal variants of `aura_flash_p1`), we can drop it.
//! - **No-result ops** (`Store`, `Loop`, `If`, `Barrier`, …) can't be
//!   "unused" — they're skipped by the `results[i].is_some()` guard.
//! - **`FusedElementwise` sub-op refs** use the `0x8000_0000` top-bit
//!   convention to encode chain-internal indices, not real kernel-wide
//!   ValueIds (see `passes::fusion`).  We pull them in via
//!   `op.value_refs()` anyway — they don't match any real `block.results`
//!   entry, so they're harmless extras in the use set.
//!
//! ## Pipeline ordering
//!
//! Runs LAST in the standard pipeline, after `dead_store_elim`.  Order
//! matters: most upstream passes (Vectorize, Unroll, CopyProp, CSE, LICM,
//! Schedule, ValueSink) can create dead values; running DCE only at the
//! end means we sweep them all up in one fixpoint.
//!
//! ## References
//! - Aho, Lam, Sethi & Ullman (2006), "Compilers: Principles, Techniques,
//!   and Tools", 2nd ed., §9.1.  Mark-and-sweep dead-code elimination,
//!   including the iterate-to-fixpoint property exploited here.
//! - Kennedy (1979), "Use-definition chains with applications", Computer
//!   Languages 3(3):163–179.  Worklist-driven DCE on SSA form.

use metaltile_core::ir::{BlockId, Kernel};
use rustc_hash::FxHashSet;

use super::block_util;
use crate::error::{Error, Result};

/// Cap on fixpoint iterations.  An upper bound is enough — each iteration
/// strictly shrinks the kernel, and real kernels converge in 2-3 passes
/// (initial sweep + 1-2 chain-of-dependencies sweeps).  16 is plenty of
/// headroom without risking a runaway loop on a pathological IR.
const MAX_ITERATIONS: usize = 16;

pub struct DeadValueElimPass;

impl super::Pass for DeadValueElimPass {
    fn name(&self) -> &str { "dead_value_elim" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> { force_eliminate_dead_values(kernel) }
}

/// Per-pass DCE postcondition (#209/1).  Each orphan-producing pass
/// (Vectorize, Unroll, CopyProp, CSE, LICM, IfConversion, ValueSink,
/// Fusion, FmaFusion, AlgebraicSimplify, ConstFold) calls this at
/// the end of its `run()` to enforce the "if you remove the only
/// use of a value, you remove the value" invariant *as the pass's
/// own postcondition* rather than relying on a global DCE sweep at
/// the end of the pipeline.
///
/// **Test-fixture guard.**  Kernels with no side-effecting op (no
/// Store, no Barrier, no ThreadgroupStore, …) have no liveness
/// anchor; every value is dead by definition.  Skipping DCE in that
/// case keeps degenerate pass-level test fixtures working without
/// forcing every unit test to scaffold an output Store.  Production
/// kernels always have at least one Store to an output — they hit
/// the full DCE loop.  Tests that explicitly want to exercise DCE
/// on an anchorless kernel call [`DeadValueElimPass`] / [`force_eliminate_dead_values`]
/// instead.
pub fn eliminate_dead_values(kernel: &mut Kernel) -> Result<()> {
    let has_anchor = |b: &metaltile_core::ir::Block| b.ops.iter().any(|op| op.has_side_effects());
    if !kernel.iter_blocks().any(has_anchor) {
        return Ok(());
    }
    force_eliminate_dead_values(kernel)
}

/// Unconditional DCE — runs the full fixpoint sweep regardless of
/// whether the kernel has a liveness anchor.  Used by
/// `DeadValueElimPass::run` (the registry entry point) and by tests
/// that explicitly want to drive DCE without scaffolding a Store.
///
/// Most production callers should use [`eliminate_dead_values`] —
/// the guarded variant correctly skips no-op work on anchorless
/// fixtures.
pub fn force_eliminate_dead_values(kernel: &mut Kernel) -> Result<()> {
    for _ in 0..MAX_ITERATIONS {
        let used = collect_used_value_ids(kernel);
        let mut removed_any = false;

        // Entry block — `kernel.body` is the canonical entry block;
        // `kernel.blocks` only holds nested blocks (post-#209/2).
        removed_any |= dve_block(&mut kernel.body, &used);

        // Nested blocks.  Snapshot ids so we can iterate-and-mutate.
        let block_ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();
        for bid in block_ids {
            let mut block =
                kernel.blocks.remove(&bid).ok_or_else(|| Error::BlockNotFound(bid.as_u32()))?;
            removed_any |= dve_block(&mut block, &used);
            kernel.blocks.insert(bid, block);
        }

        if !removed_any {
            break;
        }
    }
    Ok(())
}

/// Walk every op in every block of the kernel; return the union of
/// `ValueId`s any op references as an operand.
///
/// Includes refs from the body and from every nested block.  A value
/// defined in an outer block can be used inside an `If`/`Loop` body, and
/// codegen passes (Unroll especially) freely move ops between blocks —
/// we have to consider the whole kernel as one liveness graph.
fn collect_used_value_ids(kernel: &Kernel) -> FxHashSet<u32> {
    let total_ops =
        kernel.body.ops.len() + kernel.blocks.values().map(|b| b.ops.len()).sum::<usize>();
    let mut used: FxHashSet<u32> =
        FxHashSet::with_capacity_and_hasher(total_ops * 4, Default::default());
    for block in kernel.iter_blocks() {
        for op in &block.ops {
            for v in op.value_refs() {
                used.insert(v.as_u32());
            }
        }
    }
    used
}

/// Eliminate dead value-producing ops from a single block.  Returns
/// `true` if any op was removed.
fn dve_block(block: &mut metaltile_core::ir::Block, used: &FxHashSet<u32>) -> bool {
    let n = block.ops.len();
    if n == 0 {
        return false;
    }

    let mut dead: Vec<usize> = Vec::new();
    for i in 0..n {
        // Only consider ops that PRODUCE a value.  No-result ops (Store,
        // Loop, If, Barrier, …) can't be "unused".
        let Some(Some(result_vid)) = block.results.get(i) else { continue };

        let op = &block.ops[i];

        // Conservative safety filters — see module doc "Safety" section.
        if op.has_side_effects() {
            continue;
        }
        // INDEXED loads (`Op::Load { indices: [_, …], … }`) are
        // conservatively preserved — they read from device/threadgroup
        // memory and synchronise with prior Stores under Metal's memory
        // model.  SCALAR loads (`indices.is_empty()`) are uniform reads:
        // either a kernel-builtin identifier (`tid`, `n_simd`, …), a
        // named constant (`0.0f`, `INFINITY`, …), or a constexpr param
        // loaded as a scalar.  None of these participate in
        // memory-order dependencies, so they're safe to eliminate when
        // their result has no consumer (e.g. a const-folded `select`
        // branch dropped the only user of a constexpr param).
        if op.is_load() && !op.load_indices().is_empty() {
            continue;
        }

        // Used somewhere?  Then keep.
        if used.contains(&result_vid.as_u32()) {
            continue;
        }

        dead.push(i);
    }

    if dead.is_empty() {
        return false;
    }

    // remove_ops requires sorted ascending indices.
    block_util::remove_ops(block, &dead);
    true
}

#[cfg(test)]
mod tests {
    use metaltile_core::ir::{Block, BlockId, IndexExpr, Op, ValueId, VarId};

    use super::*;
    use crate::passes::Pass;

    /// Regression: Vectorize collapses 4 consecutive scalar Loads into 1
    /// VectorLoad + 4 VectorExtracts.  The original *index* BinOps
    /// (`base + 1`, `base + 2`, `base + 3`) that fed the scalar Loads are
    /// left in the block — they have no remaining uses.  Pre-fix, each
    /// became `auto vNN = base + offset;` in MSL → `unused-variable`
    /// warning.  This test pins the fix: a `BinOp` whose result is
    /// unreferenced must be eliminated.
    #[test]
    fn removes_orphan_binop_left_by_vectorize() {
        let mut k = Kernel::new("dve_orphan_binop");
        // ProgramId → v0 (used by Store as the destination index)
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        // Const 1 → v1
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(1));
        // Const 2 → v2 (orphan operand to the dead BinOp below)
        k.body.push_op(Op::Const { value: 2 }, ValueId::new(2));
        // ORPHAN BinOp: v0 + v2 → v3, but nothing references v3
        // (mimics the address-add that vectorize leaves behind after
        // collapsing scalar Loads into a VectorLoad)
        k.body.push_op(
            Op::BinOp {
                op: metaltile_core::ir::BinOpKind::Add,
                lhs: ValueId::new(0),
                rhs: ValueId::new(2),
            },
            ValueId::new(3),
        );
        // Real Store using v0 + v1 (the surviving uses)
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(1),
            mask: None,
        });

        let before = k.body.ops.len();
        DeadValueElimPass.run(&mut k).unwrap();
        let after = k.body.ops.len();

        // The orphan BinOp at idx 3 should be gone.  v2 is now also
        // unreferenced (it only fed the dead BinOp), so the fixpoint
        // should sweep it too — that's the multi-iteration behaviour.
        assert_eq!(after, before - 2, "should remove orphan BinOp and its sole-use Const");

        // Concretely: no op should have result == v3 (the dead value)
        // and none should have result == v2 (cleaned by the fixpoint).
        for r in k.body.results.iter().flatten() {
            assert_ne!(r.as_u32(), 3, "dead BinOp result v3 must be removed");
            assert_ne!(r.as_u32(), 2, "now-dead Const v2 must be removed (fixpoint)");
        }
    }

    /// Regression: Unroll inlines `for _r in range(0u32, 4u32, 1u32) { … }`
    /// by replacing the `Op::Loop` with the unrolled body.  The
    /// `start`/`end`/`step` `Const`s in the parent block (0, 4, 1) become
    /// orphans — `Op::Loop` was their only consumer.  Pre-fix, every
    /// unrolled range loop in the kernel suite left 3 dead `uint vNN = Ku;`
    /// lines in MSL.
    #[test]
    fn removes_orphan_loop_bounds_after_unroll() {
        let mut k = Kernel::new("dve_unroll_bounds");
        // Simulate the post-unroll IR: start/end/step Consts left over,
        // no Op::Loop referencing them.
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0)); // start (orphan)
        k.body.push_op(Op::Const { value: 4 }, ValueId::new(1)); // end (orphan)
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(2)); // step (orphan)

        // A surviving live store so the kernel isn't trivially empty.
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(3));
        k.body.push_op(Op::Const { value: 42 }, ValueId::new(4));
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![IndexExpr::Value(ValueId::new(3))],
            value: ValueId::new(4),
            mask: None,
        });

        DeadValueElimPass.run(&mut k).unwrap();

        // The three orphan Consts must be gone; the live ProgramId / 42 /
        // Store must remain.
        let const_results: Vec<u32> = k
            .body
            .ops
            .iter()
            .zip(k.body.results.iter())
            .filter_map(|(op, r)| if matches!(op, Op::Const { .. }) { *r } else { None })
            .map(|v| v.as_u32())
            .collect();
        assert_eq!(const_results, vec![4], "only the live Const (42 → v4) should survive");

        // Store still there.
        assert!(k.body.ops.iter().any(|op| matches!(op, Op::Store { .. })));
    }

    /// Stores are side-effecting — never removed, even if no one reads
    /// their target.  (DSE handles overwritten-without-read; that's a
    /// separate concern.)
    #[test]
    fn preserves_stores_with_no_reader() {
        let mut k = Kernel::new("dve_keep_store");
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 7 }, ValueId::new(1));
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(1),
            mask: None,
        });

        let before = k.body.ops.len();
        DeadValueElimPass.run(&mut k).unwrap();
        assert_eq!(k.body.ops.len(), before, "side-effecting Store must not be eliminated");
    }

    /// INDEXED loads (`Op::Load { indices: [_, …], … }`) are
    /// conservatively preserved — they read from device/threadgroup
    /// memory and synchronise with prior Stores under Metal's memory
    /// model, so eliding an unread tensor load could change observable
    /// behaviour around a barrier.  Even an unread load with no mask
    /// stays.
    #[test]
    fn preserves_unused_indexed_loads_conservatively() {
        let mut k = Kernel::new("dve_keep_indexed_load");
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        // Load whose result is unused — kept regardless.
        k.body.push_op(
            Op::Load {
                src: "x".into(),
                indices: vec![IndexExpr::Value(ValueId::new(0))],
                mask: None,
                other: None,
            },
            ValueId::new(1),
        );

        DeadValueElimPass.run(&mut k).unwrap();
        assert!(
            k.body.ops.iter().any(|op| matches!(op, Op::Load { .. })),
            "unused indexed Load must be preserved (conservative)"
        );
    }

    /// SCALAR loads (`Op::Load { indices: [], … }`) are NOT preserved
    /// — they're uniform reads of a builtin identifier (`tid`,
    /// `n_simd`, …), a named constant (`0.0f`, `INFINITY`, …), or a
    /// constexpr param loaded as a scalar.  None of these participate
    /// in memory-order dependencies, so DCE can drop them when the
    /// result is unused.  This is the path that catches `q_position`
    /// in the non-causal variants of `aura_flash_p1_*`, where const
    /// folding of `select($causal == 1, q_position + 1, …)` makes the
    /// load orphan-but-still-present (Load conservatively preserved
    /// pre-fix → trailing `-Wunused-variable v24 = q_position;` in
    /// emitted MSL).
    #[test]
    fn eliminates_unused_scalar_load() {
        let mut k = Kernel::new("dve_drop_scalar_load");
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        // Scalar identifier load — empty indices — whose result is
        // never referenced.  The status quo emits `auto v1 = q_position;`
        // and the Metal compiler flags it `-Wunused-variable`.
        k.body.push_op(
            Op::Load { src: "q_position".into(), indices: Vec::new(), mask: None, other: None },
            ValueId::new(1),
        );
        // Surviving live op so the kernel isn't trivially empty.
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(2));
        k.body.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(2),
            mask: None,
        });

        DeadValueElimPass.run(&mut k).unwrap();
        // The scalar Load on q_position must be gone.
        assert!(
            !k.body.ops.iter().any(|op| matches!(op, Op::Load { .. })),
            "unused scalar Load (empty indices) must be eliminated"
        );
        // The Store still there.
        assert!(k.body.ops.iter().any(|op| matches!(op, Op::Store { .. })));
    }

    /// Uses inside a nested `Op::If` then_block must keep the outer-block
    /// producer alive — DCE's use-set has to walk all blocks, not just
    /// the body.
    #[test]
    fn cross_block_use_keeps_outer_producer_alive() {
        let mut k = Kernel::new("dve_cross_block");
        // Outer body: produce v1, then If(v0) { store v1 }
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        // v1 is produced in the outer block …
        k.body.push_op(Op::Const { value: 99 }, ValueId::new(1));
        // … and referenced ONLY inside the If's then-block (below).
        let mut then_block = Block::new(BlockId::new(1));
        then_block.push_op_no_result(Op::Store {
            dst: "out".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(1),
            mask: None,
        });
        let then_id = k.add_block(then_block);
        k.body.push_op_no_result(Op::If {
            cond: ValueId::new(0),
            then_block: then_id,
            else_block: None,
        });

        let const_before = k.body.ops.iter().filter(|op| matches!(op, Op::Const { .. })).count();
        assert_eq!(const_before, 1);

        DeadValueElimPass.run(&mut k).unwrap();

        // v1's only use is across the block boundary — DCE must NOT
        // eliminate it.
        let const_after = k.body.ops.iter().filter(|op| matches!(op, Op::Const { .. })).count();
        assert_eq!(const_after, 1, "Const used by nested-block Store must survive");
    }

    /// Iterate to fixpoint: a chain of dead arithmetic (`v_a + v_b` →
    /// then `v_c * v_d` → …) collapses in successive sweeps as each
    /// removal makes its operands eligible.
    #[test]
    fn iterates_to_fixpoint_on_dead_chain() {
        let mut k = Kernel::new("dve_chain");
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 2 }, ValueId::new(1));
        // chain: v2 = v0 + v1 ; v3 = v2 * v0 ; v4 = v3 - v1  (no use of v4)
        k.body.push_op(
            Op::BinOp {
                op: metaltile_core::ir::BinOpKind::Add,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            },
            ValueId::new(2),
        );
        k.body.push_op(
            Op::BinOp {
                op: metaltile_core::ir::BinOpKind::Mul,
                lhs: ValueId::new(2),
                rhs: ValueId::new(0),
            },
            ValueId::new(3),
        );
        k.body.push_op(
            Op::BinOp {
                op: metaltile_core::ir::BinOpKind::Sub,
                lhs: ValueId::new(3),
                rhs: ValueId::new(1),
            },
            ValueId::new(4),
        );

        DeadValueElimPass.run(&mut k).unwrap();

        // Whole chain — including the two leading Consts — must collapse.
        assert_eq!(k.body.ops.len(), 0, "fixpoint should remove the entire dead chain");
    }

    /// REGRESSION: `Kernel` stores the entry block twice — once as
    /// Post-#209/2 the entry block lives only at `kernel.body`;
    /// `kernel.blocks` holds nested blocks only.  This test pins that
    /// invariant: build a kernel, observe that `kernel.body.id` is
    /// NOT a key in `kernel.blocks`, and confirm DCE only needs the
    /// one canonical entry block to do its job.
    ///
    /// Replaces the pre-fix `ignores_stale_entry_block_copy_in_kernel_blocks`
    /// test, which was specifically reproducing the stale-copy footgun
    /// that no longer exists.
    #[test]
    fn entry_block_is_not_duplicated_in_kernel_blocks() {
        let mut k = Kernel::new("dve_entry_block");
        // Bounds + a fake Op::Loop that originally consumed them, then
        // strip the loop from `kernel.body` — exactly the state Unroll
        // leaves behind after inlining a loop body.
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 4 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(2));
        let mut loop_body = Block::new(BlockId::new(1));
        loop_body.push_op(Op::Const { value: 99 }, ValueId::new(10));
        let body_id = k.add_block(loop_body);
        k.body.push_op_no_result(Op::Loop {
            var: VarId::new(0),
            start: ValueId::new(0),
            end: ValueId::new(1),
            step: ValueId::new(2),
            body: body_id,
        });
        k.body.ops.pop();
        k.body.results.pop();

        // Invariant: the entry block is NOT in `kernel.blocks`.
        assert!(
            !k.blocks.contains_key(&k.body.id),
            "Kernel::new must not duplicate the entry block into kernel.blocks"
        );

        DeadValueElimPass.run(&mut k).unwrap();

        let surviving_consts: Vec<i64> = k
            .body
            .ops
            .iter()
            .filter_map(|op| if let Op::Const { value } = op { Some(*value) } else { None })
            .collect();
        assert!(
            surviving_consts.is_empty(),
            "orphan loop-bound Consts must be eliminated: {surviving_consts:?}"
        );
    }

    /// Loop-IV-bound `Op::Loop` references on `start`/`end`/`step`: don't
    /// eliminate those Consts.  This mirrors the in-progress IR state
    /// *before* Unroll runs (or for loops Unroll can't unroll), where
    /// the bounds are real uses.
    #[test]
    fn preserves_consts_referenced_by_unrolled_loop_bounds() {
        // Hold on — this test is really about the OPPOSITE: the loop is
        // NOT unrolled (e.g. non-constant trip count), so the Op::Loop
        // op is still present and the bounds Consts ARE referenced.
        let mut k = Kernel::new("dve_keep_loop_bounds");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 100 }, ValueId::new(1)); // dynamic trip count
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(2));

        let mut loop_body = Block::new(BlockId::new(1));
        loop_body.push_op(Op::Const { value: 0 }, ValueId::new(10));
        let body_id = k.add_block(loop_body);

        k.body.push_op_no_result(Op::Loop {
            var: VarId::new(0),
            start: ValueId::new(0),
            end: ValueId::new(1),
            step: ValueId::new(2),
            body: body_id,
        });

        DeadValueElimPass.run(&mut k).unwrap();

        // All three bound Consts referenced by Op::Loop must survive.
        let const_count = k.body.ops.iter().filter(|op| matches!(op, Op::Const { .. })).count();
        assert_eq!(const_count, 3, "loop start/end/step Consts must be preserved");
    }
}
