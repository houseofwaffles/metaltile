//! Fusion pass: merge adjacent elementwise operations into a single FusedElementwise op.
//!
//! Algorithm:
//! 1. Build def-use graph: for each ValueId, which op indices use it?
//! 2. Find chains where each op produces a value used only by the next op.
//! 3. Create an `Op::FusedElementwise` containing the whole chain.
//! 4. Replace the original ops; the MSL emitter then emits a single expression.

use std::collections::{BTreeMap, BTreeSet};

use metaltile_core::{
    error::Result,
    ir::{Block, BlockId, IndexExpr, Kernel, Op, ValueId},
};

/// Mask for encoding internal sub-op references within FusedElementwise chains.
pub const SUB_OP_FLAG: u32 = 0x8000_0000;

/// Create a ValueId that references the result of sub-op at position `idx`
/// within a FusedElementwise chain.
pub fn sub_op_ref(idx: usize) -> ValueId { ValueId::new(SUB_OP_FLAG | idx as u32) }

pub struct FusionPass;

impl super::Pass for FusionPass {
    fn name(&self) -> &str { "fusion" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        // Build a map: ValueId → the BlockId that defines (produces) it.
        let mut def_block: BTreeMap<ValueId, BlockId> = BTreeMap::new();
        for vid in kernel.body.results.iter().flatten() {
            def_block.insert(*vid, kernel.body.id);
        }
        for (bid, block) in &kernel.blocks {
            for vid in block.results.iter().flatten() {
                def_block.insert(*vid, *bid);
            }
        }

        // Build a map: ValueId → set of BlockIds that reference (use) it.
        let mut used_in: BTreeMap<ValueId, BTreeSet<BlockId>> = BTreeMap::new();
        for op in &kernel.body.ops {
            for vid in op_value_refs(op) {
                used_in.entry(vid).or_default().insert(kernel.body.id);
            }
        }
        for (bid, block) in &kernel.blocks {
            for op in &block.ops {
                for vid in op_value_refs(op) {
                    used_in.entry(vid).or_default().insert(*bid);
                }
            }
        }

        // Compute per-block pinned sets: a ValueId is pinned in block B if it is
        // defined in B but used in at least one other block (i.e. a child block).
        // Pinned values must not be fused away — they need a standalone declaration
        // so that child blocks can reference the variable by name.
        let mut pinned_per_block: BTreeMap<BlockId, BTreeSet<ValueId>> = BTreeMap::new();
        for (vid, def_bid) in &def_block {
            if let Some(use_bids) = used_in.get(vid) {
                for &use_bid in use_bids {
                    if use_bid != *def_bid {
                        pinned_per_block.entry(*def_bid).or_default().insert(*vid);
                        break;
                    }
                }
            }
        }

        // Fuse the kernel body block.
        let body_pins = pinned_per_block.get(&kernel.body.id).cloned().unwrap_or_default();
        fuse_block(&mut kernel.body, &body_pins);

        // Fuse all nested blocks (loop bodies, if/else branches) with their own
        // per-block pinned sets so values used in grandchild blocks are preserved.
        let block_ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();
        for bid in block_ids {
            let pins = pinned_per_block.get(&bid).cloned().unwrap_or_default();
            if let Some(block) = kernel.blocks.get_mut(&bid) {
                fuse_block(block, &pins);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Fusion on a single block
// ---------------------------------------------------------------------------

fn fuse_block(block: &mut Block, pinned: &BTreeSet<ValueId>) {
    // Phase 1: build def-use graph.
    // uses[vid] = set of op indices that reference vid.
    let mut uses: BTreeMap<ValueId, Vec<usize>> = BTreeMap::new();
    for (i, op) in block.ops.iter().enumerate() {
        for vid in op_value_refs(op) {
            uses.entry(vid).or_default().push(i);
        }
    }

    // Phase 2: find maximal fusible chains by scanning backward.
    // A chain ends at an op C where C does NOT have exactly one user.
    // Walk backward from the last op, collecting fusible producers.
    let n = block.ops.len();
    let mut fused: BTreeSet<usize> = BTreeSet::new(); // op indices already in a fused chain.
    let mut chains: Vec<Vec<usize>> = Vec::new();

    for i in (0..n).rev() {
        if fused.contains(&i) {
            continue;
        }
        if !is_fusible(&block.ops[i]) {
            continue;
        }

        // Walk backward from op i, collecting a linear chain.
        let mut chain: Vec<usize> = vec![i];
        let mut cursor = i;

        while let Some(prev_result) = first_value_input(&block.ops[cursor]) {
            // Find which op in this block produced prev_result.
            let Some(prev_idx) = block.results.iter().position(|r| *r == Some(prev_result)) else {
                break;
            };
            // The producer must:
            // - Be fusible
            // - Produce a value used ONLY by cursor (single-use)
            // - Come before cursor in the block
            if prev_idx >= cursor || !is_fusible(&block.ops[prev_idx]) || fused.contains(&prev_idx)
            {
                break;
            }
            let use_count = uses.get(&prev_result).map(|v| v.len()).unwrap_or(0);
            if use_count != 1 || pinned.contains(&prev_result) {
                break;
            }
            // Good — add to chain.
            chain.push(prev_idx);
            cursor = prev_idx;
        }

        if chain.len() >= 2 {
            // If the terminal result is pinned (used in child blocks), don't fuse.
            // Child blocks reference the variable by name; fusing would make it a
            // FusedElementwise typed by type_env, which may disagree with the Metal
            // compiler's type deduction (e.g., uint arithmetic typed as float).
            let terminal_vid = block.results.get(chain[0]).and_then(|r| *r);
            if terminal_vid.is_some_and(|v| pinned.contains(&v)) {
                continue;
            }
            // Reverse so ops are in execution order (producer first).
            chain.reverse();
            for &idx in &chain {
                fused.insert(idx);
            }
            chains.push(chain);
        }
    }

    if chains.is_empty() {
        return;
    }

    // Phase 3: rewrite the block — replace chains with FusedElementwise.
    // Build a new ops/results vec.
    let mut new_ops: Vec<Op> = Vec::new();
    let mut new_results: Vec<Option<ValueId>> = Vec::new();
    let old_results = std::mem::take(&mut block.results);
    let old_ops = std::mem::take(&mut block.ops);

    // Build a mapping: old op index → new ValueId for its result (if it survives).
    // For ops in fused chains, their results are replaced by the chain's output vid.
    let mut chain_result_map: BTreeMap<usize, ValueId> = BTreeMap::new();

    // Pre-compute: for each chain, what is its output ValueId?
    // The output ValueId is the result of the LAST op in the chain.
    for chain in &chains {
        let last_idx = chain[chain.len() - 1];
        if let Some(Some(out_vid)) = old_results.get(last_idx).copied() {
            for &idx in chain {
                chain_result_map.insert(idx, out_vid);
            }
        }
    }

    // Also track which old indices should be skipped (they're in a fused chain,
    // but not the last one — only the last one gets emitted).
    let mut skip_indices: BTreeSet<usize> = BTreeSet::new();
    for chain in &chains {
        for &idx in chain.iter().take(chain.len() - 1) {
            skip_indices.insert(idx);
        }
    }

    // Rewrite ValueId references in surviving ops to use chain outputs.
    // When an op (not in a chain) references a ValueId produced by a fused chain,
    // it should reference the chain's output ValueId.
    fn remap_value(
        v: &mut ValueId,
        chain_map: &BTreeMap<usize, ValueId>,
        old_results: &[Option<ValueId>],
    ) {
        for (&old_idx, &new_vid) in chain_map {
            if old_idx < old_results.len() && old_results[old_idx] == Some(*v) {
                *v = new_vid;
                return;
            }
        }
    }

    fn remap_op(
        op: &mut Op,
        chain_map: &BTreeMap<usize, ValueId>,
        old_results: &[Option<ValueId>],
    ) {
        match op {
            Op::BinOp { lhs, rhs, .. } => {
                remap_value(lhs, chain_map, old_results);
                remap_value(rhs, chain_map, old_results);
            },
            Op::UnaryOp { value, .. }
            | Op::Activation { value, .. }
            | Op::Cast { value, .. }
            | Op::Reduce { value, .. }
            | Op::Transpose { value }
            | Op::Slice { value, .. }
            | Op::Broadcast { value, .. } => {
                remap_value(value, chain_map, old_results);
            },
            Op::Select { cond, on_true, on_false } => {
                remap_value(cond, chain_map, old_results);
                remap_value(on_true, chain_map, old_results);
                remap_value(on_false, chain_map, old_results);
            },
            Op::Dot { a, b } => {
                remap_value(a, chain_map, old_results);
                remap_value(b, chain_map, old_results);
            },
            Op::Store { value, .. } => {
                remap_value(value, chain_map, old_results);
            },
            Op::Loop { start, end, step, .. } => {
                remap_value(start, chain_map, old_results);
                remap_value(end, chain_map, old_results);
                remap_value(step, chain_map, old_results);
            },
            Op::DeclareLocal { value, .. } | Op::SetLocal { value, .. } => {
                remap_value(value, chain_map, old_results);
            },
            Op::ThreadgroupStore { index, value, .. } => {
                remap_value(index, chain_map, old_results);
                remap_value(value, chain_map, old_results);
            },
            Op::SimdReduce { value, .. } | Op::ArgReduce { value, .. } => {
                remap_value(value, chain_map, old_results);
            },
            _ => {},
        }
    }

    let old_ops_snapshot = old_ops.clone();
    for (i, mut op) in old_ops.into_iter().enumerate() {
        if skip_indices.contains(&i) {
            continue; // op is fused into a chain — skip it.
        }

        // Remap value references.
        remap_op(&mut op, &chain_result_map, &old_results);

        // If this op is the last in a fused chain, emit the FusedElementwise instead.
        if let Some(chain) = chains.iter().find(|c| c[c.len() - 1] == i) {
            let fused_ops: Vec<Op> = chain
                .iter()
                .map(|&idx| build_fused_sub_op(idx, chain, &old_ops_snapshot, &old_results))
                .collect();
            new_ops.push(Op::FusedElementwise { ops: fused_ops });
            new_results.push(old_results.get(i).copied().unwrap_or(None));
        } else {
            new_ops.push(op);
            new_results.push(old_results.get(i).copied().unwrap_or(None));
        }
    }

    block.ops = new_ops;
    block.results = new_results;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns true if the op is an elementwise op that can participate in fusion.
fn is_fusible(op: &Op) -> bool {
    matches!(
        op,
        Op::BinOp { .. }
            | Op::UnaryOp { .. }
            | Op::Activation { .. }
            | Op::Cast { .. }
            | Op::Select { .. }
            | Op::Zeros { .. }
            | Op::Splat { .. }
            | Op::Broadcast { .. }
    )
}

/// Return all ValueId references in an op.
fn op_value_refs(op: &Op) -> Vec<ValueId> {
    let mut refs = Vec::new();
    match op {
        Op::BinOp { lhs, rhs, .. } => {
            refs.push(*lhs);
            refs.push(*rhs);
        },
        Op::UnaryOp { value, .. }
        | Op::Activation { value, .. }
        | Op::Cast { value, .. }
        | Op::Reduce { value, .. }
        | Op::Transpose { value }
        | Op::Slice { value, .. }
        | Op::Broadcast { value, .. } => {
            refs.push(*value);
        },
        Op::Select { cond, on_true, on_false } => {
            refs.push(*cond);
            refs.push(*on_true);
            refs.push(*on_false);
        },
        Op::Dot { a, b } => {
            refs.push(*a);
            refs.push(*b);
        },
        Op::Load { indices, mask, .. } => {
            for ix in indices {
                if let IndexExpr::Value(v) | IndexExpr::Range(v, _) = ix {
                    refs.push(*v);
                }
            }
            if let Some(m) = mask {
                refs.push(*m);
            }
        },
        Op::Store { indices, value, mask, .. } => {
            for ix in indices {
                if let IndexExpr::Value(v) | IndexExpr::Range(v, _) = ix {
                    refs.push(*v);
                }
            }
            refs.push(*value);
            if let Some(m) = mask {
                refs.push(*m);
            }
        },
        Op::Loop { start, end, step, .. } => {
            refs.push(*start);
            refs.push(*end);
            refs.push(*step);
        },
        Op::InlineMsl { inputs, .. } => {
            refs.extend(inputs);
        },
        Op::FlashAttention { q, k, v, .. } => {
            refs.push(*q);
            refs.push(*k);
            refs.push(*v);
        },
        Op::SlidingWindowAttention { q, k, v, .. } => {
            refs.push(*q);
            refs.push(*k);
            refs.push(*v);
        },
        Op::RmsNorm { x, scale, .. } => {
            refs.push(*x);
            refs.push(*scale);
        },
        Op::GatedMlp { x, gate_proj, up_proj, down_proj } => {
            refs.push(*x);
            refs.push(*gate_proj);
            refs.push(*up_proj);
            refs.push(*down_proj);
        },
        Op::FusedElementwise { ops } =>
            for sub in ops {
                refs.extend(op_value_refs(sub));
            },
        Op::VectorLoad { byte_offset, .. } => {
            refs.push(*byte_offset);
        },
        Op::VectorStore { byte_offset, value, .. } => {
            refs.push(*byte_offset);
            refs.push(*value);
        },
        Op::StrideReduce { offset, stride, end, .. } => {
            refs.push(*offset);
            refs.push(*stride);
            refs.push(*end);
        },
        Op::If { cond, .. } => {
            refs.push(*cond);
        },
        Op::ExpandDims { value, .. } => {
            refs.push(*value);
        },
        Op::Reshape { value, .. } => {
            refs.push(*value);
        },
        Op::Cat { values, .. } => {
            refs.extend(values);
        },
        Op::Gather { indices, .. } => {
            refs.push(*indices);
        },
        Op::Scatter { indices, value, .. } => {
            refs.push(*indices);
            refs.push(*value);
        },
        Op::Atomic { index, value, .. } => {
            refs.push(*index);
            refs.push(*value);
        },
        Op::Scan { value, .. } => {
            refs.push(*value);
        },
        Op::StrideScan { offset, end, .. } => {
            refs.push(*offset);
            refs.push(*end);
        },
        Op::StrideArgReduce { offset, end, .. } => {
            refs.push(*offset);
            refs.push(*end);
        },
        Op::DeclareLocal { value, .. } | Op::SetLocal { value, .. } => {
            refs.push(*value);
        },
        Op::StrideStore { offset, end, scalar, .. } => {
            refs.push(*offset);
            refs.push(*end);
            refs.push(*scalar);
        },
        Op::ThreadgroupLoad { index, .. } => {
            refs.push(*index);
        },
        Op::ThreadgroupStore { index, value, .. } => {
            refs.push(*index);
            refs.push(*value);
        },
        Op::SimdReduce { value, .. } | Op::ArgReduce { value, .. } => {
            refs.push(*value);
        },
        _ => {},
    }
    refs
}

/// Return the first ValueId input of an op (used to trace the chain backward).
fn first_value_input(op: &Op) -> Option<ValueId> {
    match op {
        Op::BinOp { lhs, .. }
        | Op::UnaryOp { value: lhs, .. }
        | Op::Activation { value: lhs, .. }
        | Op::Cast { value: lhs, .. }
        | Op::Select { cond: lhs, .. }
        | Op::Broadcast { value: lhs, .. } => Some(*lhs),
        _ => None,
    }
}

/// Build a sub-op for a FusedElementwise chain.
/// Rewrites ValueId references: external ValueIds stay as-is, internal
/// (to the chain) references are encoded with `sub_op_ref(relative_idx)`.
fn build_fused_sub_op(
    idx: usize,
    chain: &[usize],
    old_ops: &[Op],
    old_results: &[Option<ValueId>],
) -> Op {
    let op = old_ops[idx].clone();
    let pos_in_chain = chain.iter().position(|&c| c == idx).unwrap();

    let map = |v: &mut ValueId| {
        // If this ValueId is produced by a previous op in the same chain,
        // encode it as an internal reference.
        if let Some(producer_pos) =
            chain.iter().position(|&c| c < old_results.len() && old_results[c] == Some(*v))
            && producer_pos < pos_in_chain
        {
            *v = sub_op_ref(producer_pos);
        }
    };

    let mut new_op = op;
    match &mut new_op {
        Op::BinOp { lhs, rhs, .. } => {
            map(lhs);
            map(rhs);
        },
        Op::UnaryOp { value, .. }
        | Op::Activation { value, .. }
        | Op::Cast { value, .. }
        | Op::Reduce { value, .. }
        | Op::Transpose { value }
        | Op::Slice { value, .. }
        | Op::Broadcast { value, .. } => {
            map(value);
        },
        Op::Select { cond, on_true, on_false } => {
            map(cond);
            map(on_true);
            map(on_false);
        },
        _ => {},
    }

    new_op
}
