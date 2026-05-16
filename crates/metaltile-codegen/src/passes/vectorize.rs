//! Vectorization pass: promote scalar ops to `half4`/`half8`/`float4` etc.
//!
//! Scans for consecutive scalar Load/Store ops with contiguous indices and
//! replaces them with vectorized VectorLoad/VectorStore ops.

use std::collections::BTreeMap;

use metaltile_core::{
    dtype::DType,
    error::Result,
    ir::{Block, BlockId, IndexExpr, Kernel, Op, Param, ValueId},
};

pub struct VectorizePass;

impl super::Pass for VectorizePass {
    fn name(&self) -> &str { "vectorize" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        let block_ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();
        // Pass params separately to avoid borrow conflict.
        let params = &kernel.params;
        for bid in &block_ids {
            if let Some(block) = kernel.blocks.get_mut(bid) {
                vectorize_block(block, params);
            }
        }
        vectorize_block(&mut kernel.body, params);
        Ok(())
    }
}

/// Maximum vector width to try (2, 4, or 8).
const MAX_VEC_LEN: usize = 4;

fn vectorize_block(block: &mut Block, params: &[Param]) {
    let n = block.ops.len();
    let mut skip: Vec<bool> = vec![false; n];

    // Phase 1: find contiguous Load sequences.
    for i in 0..n {
        if skip[i] {
            continue;
        }

        // Check for a run of identical Load ops with consecutive indices.
        if let Op::Load { src, indices, .. } = &block.ops[i] {
            if indices.len() != 1 {
                continue;
            }
            let base = match &indices[0] {
                IndexExpr::Value(v) | IndexExpr::Range(v, _) => *v,
                _ => continue,
            };
            let Some(param) = params.iter().find(|p| &p.name == src) else {
                continue;
            };
            if !matches!(param.dtype, DType::F16 | DType::F32) {
                continue;
            }

            // Collect a run of up to MAX_VEC_LEN consecutive loads.
            let mut run_indices: Vec<usize> = vec![i];
            for (j, skip_val) in
                skip.iter().enumerate().skip(i + 1).take(n.min(i + MAX_VEC_LEN) - (i + 1))
            {
                if *skip_val {
                    break;
                }
                match &block.ops[j] {
                    Op::Load { src: s2, indices: idx2, .. } if *s2 == *src && idx2.len() == 1 => {
                        let next_base = match &idx2[0] {
                            IndexExpr::Value(v) | IndexExpr::Range(v, _) => *v,
                            _ => break,
                        };
                        // Check contiguity: next_base should be base + (j - i).
                        // For simple index expressions, we check the ValueId chain.
                        // Accept if the index is the same ValueId + constant offset.
                        if next_base.as_u32() == base.as_u32().wrapping_add((j - i) as u32) {
                            run_indices.push(j);
                        } else {
                            break;
                        }
                    },
                    _ => break,
                }
            }

            if run_indices.len() >= 2 {
                let vlen = run_indices.len() as u32;
                // Replace the first load with a VectorLoad, skip the rest.
                block.ops[i] = Op::VectorLoad { src: src.clone(), byte_offset: base, len: vlen };
                // Remove the skipped loads and their results.
                for &idx in run_indices[1..].iter().rev() {
                    skip[idx] = true;
                }
            } else {
                continue;
            }
        }
    }

    // Phase 2: find contiguous Store sequences (similar pattern).
    for i in 0..n {
        if skip[i] {
            continue;
        }

        if let Op::Store { dst, indices, .. } = &block.ops[i] {
            if indices.len() != 1 {
                continue;
            }
            let base = match &indices[0] {
                IndexExpr::Value(v) | IndexExpr::Range(v, _) => *v,
                _ => continue,
            };
            let Some(param) = params.iter().find(|p| &p.name == dst) else {
                continue;
            };
            if !matches!(param.dtype, DType::F16 | DType::F32) {
                continue;
            }

            let mut run_indices: Vec<usize> = vec![i];
            for (j, skip_val) in
                skip.iter().enumerate().skip(i + 1).take(n.min(i + MAX_VEC_LEN) - (i + 1))
            {
                if *skip_val {
                    break;
                }
                match &block.ops[j] {
                    Op::Store { dst: d2, indices: idx2, .. } if *d2 == *dst && idx2.len() == 1 => {
                        let next_base = match &idx2[0] {
                            IndexExpr::Value(v) | IndexExpr::Range(v, _) => *v,
                            _ => break,
                        };
                        if next_base.as_u32() == base.as_u32().wrapping_add((j - i) as u32) {
                            run_indices.push(j);
                        } else {
                            break;
                        }
                    },
                    _ => break,
                }
            }

            if run_indices.len() >= 2 {
                let vlen = run_indices.len() as u32;
                // Gather all the values being stored — they'll need to be packed into a vector somehow.
                let first_value = match &block.ops[i] {
                    Op::Store { value, .. } => *value,
                    _ => continue,
                };
                block.ops[i] = Op::VectorStore {
                    dst: dst.clone(),
                    byte_offset: base,
                    len: vlen,
                    value: first_value,
                };
                for &idx in run_indices[1..].iter().rev() {
                    skip[idx] = true;
                }
            }
        }
    }

    // Phase 3: rebuild the block without skipped ops.
    let mut new_ops: Vec<Op> = Vec::new();
    let mut new_results: Vec<Option<ValueId>> = Vec::new();

    // Map old op index → new ValueId that replaced its result.
    // For vectorized runs, all results from the scatters are now produced
    // by the VectorLoad at the start of the run.
    let mut result_remap: BTreeMap<usize, ValueId> = BTreeMap::new();

    {
        let mut i = 0;
        while i < n {
            if skip[i] {
                i += 1;
                continue;
            }

            // Check if this is a VectorLoad (start of a run).
            if let Op::VectorLoad { len, .. } = &block.ops[i] {
                let vlen = *len as usize;
                // results is now always parallel to ops
                let first_vid = block.results[i].unwrap_or(ValueId::new(0));
                // The first result stays, but subsequent results are subsumed.
                for k in 1..vlen {
                    result_remap.insert(i + k, first_vid);
                }
            }

            new_ops.push(std::mem::replace(&mut block.ops[i], Op::Const { value: 0 }));
            // results is parallel to ops — always push
            new_results.push(block.results[i]);
            i += 1;
        }
    }

    // Remap value references: any op referencing a skipped result should now
    // reference the vector result.
    for op in new_ops.iter_mut() {
        remap_values_in_op(op, &result_remap);
    }

    block.ops = new_ops;
    block.results = new_results;
}

fn remap_values_in_op(op: &mut Op, remap: &BTreeMap<usize, ValueId>) {
    let s = |v: &mut ValueId| {
        // Try to find if this v corresponds to an old result index.
        for (&old_idx, &new_vid) in remap {
            if v.as_u32() == old_idx as u32 {
                *v = new_vid;
                return;
            }
        }
    };
    match op {
        Op::BinOp { lhs, rhs, .. } => {
            s(lhs);
            s(rhs);
        },
        Op::UnaryOp { value, .. }
        | Op::Activation { value, .. }
        | Op::Cast { value, .. }
        | Op::Reduce { value, .. }
        | Op::Transpose { value }
        | Op::Slice { value, .. }
        | Op::Broadcast { value, .. } => {
            s(value);
        },
        Op::Select { cond, on_true, on_false } => {
            s(cond);
            s(on_true);
            s(on_false);
        },
        Op::Dot { a, b } => {
            s(a);
            s(b);
        },
        Op::Store { value, .. } => {
            s(value);
        },
        Op::Loop { start, end, step, .. } => {
            s(start);
            s(end);
            s(step);
        },
        Op::VectorStore { value, .. } => {
            s(value);
        },
        Op::FusedElementwise { ops } =>
            for sub in ops.iter_mut() {
                remap_values_in_op(sub, remap);
            },
        _ => {},
    }
}
