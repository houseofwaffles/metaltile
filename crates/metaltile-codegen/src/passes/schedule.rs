//! Scheduling — annotate Dot ops with tile dimensions for the autotuner.
//!
//! The Autotuner varies tile sizes at runtime.  This pass stores the chosen
//! tile dimensions on the Kernel so the MSL generator can read them directly,
//! eliminating the need to modify MSL configuration on every autotune iteration.
//!
//! The schedule-separated design (algorithm vs. schedule) follows the Halide
//! philosophy: the IR describes *what* to compute, the schedule describes *how*
//! (tile size, thread mapping).
//!
//! ## References
//! - Ragan-Kelley, Barnes, Adams, Paris, Durand & Amarasinghe (2013),
//!   "Halide: A Language and Compiler for Optimizing Parallelism, Locality,
//!   and Recomputation in Image Processing Pipelines", PLDI 2013.
//!   Established the algorithm/schedule separation.
//! - Bacon, Graham & Sharp (1994), "Compiler Transformations for High-
//!   Performance Computing", ACM Computing Surveys 26(4):345–420.
//!   Surveys loop tiling and scheduling transformations.

use metaltile_core::ir::{Block, Kernel, Op, ValueId};
use rustc_hash::FxHashMap;

use crate::error::Result;

/// A schedule configuration: how many threads per threadgroup and
/// how tiles map to those threads.
#[derive(Debug, Clone)]
pub struct ScheduleConfig {
    /// Threads per threadgroup (x, y, z).
    pub threads_per_threadgroup: (u32, u32, u32),
    /// Threadgroups per grid (x, y, z).
    pub threadgroups_per_grid: (u32, u32, u32),
    /// Dot tile dimensions (M, N, K).
    pub tile_dims: (u32, u32, u32),
    /// SIMD group size.
    pub simd_size: u32,
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        ScheduleConfig {
            threads_per_threadgroup: (256, 1, 1),
            threadgroups_per_grid: (1, 1, 1),
            tile_dims: (32, 32, 16),
            simd_size: 32,
        }
    }
}

pub struct SchedulePass {
    config: ScheduleConfig,
}

impl SchedulePass {
    pub fn new(config: ScheduleConfig) -> Self { SchedulePass { config } }
}

impl Default for SchedulePass {
    fn default() -> Self { SchedulePass::new(ScheduleConfig::default()) }
}

impl super::Pass for SchedulePass {
    fn name(&self) -> &str { "schedule" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        kernel.tile_annotations.clear();

        // Walk all blocks and annotate Op::Dot ops with tile dimensions.
        let block_ids: Vec<_> = kernel.blocks.keys().copied().collect();
        let mut next_vid_hint = 0u32;

        for bid in &block_ids {
            if let Some(block) = kernel.blocks.get(bid) {
                annotate_block(
                    block,
                    &self.config,
                    &mut kernel.tile_annotations,
                    &mut next_vid_hint,
                );
            }
        }
        // Also annotate the body block.
        annotate_block(
            &kernel.body,
            &self.config,
            &mut kernel.tile_annotations,
            &mut next_vid_hint,
        );

        Ok(())
    }
}

fn annotate_block(
    block: &Block,
    config: &ScheduleConfig,
    annotations: &mut FxHashMap<ValueId, (u32, u32, u32)>,
    next_vid_hint: &mut u32,
) {
    for (i, op) in block.ops.iter().enumerate() {
        if matches!(op, Op::Dot { .. }) {
            // Use the result ValueId as the key if available, otherwise generate a synthetic key.
            let key = block.results.get(i).and_then(|x| *x).unwrap_or_else(|| {
                let vid = ValueId::new(*next_vid_hint + 100_000);
                *next_vid_hint += 1;
                vid
            });
            annotations.insert(key, config.tile_dims);
        }
    }
}

#[cfg(test)]
mod tests {
    use metaltile_core::ir::{Block, BlockId, VarId};

    use super::*;
    use crate::passes::Pass;

    #[test]
    fn annotates_dot_ops_with_tile_dims() {
        let mut k = Kernel::new("schedule_dot");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::Dot { a: ValueId::new(0), b: ValueId::new(0) }, ValueId::new(1));
        SchedulePass::default().run(&mut k).unwrap();

        assert_eq!(k.tile_annotations.len(), 1);
        let (tm, tn, tk) = k.tile_annotations[&ValueId::new(1)];
        assert_eq!((tm, tn, tk), (32, 32, 16), "default tile dims should be (32, 32, 16)");
    }

    #[test]
    fn annotates_multiple_dots() {
        let mut k = Kernel::new("schedule_two_dots");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::Dot { a: ValueId::new(0), b: ValueId::new(0) }, ValueId::new(1));
        k.body.push_op(Op::Dot { a: ValueId::new(0), b: ValueId::new(0) }, ValueId::new(2));
        SchedulePass::default().run(&mut k).unwrap();
        assert_eq!(k.tile_annotations.len(), 2);
    }

    #[test]
    fn custom_config_respected() {
        let mut k = Kernel::new("schedule_custom");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::Dot { a: ValueId::new(0), b: ValueId::new(0) }, ValueId::new(1));

        let config = ScheduleConfig {
            threads_per_threadgroup: (128, 1, 1),
            threadgroups_per_grid: (2, 2, 1),
            tile_dims: (16, 16, 8),
            simd_size: 32,
        };
        SchedulePass::new(config).run(&mut k).unwrap();

        let (tm, tn, tk) = k.tile_annotations[&ValueId::new(1)];
        assert_eq!((tm, tn, tk), (16, 16, 8), "custom tile dims should be applied");
    }

    #[test]
    fn no_ops_no_annotations() {
        let mut k = Kernel::new("schedule_empty");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        SchedulePass::default().run(&mut k).unwrap();
        assert!(k.tile_annotations.is_empty(), "no Dot ops → no annotations");
    }

    #[test]
    fn annotates_dots_in_nested_blocks() {
        let mut k = Kernel::new("schedule_nested");
        k.body.push_op(Op::Const { value: 0 }, ValueId::new(0));
        k.body.push_op(Op::Const { value: 4 }, ValueId::new(1));
        k.body.push_op(Op::Const { value: 1 }, ValueId::new(2));

        let mut loop_body = Block::new(BlockId::new(1));
        loop_body.push_op(Op::Dot { a: ValueId::new(0), b: ValueId::new(0) }, ValueId::new(10));
        let body_id = k.add_block(loop_body);

        k.body.push_op_no_result(Op::Loop {
            var: VarId::new(0),
            start: ValueId::new(0),
            end: ValueId::new(1),
            step: ValueId::new(2),
            body: body_id,
        });

        SchedulePass::default().run(&mut k).unwrap();
        // Dot in nested block should also be annotated.
        assert_eq!(k.tile_annotations.len(), 1);
    }
}
