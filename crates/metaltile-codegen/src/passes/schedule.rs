//! Scheduling pass: annotates Dot ops with tile dimensions for the autotuner.
//!
//! The Autotuner varies tile sizes at runtime. This pass stores the chosen
//! tile dimensions on the Kernel so MslGenerator can read them directly,
//! eliminating the need to modify MslConfig on every autotune iteration.

use std::collections::BTreeMap;

use metaltile_core::{
    error::Result,
    ir::{Block, Kernel, Op, ValueId},
};

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
    annotations: &mut BTreeMap<ValueId, (u32, u32, u32)>,
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
