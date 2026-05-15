//! Tile lowering pass: expands high-level tile ops into thread-mapped MSL.
//!
//! Phase 2: Op::Dot → full tiled matmul.
//! The actual lowering happens in the MSL generator (msl.rs::emit_tiled).
//! This pass validates and annotates the IR for tile operations.

use metaltile_core::{error::Result, ir::Kernel};

#[derive(Debug, Clone)]
pub struct TileSchedule {
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub threads: (u32, u32, u32),
    pub rows_per_thread: u32,
    pub cols_per_thread: u32,
}

impl Default for TileSchedule {
    fn default() -> Self {
        TileSchedule {
            tile_m: 64,
            tile_n: 64,
            tile_k: 32,
            threads: (16, 16, 1),
            rows_per_thread: 4,
            cols_per_thread: 4,
        }
    }
}

pub struct TileLoweringPass {
    #[allow(dead_code)]
    schedule: TileSchedule,
}

impl TileLoweringPass {
    pub fn new(#[allow(dead_code)] schedule: TileSchedule) -> Self { TileLoweringPass { schedule } }
}

impl Default for TileLoweringPass {
    fn default() -> Self { TileLoweringPass::new(TileSchedule::default()) }
}

impl super::Pass for TileLoweringPass {
    fn name(&self) -> &str { "tile_lowering" }
    fn run(&self, _kernel: &mut Kernel) -> Result<()> {
        // Lowering happens in MSL generator via emit_tiled.
        // This pass is a placeholder for future IR-level expansion.
        Ok(())
    }
}
