//! Pass infrastructure and optimization passes.

pub mod const_fold;
pub mod fusion;
pub mod schedule;
pub mod tile_lowering;
pub mod type_check;
pub mod vectorize;

use metaltile_core::ir::Kernel;

/// A transformation pass on the IR.
pub trait Pass {
    fn name(&self) -> &str;
    fn run(&self, kernel: &mut Kernel) -> metaltile_core::error::Result<()>;
}

/// Run a sequence of passes on a kernel.
pub fn run_passes(
    kernel: &mut Kernel,
    passes: &[Box<dyn Pass>],
) -> metaltile_core::error::Result<()> {
    for pass in passes {
        pass.run(kernel)?;
    }
    Ok(())
}

/// Standard optimization pipeline (PLAN.md §7 order):
/// 1. Type & shape checking
/// 2. Constant folding & DCE
/// 3. Tile lowering (high-level tile ops → expanded IR)
/// 4. Fusion (per-layer elementwise chains)
/// 5. Schedule selection (apply autotuner config)
/// 6. Vectorization (scalar → vec4/vec8)
pub fn standard_pipeline() -> Vec<Box<dyn Pass>> {
    vec![
        Box::new(type_check::TypeCheckPass),
        Box::new(const_fold::ConstFoldPass::new()),
        Box::new(tile_lowering::TileLoweringPass::default()),
        Box::new(fusion::FusionPass),
        Box::new(schedule::SchedulePass::default()),
        Box::new(vectorize::VectorizePass),
    ]
}
