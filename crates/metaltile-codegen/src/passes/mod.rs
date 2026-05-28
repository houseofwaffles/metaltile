//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Pass Infrastructure — trait, runner, and pass ordering.
//!
//! Defines the [`Pass`] trait that all optimization passes implement, the
//! [`run_passes`] orchestration function, and [`PassStats`] for timing and
//! IR-size tracking.  Module declarations for all passes live here.

pub mod algebraic_simplify;
pub mod block_util;
pub mod const_fold;
pub mod copy_prop;
pub mod cse;
pub mod dead_store_elim;
pub mod dead_value_elim;
pub mod fma_fusion;
pub mod fusion;
pub mod if_conversion;
pub mod kernel_inline;
pub mod licm;
pub mod occupancy;
pub mod register_estimate;
pub mod remap;
pub mod schedule;
pub mod tile_lowering;
pub mod type_check;
pub mod unroll;
pub mod value_sink;
pub mod vectorize;

use std::time::Instant;

use metaltile_core::ir::Kernel;

use crate::error::Result;

/// A transformation pass on the IR.
pub trait Pass {
    fn name(&self) -> &str;
    fn run(&self, kernel: &mut Kernel) -> Result<()>;
}

/// Run a sequence of passes on a kernel.
pub fn run_passes(kernel: &mut Kernel, passes: &[Box<dyn Pass>]) -> Result<()> {
    for pass in passes {
        tracing::debug!(kernel = %kernel.name, pass = pass.name(), "running pass");
        pass.run(kernel)?;
    }
    Ok(())
}

/// Statistics for a single pass execution.
#[derive(Debug, Clone)]
pub struct PassStats {
    pub name: String,
    pub ops_before: usize,
    pub ops_after: usize,
    pub wall_us: u64,
}

/// Run a sequence of passes on a kernel, collecting statistics.
pub fn run_passes_with_stats(
    kernel: &mut Kernel,
    passes: &[Box<dyn Pass>],
) -> Result<Vec<PassStats>> {
    let mut stats = Vec::with_capacity(passes.len());

    for pass in passes {
        let ops_before = count_total_ops(kernel);
        let start = Instant::now();
        tracing::debug!(kernel = %kernel.name, pass = pass.name(), ops = ops_before, "running pass");
        pass.run(kernel)?;
        let elapsed = start.elapsed();
        let ops_after = count_total_ops(kernel);
        let s = PassStats {
            name: pass.name().to_string(),
            ops_before,
            ops_after,
            wall_us: elapsed.as_micros() as u64,
        };
        tracing::trace!(
            pass = pass.name(),
            wall_us = s.wall_us,
            ops_before = ops_before,
            ops_after = ops_after,
            eliminated = ops_before.saturating_sub(ops_after),
            "pass complete"
        );
        stats.push(s);
    }

    Ok(stats)
}

/// Count all ops across the kernel body and all nested blocks.
pub fn count_total_ops(kernel: &Kernel) -> usize {
    let mut total = kernel.body.ops.len();
    for block in kernel.blocks.values() {
        total += block.ops.len();
    }
    total
}

// ---------------------------------------------------------------------------
// PassRegistry — canonical pass list and name lookup
// ---------------------------------------------------------------------------

/// Registry of all available passes.
///
/// The canonical pass list lives here.  Consumers that need the full pipeline,
/// a pass-by-name lookup, or a named walk use [`PassRegistry`] instead of
/// hand-rolling their own list.  Adding a new pass only requires updating this
/// registry (and the `pub mod` declaration above).
pub struct PassRegistry;

impl PassRegistry {
    /// The standard pass order (names, in pipeline sequence).
    ///
    /// TypeCheck → ConstFold → AlgebraicSimplify → CopyProp → CSE → LICM
    ///   → IfConversion → ValueSink → Fusion → FmaFusion → Unroll
    ///   → Schedule → Vectorize → DeadStoreElim
    ///
    /// Each pass that can produce orphan SSA values (the producer is
    /// left in the block after the pass removes/redirects its last
    /// consumer) invokes
    /// `dead_value_elim::eliminate_dead_values(kernel)` at the end of
    /// its own `run()` — see #209/1.  Pre-#209/1 a separate
    /// `dead_value_elim` slot ran last in the pipeline to sweep up
    /// every pass's accumulated debris; with the per-pass
    /// postcondition in place, that slot is redundant and removed.
    /// `DeadValueElimPass` remains a registry-callable pass for tests
    /// and tooling that want to invoke DCE explicitly.
    pub fn order() -> &'static [&'static str] {
        &[
            "kernel_inline",
            "type_check",
            "const_fold",
            "algebraic_simplify",
            "copy_prop",
            "cse",
            "licm",
            "if_conversion",
            "value_sink",
            "fusion",
            // FmaFusion runs after the FusedElementwise chain builder
            // and before Unroll — it rewrites `Add(Mul, c)` → `Fma`
            // in-place and relies on type inference, so it needs to
            // run after `type_check`.  The standalone Mul becomes a
            // dead value that the pass's own DCE postcondition sweeps.
            "fma_fusion",
            "unroll",
            "schedule",
            "vectorize",
            "dead_store_elim",
        ]
    }

    /// Look up a pass by name.  Returns `None` for unknown names.
    pub fn get(name: &str) -> Option<Box<dyn Pass>> {
        match name {
            "kernel_inline" => Some(Box::new(kernel_inline::KernelInlinePass)),
            "type_check" => Some(Box::new(type_check::TypeCheckPass)),
            "const_fold" => Some(Box::new(const_fold::ConstFoldPass::new())),
            "algebraic_simplify" => Some(Box::new(algebraic_simplify::AlgebraicSimplifyPass)),
            "copy_prop" => Some(Box::new(copy_prop::CopyPropPass)),
            "cse" => Some(Box::new(cse::CsePass)),
            "licm" => Some(Box::new(licm::LicmPass)),
            "if_conversion" => Some(Box::new(if_conversion::IfConversionPass)),
            "value_sink" => Some(Box::new(value_sink::ValueSinkPass)),
            "tile_lowering" => Some(Box::new(tile_lowering::TileLoweringPass::default())),
            "fusion" => Some(Box::new(fusion::FusionPass)),
            "fma_fusion" => Some(Box::new(fma_fusion::FmaFusionPass)),
            "unroll" => Some(Box::new(unroll::UnrollPass::default())),
            "schedule" => Some(Box::new(schedule::SchedulePass::default())),
            "vectorize" => Some(Box::new(vectorize::VectorizePass)),
            "cse_2" => Some(Box::new(cse::CsePass)),
            "const_fold_2" => Some(Box::new(const_fold::ConstFoldPass::new())),
            "dead_store_elim" => Some(Box::new(dead_store_elim::DeadStoreElimPass)),
            "dead_value_elim" => Some(Box::new(dead_value_elim::DeadValueElimPass)),
            _ => None,
        }
    }

    /// Build the standard pipeline (all passes, in canonical order).
    pub fn standard_pipeline() -> Vec<Box<dyn Pass>> {
        Self::order().iter().filter_map(|&n| Self::get(n)).collect()
    }

    /// Return the standard pipeline with names attached (for debug/inspect).
    pub fn standard_with_names() -> Vec<(&'static str, Box<dyn Pass>)> {
        Self::order().iter().filter_map(|&n| Some((n, Self::get(n)?))).collect()
    }

    /// Return sorted pass names (for usage / error messages).
    pub fn names() -> Vec<&'static str> {
        let mut n: Vec<_> = Self::order().to_vec();
        n.sort_unstable();
        n
    }
}

// ---------------------------------------------------------------------------
// PipelineBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing an optimization pipeline with optional overrides.
pub struct PipelineBuilder {
    passes: Vec<Box<dyn Pass>>,
}

impl PipelineBuilder {
    /// Create a builder with the standard pipeline from [`PassRegistry`].
    pub fn standard() -> Self { PipelineBuilder { passes: PassRegistry::standard_pipeline() } }

    /// Remove a pass by name from the pipeline.
    pub fn without(mut self, name: &str) -> Self {
        self.passes.retain(|p| p.name() != name);
        self
    }

    /// Override the unroll factor.
    pub fn with_unroll_factor(self, factor: u32) -> Self {
        let mut passes = self.passes;
        for p in passes.iter_mut() {
            if p.name() == "unroll" {
                *p = Box::new(unroll::UnrollPass::new(factor));
                break;
            }
        }
        PipelineBuilder { passes }
    }

    /// Build the final pass list.
    pub fn build(self) -> Vec<Box<dyn Pass>> { self.passes }
}

/// Standard optimization pipeline.
///
/// Convenience wrapper around [`PassRegistry::standard_pipeline`].
pub fn standard_pipeline() -> Vec<Box<dyn Pass>> { PassRegistry::standard_pipeline() }
