//! FFAI / model-specific kernels.
//!
//! Kernels here are ports from FFAI / mlx-swift-lm / Eric's `mlx` fork
//! that don't have a matching template in mainline MLX at the pinned
//! commit (see `metaltile-std/build.rs` — `MLX_COMMIT`). They register
//! a `BenchSpec` so `tile build` / `tile inspect` can find them, but
//! the spec uses `shapes: &[]` and `dispatch: BenchDispatch::Generic`,
//! so `tile bench` skips them (no MLX side-by-side, no GPU shapes).
//!
//! Correctness for these kernels is validated end-to-end in FFAI's
//! integration tests against real models. Once a kernel has been
//! verified there, the shape spec / bench dispatch can be added back
//! here so `tile bench` can track it for regressions — and if its MLX
//! counterpart lands in mainline at a future pin, the file moves to
//! `mlx/`.

pub mod arg_reduce;
pub mod dequant_gather;
pub mod dequant_gemv;
pub mod gather;
pub mod kv_cache;
pub mod rope_llama;
pub mod sampling;
pub mod sdpa_decode;
pub mod ssm;
