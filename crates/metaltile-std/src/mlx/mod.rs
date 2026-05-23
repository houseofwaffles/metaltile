//! MLX-compared kernels.
//!
//! Every kernel in this submodule has (or can have) a side-by-side
//! correctness/perf comparison against an MLX reference kernel — the
//! benches embed MLX's `.metal` source via `metal_file = "..."` and
//! dispatch the MLX kernel through `compile_with_bool_constants` / a
//! constructed kernel name.
//!
//! When a kernel can't be directly compared today (MLX template not
//! shipped at the pinned commit, or the comparison isn't wired yet)
//! but the implementation faithfully mirrors MLX semantics and is
//! expected to wire up eventually, it lives in `ffai/` until the
//! comparison lands.

pub mod arange;
pub mod arg_reduce;
pub mod binary;
pub mod binary_two;
pub mod copy;
pub mod fft;
pub mod fp_quantized;
pub mod fp_quantized_mma;
#[cfg(feature = "nax")]
pub mod fp_quantized_nax;
pub mod fused_gate_activation;
pub mod gather_axis;
pub mod gemv;
pub mod gemv_masked;
pub mod hadamard;
pub mod hadamard_m;
pub mod indexing;
pub mod layer_norm;
pub mod logsumexp;
pub mod quantized;
pub mod quantized_mma_dynamic_m;
pub mod quantized_mpp;
pub mod quantized_mpp_int8;
#[cfg(feature = "nax")]
pub mod quantized_nax;
#[cfg(feature = "nax")]
pub mod quantized_nax_int8;
pub mod random;
pub mod reduce;
pub mod rms_norm;
pub mod rope;
pub mod scaled_dot_product_attention;
pub mod scan;
pub mod scatter_axis;
pub mod sdpa_vector;
pub mod sgload_smoke;
pub mod softmax;
pub mod sort;
pub mod steel;
pub mod strided;
pub mod swiglu;
pub mod ternary;
pub mod unary;

// `conv.rs` and `shared.rs` are placeholder/stale stubs left over from
// the old `metaltile-bench` crate. They reference `crate::runner` which
// lives in `metaltile-cli`, so they don't compile — kept on disk for
// the kernel docs / future-work notes but intentionally not declared
// here. Delete or port when those kernels land in the #[kernel] DSL.
// `fft.rs` and `fence.rs` are now real `#[kernel]` ports (declared
// above / in `ffai/`).
