//! Llama-style RoPE with optional Llama-3 frequency-band scaling.
//! Per-token decode form (single position constexpr), generic over T.
//!
//! Different from `mt_rope` (in `mlx/rope.rs`):
//!   - decode-only (no batch / seq grid)
//!   - generic dtype (mt_rope is f16-only)
//!   - supports Llama-3 wavelength banding (low / high / smoothed)
//!
//! For each (head, i in 0..head_dim/2):
//!
//!   base inv_freq = 1 / theta_base^(2i / head_dim)
//!   wavelen       = 2*pi / inv_freq
//!   if wavelen > low_freq_wavelen:        inv_freq /= scale_factor      (low-freq band)
//!   else if wavelen < high_freq_wavelen:  inv_freq                       (high-freq band)
//!   else (medium band):                   smoothed interpolation
//!
//! To turn scaling OFF, pass scale_factor=1, low_freq_factor=1,
//! high_freq_factor=1, original_max_position=very_large (e.g. 1e9).
//!
//! Codegen-only. Validated end-to-end in FFAI integration tests.

use metaltile::kernel;
use metaltile_core::ir::KernelMode;

use crate::{
    bench_types::DType,
    spec::{BenchDispatch, BenchSpec},
};

#[kernel]
pub fn rope_llama<T>(
    qk: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] half_dim: u32,
    #[constexpr] position: u32,
    #[constexpr] theta_base: f32,
    #[constexpr] scale_factor: f32,
    #[constexpr] low_freq_factor: f32,
    #[constexpr] high_freq_factor: f32,
    #[constexpr] original_max_position: f32,
) {
    let head = program_id::<0>();
    let i = program_id::<1>();

    let i_f = i.cast::<f32>();
    let half_f = half_dim.cast::<f32>();
    let inv_freq_base = exp2(-i_f * log2(theta_base) / half_f);

    let two_pi = 6.283185307179586f32;
    let wavelen = two_pi / inv_freq_base;
    let low_freq_wavelen = original_max_position / low_freq_factor;
    let high_freq_wavelen = original_max_position / high_freq_factor;

    let scaled = inv_freq_base / scale_factor;
    let smooth_num = original_max_position / wavelen - low_freq_factor;
    let smooth_den = high_freq_factor - low_freq_factor;
    let s = smooth_num / smooth_den;
    let smoothed = (1.0f32 - s) * scaled + s * inv_freq_base;

    let is_low_freq = wavelen > low_freq_wavelen;
    let is_high_freq = wavelen < high_freq_wavelen;
    let inv_freq = select(is_low_freq, scaled, select(is_high_freq, inv_freq_base, smoothed));

    let pos_f = position.cast::<f32>();
    let theta = pos_f * inv_freq;
    let cos_t = cos(theta);
    let sin_t = sin(theta);

    let base = head * head_dim;
    let i1 = base + i;
    let i2 = base + i + half_dim;

    let x1 = load(qk[i1]).cast::<f32>();
    let x2 = load(qk[i2]).cast::<f32>();
    let o1 = x1 * cos_t - x2 * sin_t;
    let o2 = x1 * sin_t + x2 * cos_t;

    store(out[i1], o1.cast::<T>());
    store(out[i2], o2.cast::<T>());
}

inventory::submit! {
    BenchSpec {
        op: "rope",
        subop: "rope_llama",
        kernel_name: "rope_llama",
        kernel_ir: rope_llama::kernel_ir_for,
        dtypes: &[DType::F32, DType::F16, DType::BF16],
        tol: 0.0,
        mlx_src: None,
        mlx_pattern: None,
        shapes: &[],
        dispatch: BenchDispatch::Generic,
        kernel_mode: Some(KernelMode::Grid3D),
    }
}
