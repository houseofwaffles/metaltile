//! Shared test helpers for metaltile-std GPU integration tests.

#![allow(dead_code)]

use std::sync::{Mutex, MutexGuard, OnceLock};

use metaltile_core::dtype::DType;

/// Serialise GPU dispatches across all integration tests that pull in
/// this module. cargo runs integration tests in parallel by default;
/// concurrent dispatches on the shared Metal pipeline race the PSO
/// cache + library compilation path and surface as cross-test numeric
/// corruption (caught e.g. when an f16 test ran after an f32 test in
/// a single `cargo test` invocation and produced output ≈ 0.45× the
/// expected magnitude). Lighter than requiring `--test-threads=1` at
/// the command line.
///
/// Tests that grab this lock at the top of their body serialise across
/// the entire integration-test binary they're linked into. Mutex
/// poisoning unwraps to `into_inner()` so a panic in one test still
/// lets the others run.
pub fn gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Clone, Copy)]
pub enum Dt {
    F32,
    F16,
    Bf16,
}

impl Dt {
    pub fn bytes(self) -> usize {
        match self {
            Dt::F32 => 4,
            Dt::F16 | Dt::Bf16 => 2,
        }
    }
    pub fn to_dtype(self) -> DType {
        match self {
            Dt::F32 => DType::F32,
            Dt::F16 => DType::F16,
            Dt::Bf16 => DType::BF16,
        }
    }
    /// Round-trip a value through this dtype's precision. Used by
    /// per-dtype correctness oracles so the CPU reference sees the
    /// same load-cast quantisation the kernel does (no-op for f32,
    /// 10-bit mantissa for f16, 7-bit for bf16).
    pub fn round(self, v: f32) -> f32 {
        match self {
            Dt::F32 => v,
            Dt::F16 => half::f16::from_f32(v).to_f32(),
            Dt::Bf16 => half::bf16::from_f32(v).to_f32(),
        }
    }
}

pub fn pack_bytes(vals: &[f32], dt: Dt) -> Vec<u8> {
    match dt {
        // Host is little-endian on all current Metal targets — single
        // memcpy beats `flat_map(to_le_bytes)`'s per-element iter churn.
        // Noticeable on the SWA perf bench's 4M-element K/V ramps.
        Dt::F32 => bytemuck::cast_slice::<f32, u8>(vals).to_vec(),
        Dt::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
        Dt::Bf16 => vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
    }
}

pub fn unpack_bytes(bytes: &[u8], dt: Dt) -> Vec<f32> {
    match dt {
        Dt::F32 => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
        Dt::F16 =>
            bytes.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        Dt::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
    }
}

pub struct SdpaShape {
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_kv: usize,
    pub scale: f32,
}

/// Naive triple-loop SDPA reference: `O = softmax(Q · Kᵀ · scale) · V`
/// per Q head, GQA via `kv_head = q_head / heads_per_group`, fp32.
pub fn naive_sdpa_f32(q: &[f32], k: &[f32], v: &[f32], s: &SdpaShape) -> Vec<f32> {
    naive_sdpa_swa_f32(q, k, v, s, 0, 0)
}

/// Sliding-window + sink-token SDPA reference. Attended positions are
/// `[0, sink_end) ∪ [window_start, n_kv)`; masked positions contribute
/// nothing (no score, no softmax weight). Caller must satisfy
/// `window_start >= sink_end` and `window_start <= n_kv`, the same
/// preconditions the GPU kernel enforces. With `sink_end = 0` and
/// `window_start = 0` this is the dense reference (used by
/// [`naive_sdpa_f32`]).
pub fn naive_sdpa_swa_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    s: &SdpaShape,
    sink_end: usize,
    window_start: usize,
) -> Vec<f32> {
    assert!(s.n_q_heads.is_multiple_of(s.n_kv_heads));
    assert!(
        window_start >= sink_end,
        "window_start must be >= sink_end (overlap would double-count)"
    );
    assert!(window_start <= s.n_kv && sink_end <= s.n_kv);
    let gqa = s.n_q_heads / s.n_kv_heads;
    let mut out = vec![0.0f32; s.n_q_heads * s.head_dim];
    let attended = |t: usize| t < sink_end || t >= window_start;
    for qh in 0..s.n_q_heads {
        let kvh = qh / gqa;
        let q_off = qh * s.head_dim;
        let kv_slab = kvh * s.n_kv * s.head_dim;
        let mut scores = vec![f32::NEG_INFINITY; s.n_kv];
        for (t, score) in scores.iter_mut().enumerate() {
            if !attended(t) {
                continue;
            }
            let k_off = kv_slab + t * s.head_dim;
            let mut dot = 0.0f32;
            for d in 0..s.head_dim {
                dot += q[q_off + d] * k[k_off + d];
            }
            *score = dot * s.scale;
        }
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for score in scores.iter_mut() {
            if score.is_finite() {
                *score = (*score - m).exp();
                sum += *score;
            } else {
                *score = 0.0;
            }
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for d in 0..s.head_dim {
            let mut acc = 0.0f32;
            for (t, score) in scores.iter().enumerate() {
                acc += *score * inv * v[kv_slab + t * s.head_dim + d];
            }
            out[q_off + d] = acc;
        }
    }
    out
}

/// Deterministic init pattern — small repeating modulus avoids both
/// degenerate all-zero softmax and uniform-value short-circuits.
pub fn ramp(n: usize, modulus: usize, offset: f32) -> Vec<f32> {
    (0..n).map(|i| ((i % modulus) as f32 - offset) * 0.05).collect()
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}
