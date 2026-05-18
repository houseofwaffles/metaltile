//! Shared test helpers for SDPA decode integration tests.

#![allow(dead_code)]

use metaltile_core::dtype::DType;

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
}

pub fn pack_bytes(vals: &[f32], dt: Dt) -> Vec<u8> {
    match dt {
        Dt::F32 => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Dt::F16 => vals.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect(),
        Dt::Bf16 => vals.iter().flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()).collect(),
    }
}

pub fn unpack_bytes(bytes: &[u8], dt: Dt) -> Vec<f32> {
    match dt {
        Dt::F32 =>
            bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
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
    assert!(s.n_q_heads.is_multiple_of(s.n_kv_heads));
    let gqa = s.n_q_heads / s.n_kv_heads;
    let mut out = vec![0.0f32; s.n_q_heads * s.head_dim];
    for qh in 0..s.n_q_heads {
        let kvh = qh / gqa;
        let q_off = qh * s.head_dim;
        let kv_slab = kvh * s.n_kv * s.head_dim;
        let mut scores = vec![0.0f32; s.n_kv];
        for (t, score) in scores.iter_mut().enumerate() {
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
            *score = (*score - m).exp();
            sum += *score;
        }
        let inv = 1.0 / sum;
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
