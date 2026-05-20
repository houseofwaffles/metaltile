//! End-to-end correctness for MoE orchestration kernels — router
//! top-k (plus future permute / unpermute).
//!
//! Compares GPU output to a straight CPU reference. The reference is
//! a faithful re-statement of the kernel algorithm: k iterative
//! argmax passes with mask of previously-chosen indices, then softmax
//! over the chosen k values.

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;

mod common;

use common::gpu_lock;
use metaltile_core::dtype::DType;
use metaltile_runtime::Context;
use metaltile_std::ffai::moe::{mt_moe_permute, mt_moe_router_topk, mt_moe_unpermute};

#[allow(clippy::too_many_arguments)]
fn run_topk(
    ctx: &Context,
    dtype: DType,
    router_logits_bytes: &[u8],
    n_rows: usize,
    n_experts: usize,
    k: usize,
    norm_topk_prob: u32,
    out_w_bytes_per_elem: usize,
) -> (Vec<u32>, Vec<u8>) {
    assert!(k <= 32, "kernel pins k ≤ 32 (tg_chosen_* allocs are 32 slots)");
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("router_logits".into(), router_logits_bytes.to_vec());
    buffers.insert("indices_out".into(), vec![0u8; n_rows * k * 4]);
    buffers.insert("weights_out".into(), vec![0u8; n_rows * k * out_w_bytes_per_elem]);
    buffers.insert("n_experts".into(), (n_experts as u32).to_le_bytes().to_vec());
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());
    buffers.insert("norm_topk_prob".into(), norm_topk_prob.to_le_bytes().to_vec());

    let mut kernel = mt_moe_router_topk::kernel_ir_for(dtype);
    kernel.mode = metaltile_core::ir::KernelMode::Reduction;
    // Grid: one TG per token row. tpg=32 (single simdgroup) — kernel invariant.
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_rows, 1, 1], [32, 1, 1])
        .expect("dispatch_with_grid should succeed");
    let idx_bytes = result.outputs.get("indices_out").expect("indices_out").clone();
    let w_bytes = result.outputs.get("weights_out").expect("weights_out").clone();
    let indices: Vec<u32> =
        idx_bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    (indices, w_bytes)
}

fn cpu_topk_reference(
    router_logits: &[f32],
    n_rows: usize,
    n_experts: usize,
    k: usize,
) -> (Vec<u32>, Vec<f32>) {
    let mut indices = vec![0u32; n_rows * k];
    let mut weights = vec![0.0f32; n_rows * k];
    for row in 0..n_rows {
        let row_base = row * n_experts;
        let mut chosen = Vec::with_capacity(k);
        let mut chosen_vals = Vec::with_capacity(k);
        for _ in 0..k {
            let mut best_val = f32::NEG_INFINITY;
            let mut best_idx = 0usize;
            for j in 0..n_experts {
                if chosen.contains(&(j as u32)) {
                    continue;
                }
                let v = router_logits[row_base + j];
                if v > best_val {
                    best_val = v;
                    best_idx = j;
                }
            }
            chosen.push(best_idx as u32);
            chosen_vals.push(best_val);
        }
        // Softmax over chosen.
        let max_v = chosen_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = chosen_vals.iter().map(|&v| (v - max_v).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        for j in 0..k {
            indices[row * k + j] = chosen[j];
            weights[row * k + j] = exp_vals[j] / sum_exp;
        }
    }
    (indices, weights)
}

#[test]
fn mt_moe_router_topk_matches_cpu_reference_f32() {
    // Small shape covering the simdgroup edge case (n_experts > 32
    // so each lane scans 2+ entries) and exercising the chosen-mask
    // logic with k > 1.
    let n_rows = 8usize;
    let n_experts = 64usize;
    let k = 4usize;

    // Deterministic logits — distinct values so top-k is unambiguous.
    let logits: Vec<f32> = (0..n_rows * n_experts)
        .map(|i| ((i as f32 * 0.13) % 7.0) - 3.5 + (i as f32 * 0.001))
        .collect();
    let (ref_idx, ref_w) = cpu_topk_reference(&logits, n_rows, n_experts, k);

    let logits_bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let (gpu_idx, gpu_w_bytes) =
        run_topk(&ctx, DType::F32, &logits_bytes, n_rows, n_experts, k, 1, 4);
    let gpu_w: Vec<f32> =
        gpu_w_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    // Indices must match exactly.
    assert_eq!(
        gpu_idx, ref_idx,
        "indices mismatch — GPU vs CPU reference\nGPU: {gpu_idx:?}\nCPU: {ref_idx:?}",
    );

    // Weights match within fp32 softmax tolerance.
    let mut max_diff = 0.0f32;
    for (i, (&g, &r)) in gpu_w.iter().zip(ref_w.iter()).enumerate() {
        let d = (g - r).abs();
        if d > max_diff {
            max_diff = d;
            assert!(d < 1e-5, "weight[{i}] diverges: gpu={g:.6} ref={r:.6} diff={d:.2e}");
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_unpermute(
    ctx: &Context,
    dtype: DType,
    expert_outputs_bytes: &[u8],
    inv_perm: &[u32],
    weights_bytes: &[u8],
    n_rows: usize,
    hidden: usize,
    k: usize,
    elem_bytes: usize,
) -> Vec<u8> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("expert_outputs".into(), expert_outputs_bytes.to_vec());
    buffers.insert("inv_perm".into(), inv_perm.iter().flat_map(|v| v.to_le_bytes()).collect());
    buffers.insert("top_k_weights".into(), weights_bytes.to_vec());
    buffers.insert("out".into(), vec![0u8; n_rows * hidden * elem_bytes]);
    buffers.insert("hidden".into(), (hidden as u32).to_le_bytes().to_vec());
    buffers.insert("k".into(), (k as u32).to_le_bytes().to_vec());

    let mut kernel = mt_moe_unpermute::kernel_ir_for(dtype);
    kernel.mode = metaltile_core::ir::KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_rows, 1, 1], [128, 1, 1])
        .expect("dispatch_with_grid should succeed");
    result.outputs.get("out").expect("out").clone()
}

#[test]
fn mt_moe_unpermute_matches_cpu_reference_f32() {
    let n_rows = 4usize;
    let hidden = 256usize;
    let k = 4usize;
    let total_expert_slots = n_rows * k;

    // Deterministic expert outputs.
    let expert_outputs: Vec<f32> =
        (0..total_expert_slots * hidden).map(|i| ((i as f32 * 0.07) % 11.0) - 5.5).collect();
    // Identity inv_perm — each (token i, slot j) stored at i*k+j.
    let inv_perm: Vec<u32> = (0..total_expert_slots as u32).collect();
    // Top-k weights — normalized so each row sums to 1.0.
    let raw_weights: Vec<f32> = (0..n_rows * k).map(|i| 0.1 + (i as f32 * 0.03)).collect();
    let mut weights = vec![0.0f32; n_rows * k];
    for row in 0..n_rows {
        let row_sum: f32 = raw_weights[row * k..(row + 1) * k].iter().sum();
        for j in 0..k {
            weights[row * k + j] = raw_weights[row * k + j] / row_sum;
        }
    }

    // CPU reference.
    let mut ref_out = vec![0.0f32; n_rows * hidden];
    for token in 0..n_rows {
        for h in 0..hidden {
            let mut acc = 0.0f32;
            for j in 0..k {
                let pos = inv_perm[token * k + j] as usize;
                acc += weights[token * k + j] * expert_outputs[pos * hidden + h];
            }
            ref_out[token * hidden + h] = acc;
        }
    }

    let exp_bytes: Vec<u8> = expert_outputs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let w_bytes: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().expect("Context::new on macOS");
    let out_bytes =
        run_unpermute(&ctx, DType::F32, &exp_bytes, &inv_perm, &w_bytes, n_rows, hidden, k, 4);
    let gpu_out: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let mut max_diff = 0.0f32;
    let mut max_at = 0usize;
    for (i, (g, r)) in gpu_out.iter().zip(ref_out.iter()).enumerate() {
        let d = (g - r).abs();
        if d > max_diff {
            max_diff = d;
            max_at = i;
        }
    }
    assert!(max_diff < 1e-4, "unpermute mismatch at [{max_at}]: max |diff| = {max_diff:.2e}",);
}

// ── extended coverage ────────────────────────────────────────────────────

fn f32_to_f16_bits(v: f32) -> u16 { half::f16::from_f32(v).to_bits() }
fn f16_bits_to_f32(b: u16) -> f32 { half::f16::from_bits(b).to_f32() }

#[test]
fn mt_moe_router_topk_qwen3_moe_shape_f32() {
    // Production cell: Qwen3-MoE = 128 experts, top-8, B*T per layer.
    // Use n_rows=16 to keep test fast while exercising the full
    // n_experts/32 = 4 entries-per-lane scan + k=8 mask iters.
    let n_rows = 16usize;
    let n_experts = 128usize;
    let k = 8usize;

    let logits: Vec<f32> = (0..n_rows * n_experts)
        .map(|i| ((i as f32 * 0.0173) % 13.0) - 6.5 + (i as f32 * 0.0001))
        .collect();
    let (ref_idx, ref_w) = cpu_topk_reference(&logits, n_rows, n_experts, k);

    let logits_bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let (gpu_idx, gpu_w_bytes) =
        run_topk(&ctx, DType::F32, &logits_bytes, n_rows, n_experts, k, 1, 4);
    let gpu_w: Vec<f32> =
        gpu_w_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    assert_eq!(gpu_idx, ref_idx, "Qwen3-MoE shape: indices diverge");
    for (i, (g, r)) in gpu_w.iter().zip(ref_w.iter()).enumerate() {
        assert!((g - r).abs() < 1e-5, "Qwen3-MoE weight[{i}] diverges: {g} vs {r}");
    }
}

#[test]
fn mt_moe_router_topk_f16() {
    let n_rows = 8usize;
    let n_experts = 64usize;
    let k = 4usize;

    let logits_f32: Vec<f32> = (0..n_rows * n_experts)
        .map(|i| ((i as f32 * 0.13) % 7.0) - 3.5 + (i as f32 * 0.001))
        .collect();
    // Round-trip through f16 so the oracle sees what the kernel sees.
    let logits_f16_to_f32: Vec<f32> =
        logits_f32.iter().map(|&v| f16_bits_to_f32(f32_to_f16_bits(v))).collect();
    let (ref_idx, ref_w) = cpu_topk_reference(&logits_f16_to_f32, n_rows, n_experts, k);

    let logits_bytes: Vec<u8> =
        logits_f32.iter().flat_map(|v| f32_to_f16_bits(*v).to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let (gpu_idx, gpu_w_bytes) =
        run_topk(&ctx, DType::F16, &logits_bytes, n_rows, n_experts, k, 1, 2);
    let gpu_w: Vec<f32> = gpu_w_bytes
        .chunks_exact(2)
        .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();

    assert_eq!(gpu_idx, ref_idx, "f16: indices diverge");
    // f16 softmax has ~10-bit mantissa precision — widen tolerance.
    for (i, (g, r)) in gpu_w.iter().zip(ref_w.iter()).enumerate() {
        let rel = (g - r).abs() / r.abs().max(1e-3);
        assert!(rel < 5e-3, "f16 weight[{i}] diverges: rel {rel:.2e}");
    }
}

#[test]
fn mt_moe_router_topk_k_equals_1() {
    // Degenerate case — top-k where k=1 reduces to argmax.
    let n_rows = 4usize;
    let n_experts = 32usize;
    let k = 1usize;
    let logits: Vec<f32> = (0..n_rows * n_experts)
        .map(|i| ((i as f32 * 0.7) % 5.0) - 2.5 + (i as f32 * 0.01))
        .collect();
    let (ref_idx, ref_w) = cpu_topk_reference(&logits, n_rows, n_experts, k);
    // Softmax over one element = 1.0
    for &w in &ref_w {
        assert!((w - 1.0).abs() < 1e-6, "ref softmax(1)={w}");
    }

    let logits_bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let (gpu_idx, gpu_w_bytes) =
        run_topk(&ctx, DType::F32, &logits_bytes, n_rows, n_experts, k, 1, 4);
    let gpu_w: Vec<f32> =
        gpu_w_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(gpu_idx, ref_idx);
    for &w in &gpu_w {
        assert!((w - 1.0).abs() < 1e-6, "k=1 weight should be 1.0, got {w}");
    }
}

#[test]
fn mt_moe_router_topk_tie_breaks_to_smaller_idx() {
    // Two experts with identical logit values — convention: lower
    // index wins. Our kernel uses simd_min on (idx | sentinel) which
    // matches that convention.
    let n_rows = 1usize;
    let n_experts = 8usize;
    let k = 2usize;
    let mut logits = vec![0.0f32; n_experts];
    // Logits descending except indices 2 and 5 tie at the top.
    logits[2] = 10.0;
    logits[5] = 10.0; // tie with [2]
    logits[3] = 8.0;
    // Top-2 should pick (2, 3) — smaller-idx wins the tie.
    let logits_bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let (gpu_idx, _) = run_topk(&ctx, DType::F32, &logits_bytes, n_rows, n_experts, k, 1, 4);
    assert_eq!(gpu_idx, vec![2u32, 5u32], "Tie-break: should pick 2 first then 5, not (5,2)");
}

#[test]
fn mt_moe_unpermute_shuffled_inv_perm_f32() {
    // Non-identity inv_perm — verifies the gather pattern with real
    // shuffled positions (the identity case in the base test would
    // hide a bug in the indexing math).
    let n_rows = 4usize;
    let hidden = 128usize;
    let k = 4usize;
    let total = n_rows * k;

    let expert_outputs: Vec<f32> =
        (0..total * hidden).map(|i| ((i as f32 * 0.11) % 9.0) - 4.5).collect();
    // Deterministic shuffled permutation.
    let inv_perm: Vec<u32> = (0..total as u32).map(|i| total as u32 - 1 - i).collect();
    let raw_weights: Vec<f32> = (0..n_rows * k).map(|i| 0.5 + (i as f32 * 0.07)).collect();
    let mut weights = vec![0.0f32; n_rows * k];
    for row in 0..n_rows {
        let s: f32 = raw_weights[row * k..(row + 1) * k].iter().sum();
        for j in 0..k {
            weights[row * k + j] = raw_weights[row * k + j] / s;
        }
    }
    let mut ref_out = vec![0.0f32; n_rows * hidden];
    for token in 0..n_rows {
        for h in 0..hidden {
            let mut acc = 0.0f32;
            for j in 0..k {
                let pos = inv_perm[token * k + j] as usize;
                acc += weights[token * k + j] * expert_outputs[pos * hidden + h];
            }
            ref_out[token * hidden + h] = acc;
        }
    }
    let exp_bytes: Vec<u8> = expert_outputs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let w_bytes: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let out_bytes =
        run_unpermute(&ctx, DType::F32, &exp_bytes, &inv_perm, &w_bytes, n_rows, hidden, k, 4);
    let gpu_out: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    for (i, (g, r)) in gpu_out.iter().zip(ref_out.iter()).enumerate() {
        assert!((g - r).abs() < 1e-4, "shuffled inv_perm[{i}]: {g} vs {r}");
    }
}

#[test]
fn mt_moe_unpermute_qwen3_moe_shape_f16() {
    // Production cell: Qwen3-MoE hidden=2048, k=8.
    let n_rows = 8usize;
    let hidden = 2048usize;
    let k = 8usize;
    let total = n_rows * k;

    let expert_outputs_f32: Vec<f32> =
        (0..total * hidden).map(|i| ((i as f32 * 0.0091) % 6.0) - 3.0).collect();
    let inv_perm: Vec<u32> = (0..total as u32).map(|i| (i * 7 + 3) % total as u32).collect();
    let raw_weights: Vec<f32> = (0..n_rows * k).map(|i| 0.4 + (i as f32 * 0.013)).collect();
    let mut weights_f32 = vec![0.0f32; n_rows * k];
    for row in 0..n_rows {
        let s: f32 = raw_weights[row * k..(row + 1) * k].iter().sum();
        for j in 0..k {
            weights_f32[row * k + j] = raw_weights[row * k + j] / s;
        }
    }
    // Round through f16 for the oracle.
    let ef16: Vec<f32> =
        expert_outputs_f32.iter().map(|&v| f16_bits_to_f32(f32_to_f16_bits(v))).collect();
    let wf16: Vec<f32> = weights_f32.iter().map(|&v| f16_bits_to_f32(f32_to_f16_bits(v))).collect();
    let mut ref_out = vec![0.0f32; n_rows * hidden];
    for token in 0..n_rows {
        for h in 0..hidden {
            let mut acc = 0.0f32;
            for j in 0..k {
                let pos = inv_perm[token * k + j] as usize;
                acc += wf16[token * k + j] * ef16[pos * hidden + h];
            }
            ref_out[token * hidden + h] = acc;
        }
    }

    let exp_bytes: Vec<u8> =
        expert_outputs_f32.iter().flat_map(|v| f32_to_f16_bits(*v).to_le_bytes()).collect();
    let w_bytes: Vec<u8> =
        weights_f32.iter().flat_map(|v| f32_to_f16_bits(*v).to_le_bytes()).collect();
    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let out_bytes =
        run_unpermute(&ctx, DType::F16, &exp_bytes, &inv_perm, &w_bytes, n_rows, hidden, k, 2);
    let gpu_out: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let mut max_rel = 0.0f32;
    for (g, r) in gpu_out.iter().zip(ref_out.iter()) {
        let rel = (g - r).abs() / r.abs().max(1e-3);
        if rel > max_rel {
            max_rel = rel;
        }
    }
    // f16 + 8-way weighted-sum: relative tolerance ~1% covers k cumulative ULP drifts.
    assert!(max_rel < 1e-2, "Qwen3-MoE f16 unpermute max rel diff {max_rel:.2e}");
}

#[test]
fn mt_moe_router_topk_qwen3_next_mode_f32() {
    // norm_topk_prob=0 (Qwen3-Next semantics):
    //   weight_i = exp(z_i) / Σ_j∈all exp(z_j)
    // Returned probs sum to < 1.0 since the renormalize step is skipped.
    let n_rows = 4usize;
    let n_experts = 32usize;
    let k = 4usize;
    let logits: Vec<f32> =
        (0..n_rows * n_experts).map(|i| ((i as f32 * 0.31) % 9.0) - 4.5).collect();

    // CPU reference for Qwen3-Next: full softmax then take chosen probs.
    let mut ref_idx = vec![0u32; n_rows * k];
    let mut ref_w = vec![0.0f32; n_rows * k];
    for row in 0..n_rows {
        let row_logits = &logits[row * n_experts..(row + 1) * n_experts];
        let max_v = row_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vec: Vec<f32> = row_logits.iter().map(|&v| (v - max_v).exp()).collect();
        let sum_all: f32 = exp_vec.iter().sum();
        let probs: Vec<f32> = exp_vec.iter().map(|&e| e / sum_all).collect();

        // Argpartition top-k.
        let mut order: Vec<usize> = (0..n_experts).collect();
        order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        for j in 0..k {
            ref_idx[row * k + j] = order[j] as u32;
            ref_w[row * k + j] = probs[order[j]];
        }
    }

    let logits_bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let (gpu_idx, gpu_w_bytes) =
        run_topk(&ctx, DType::F32, &logits_bytes, n_rows, n_experts, k, 0, 4);
    let gpu_w: Vec<f32> =
        gpu_w_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    assert_eq!(gpu_idx, ref_idx, "Qwen3-Next: indices diverge");
    for (i, (g, r)) in gpu_w.iter().zip(ref_w.iter()).enumerate() {
        assert!((g - r).abs() < 1e-5, "Qwen3-Next weight[{i}]: gpu={g:.6} ref={r:.6}");
    }

    // Sanity: per-row weight sum should be < 1.0 (not renormalized).
    for row in 0..n_rows {
        let s: f32 = gpu_w[row * k..(row + 1) * k].iter().sum();
        assert!(s < 1.0, "Qwen3-Next mode: row {row} sum is {s:.6}, should be < 1.0");
    }
}

#[test]
fn mt_moe_router_topk_bf16() {
    let n_rows = 4usize;
    let n_experts = 32usize;
    let k = 4usize;
    let logits_f32: Vec<f32> =
        (0..n_rows * n_experts).map(|i| ((i as f32 * 0.21) % 5.0) - 2.5).collect();
    // Round through bf16 (top 16 bits of fp32 representation).
    let to_bf16_bits = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };
    let from_bf16_bits = |b: u16| -> f32 { f32::from_bits((b as u32) << 16) };
    let logits_round: Vec<f32> =
        logits_f32.iter().map(|&v| from_bf16_bits(to_bf16_bits(v))).collect();
    let (ref_idx, ref_w) = cpu_topk_reference(&logits_round, n_rows, n_experts, k);
    let logits_bytes: Vec<u8> =
        logits_f32.iter().flat_map(|v| to_bf16_bits(*v).to_le_bytes()).collect();

    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let (gpu_idx, gpu_w_bytes) =
        run_topk(&ctx, DType::BF16, &logits_bytes, n_rows, n_experts, k, 1, 2);
    let gpu_w: Vec<f32> = gpu_w_bytes
        .chunks_exact(2)
        .map(|c| from_bf16_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();

    assert_eq!(gpu_idx, ref_idx, "bf16: indices diverge");
    // bf16 has ~7-bit mantissa — wider tolerance.
    for (i, (g, r)) in gpu_w.iter().zip(ref_w.iter()).enumerate() {
        let rel = (g - r).abs() / r.abs().max(1e-3);
        assert!(rel < 2e-2, "bf16 weight[{i}]: rel {rel:.2e}");
    }
}

#[test]
fn mt_moe_router_topk_nan_inf_clamp_f32() {
    // Regression: NaN/Inf in router_logits can produce duplicate expert
    // IDs because `NaN > x` is always false → the chosen-mask check
    // doesn't fire and the same expert can be selected twice. Mirrors
    // vLLM's test_fused_topk_nan_inf_clamp (test_fused_topk.py:140-204).
    //
    // Mitigation in our kernel: the per-lane local argmax uses strict
    // `>` so NaN propagates as "never wins" — the lane's best_val stays
    // at its initial neg_infinity OR the most recent non-NaN candidate.
    // This test pins that behavior.
    let n_rows = 4usize;
    let n_experts = 16usize;
    let k = 4usize;

    // Row 0: one NaN at expert 5
    // Row 1: +Inf at expert 7 (should be picked first)
    // Row 2: -Inf at expert 3
    // Row 3: mix of NaN + +Inf
    let mut logits = vec![0.0f32; n_rows * n_experts];
    for j in 0..n_experts {
        logits[j] = j as f32 * 0.1;
        logits[n_experts + j] = j as f32 * 0.1;
        logits[2 * n_experts + j] = j as f32 * 0.1;
        logits[3 * n_experts + j] = j as f32 * 0.1;
    }
    logits[5] = f32::NAN;
    logits[n_experts + 7] = f32::INFINITY;
    logits[2 * n_experts + 3] = f32::NEG_INFINITY;
    logits[3 * n_experts + 2] = f32::NAN;
    logits[3 * n_experts + 11] = f32::INFINITY;

    let logits_bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let (gpu_idx, _) = run_topk(&ctx, DType::F32, &logits_bytes, n_rows, n_experts, k, 1, 4);

    // INVARIANT 1: no row has a duplicate expert ID in its top-k slice.
    for row in 0..n_rows {
        let slice = &gpu_idx[row * k..(row + 1) * k];
        let mut sorted = slice.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), k, "row {row}: duplicate expert in top-k slice {slice:?}",);
    }

    // INVARIANT 2: row 1 picks +Inf at index 7 as the top entry.
    assert_eq!(gpu_idx[k], 7, "row 1: +Inf should be top-1, got {:?}", &gpu_idx[k..2 * k]);

    // INVARIANT 3: row 2 does NOT pick -Inf at index 3 (it's worse than 0.0).
    let row2_slice = &gpu_idx[2 * k..3 * k];
    assert!(
        !row2_slice.contains(&3u32),
        "row 2: -Inf at idx 3 should not be chosen, got {row2_slice:?}"
    );

    // INVARIANT 4: row 3 picks +Inf at index 11 as top-1.
    assert_eq!(gpu_idx[3 * k], 11, "row 3: +Inf at 11 should be top-1");

    // INVARIANT 5: NaN expert (index 5 in row 0; index 2 in row 3) is NOT
    // in the top-k slice — NaN should never compare as "better than"
    // any finite candidate.
    let row0_slice = &gpu_idx[0..k];
    assert!(
        !row0_slice.contains(&5u32),
        "row 0: NaN at idx 5 should not be chosen, got {row0_slice:?}"
    );
    let row3_slice = &gpu_idx[3 * k..4 * k];
    assert!(
        !row3_slice.contains(&2u32),
        "row 3: NaN at idx 2 should not be chosen, got {row3_slice:?}"
    );
}

#[allow(clippy::too_many_arguments)]
fn run_permute(
    ctx: &Context,
    dtype: DType,
    tokens_bytes: &[u8],
    sort_token_idx: &[u32],
    n_rows: usize,
    n_permuted: usize,
    hidden: usize,
    elem_bytes: usize,
) -> Vec<u8> {
    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("tokens".into(), tokens_bytes.to_vec());
    buffers.insert(
        "sort_token_idx".into(),
        sort_token_idx.iter().flat_map(|v| v.to_le_bytes()).collect(),
    );
    buffers.insert("permuted".into(), vec![0u8; n_permuted * hidden * elem_bytes]);
    buffers.insert("hidden".into(), (hidden as u32).to_le_bytes().to_vec());
    let _ = n_rows; // n_rows not passed to kernel — derived from input shape via sort idx

    let mut kernel = mt_moe_permute::kernel_ir_for(dtype);
    kernel.mode = metaltile_core::ir::KernelMode::Reduction;
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [n_permuted, 1, 1], [128, 1, 1])
        .expect("dispatch_with_grid should succeed");
    result.outputs.get("permuted").expect("permuted").clone()
}

#[test]
fn mt_moe_permute_gather_matches_cpu_reference_f32() {
    // Forward MoE permute: gather tokens by per-position source-token index.
    // Caller (host) computes sort_token_idx via argsort over flat
    // (token, slot) pairs by expert; this kernel does just the gather.
    let n_rows = 4usize;
    let hidden = 128usize;
    let k = 4usize;
    let n_permuted = n_rows * k;

    let tokens: Vec<f32> = (0..n_rows * hidden).map(|i| ((i as f32 * 0.11) % 9.0) - 4.5).collect();
    // Sort index: each token contributes k times to permuted, but
    // shuffled. Use a deterministic permutation that doesn't equal
    // identity so we can't accidentally pass via "kernel copies row i to
    // row i" bug.
    let sort_token_idx: Vec<u32> =
        (0..n_permuted as u32).map(|p| (p * 7 + 3) % n_rows as u32).collect();

    // CPU reference.
    let mut ref_out = vec![0.0f32; n_permuted * hidden];
    for p in 0..n_permuted {
        let src = sort_token_idx[p] as usize;
        ref_out[p * hidden..(p + 1) * hidden]
            .copy_from_slice(&tokens[src * hidden..(src + 1) * hidden]);
    }

    let tokens_bytes: Vec<u8> = tokens.iter().flat_map(|v| v.to_le_bytes()).collect();
    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let out_bytes = run_permute(
        &ctx,
        DType::F32,
        &tokens_bytes,
        &sort_token_idx,
        n_rows,
        n_permuted,
        hidden,
        4,
    );
    let gpu_out: Vec<f32> =
        out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    // Permute is an exact copy — should match byte-for-byte at f32.
    for (i, (g, r)) in gpu_out.iter().zip(ref_out.iter()).enumerate() {
        assert_eq!(g, r, "permute[{i}]: {g} != {r}");
    }
}

#[test]
fn mt_moe_permute_qwen3_moe_shape_f16() {
    // Production-shape pin: Qwen3-MoE hidden=2048, B*T=8 tokens, k=8
    // active experts per token → 64 permuted positions.
    let n_rows = 8usize;
    let hidden = 2048usize;
    let k = 8usize;
    let n_permuted = n_rows * k;

    let tokens_f32: Vec<f32> =
        (0..n_rows * hidden).map(|i| ((i as f32 * 0.013) % 5.0) - 2.5).collect();
    let sort_token_idx: Vec<u32> =
        (0..n_permuted as u32).map(|p| (p * 13 + 5) % n_rows as u32).collect();

    let mut ref_out_f32 = vec![0.0f32; n_permuted * hidden];
    for p in 0..n_permuted {
        let src = sort_token_idx[p] as usize;
        for h in 0..hidden {
            // Round through f16 since the kernel stores f16 values.
            ref_out_f32[p * hidden + h] =
                half::f16::from_f32(tokens_f32[src * hidden + h]).to_f32();
        }
    }

    let tokens_bytes: Vec<u8> =
        tokens_f32.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect();
    let _g = gpu_lock();
    let ctx = Context::new().unwrap();
    let out_bytes = run_permute(
        &ctx,
        DType::F16,
        &tokens_bytes,
        &sort_token_idx,
        n_rows,
        n_permuted,
        hidden,
        2,
    );
    let gpu_out: Vec<f32> = out_bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    for (i, (g, r)) in gpu_out.iter().zip(ref_out_f32.iter()).enumerate() {
        // f16 round-trip — exact match expected (kernel doesn't compute).
        assert_eq!(g, r, "permute[{i}]: {g} != {r}");
    }
}
