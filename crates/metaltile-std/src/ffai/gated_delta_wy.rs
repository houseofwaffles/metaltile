//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Chunked-WY Gated DeltaNet prefill kernel — `mt_gated_delta_wy_chunk`.
//!
//! Spec 028 Phase 2 (naive scalar Metal port). Process the full prefill
//! T-sequence chunk-by-chunk via the compact Woodbury-Young representation
//! of the delta-rule product. Sequential dependency across chunks; parallel
//! within each chunk. This file is the scalar foundation — MMA tiling is
//! a follow-up (Phase 3).
//!
//! Validated against the CPU oracle in
//! `tests/gated_delta_wy_cpu_oracle.rs` and the Python reference in
//! `/tmp/gdn_chunked_wy/gdn_wy_ref.py`.
//!
//! ## Algorithm (per chunk of size C, single (B, Hv) slot)
//!
//! 1. Gather Q,K,V,g,β for the chunk into TG memory.
//! 2. Prefix gates G_t = Π g_i; ratios Γ[t,j] = G_t/G_j.
//! 3. KKT[i,j] = k_i · k_j.
//! 4. Solve (I + L)·p = K       where L[t,j] = β_j·KKT[t,j], j<t.
//! 5. Solve (I + A)·u^v = β⊙V   where A[t,j] = β_t·Γ[t,j]·KKT[t,j], j<t.
//! 6. y_local[t]  = Σ_{j≤t} Γ[t,j]·QKT[t,j]·u^v[j]
//! 7. y_pass[t]   = G_t · (S_0·q_t − Σ_{i≤t} β_i·QKT[t,i]·(S_0·p_i))
//! 8. y_out[t]    = y_pass + y_local
//! 9. S_end       = G_C·(S_0 − S_0·(β⊙p)^T·K) + Σ_j (G_C/G_j)·u^v_j⊗k_j
//!
//! State at chunk N+1 is S_end of chunk N. The TG loops chunks.
//!
//! ## Dispatch
//!
//!   - **Mode**: Reduction (uses simdgroup + threadgroup ops)
//!   - **Grid**: `[1, B*Hv, 1]`
//!   - **TG**:   `[32, 1, 1]` (one simdgroup; minimum valid TPG)
//!
//! Sequential dependency across chunks means we cannot parallelize on T
//! within a single (B,Hv) slot. We parallelize across (B,Hv) only — every
//! GDN layer's `B*Hv` is large enough to saturate the GPU (Qwen3.6-35B-A3B
//! has Hv=4 per layer × B; typical inference Hv*B ≥ 32 saturates M5 Max's
//! ~480 simdgroup slots).
//!
//! ## Layouts (match `mt_gated_delta_chunk`)
//!
//!   - `q, k`:    [B, T, Hk, Dk]
//!   - `v, y`:    [B, T, Hv, Dv]
//!   - `g, beta`: [B, T, Hv]
//!   - `state_in/out`: [B, Hv, Dv, Dk]
//!
//! GQA: `hk_idx = hv_idx / (Hv / Hk)`.
//!
//! ## Constexpr params
//!
//!   - `dk`, `dv` — must be multiples of 8 (future MMA path) and 32 (lane work)
//!   - `hv`, `hk` — head counts, runtime-known for indexing
//!   - `c` — chunk size, must be ≤ 64 (TG memory budget) and multiple of 8
//!   - `t_len` — total prefill length, used to bound the chunk loop
//!
//! All `c×c`, `c×dk`, `c×dv` intermediates live in TG memory. State
//! [Dv, Dk] lives in TG between chunks (no global write-back per chunk).
//!
//! ## Numerical precision
//!
//! Matches `mt_gated_delta_chunk`: accumulators in f32, state in f32 too.
//! Triangular solves run in f32; the matmuls inside `(I+L)` and `(I+A)`
//! grow the condition number with T, so f32 is the floor for stable
//! recurrences at long context.

#![allow(clippy::too_many_arguments)]

use metaltile::kernel;

#[kernel(
    bench(
        op="gated_delta",
        subop="wy_chunk",
        class=GenericEmpty,
        tol=0.0,
        kernel_mode=Reduction,
    )
)]
pub fn mt_gated_delta_wy_chunk<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    g: Tensor<T>,
    beta: Tensor<T>,
    state_in: Tensor<T>,
    mut state_out: Tensor<T>,
    mut y: Tensor<T>,
    #[constexpr] dk: u32,
    #[constexpr] dv: u32,
    #[constexpr] hv: u32,
    #[constexpr] hk: u32,
    #[constexpr] c: u32,
    #[constexpr] t_len: u32,
) {
    // ── Geometry ───────────────────────────────────────────────────────
    let n = tgid_y; // batch*hv slot
    let b_idx = n / hv;
    let hv_idx = n % hv;
    let hv_per_hk = hv / hk;
    let hk_idx = hv_idx / hv_per_hk;
    let lane = simd_lane;
    // ── TG buffers ─────────────────────────────────────────────────────
    //
    // Scalar correctness path — supports up to Dk=Dv=32, C=16.
    // Apple TG memory cap is ~32 KB per kernel; full Qwen3.6 dims (Dk=Dv=128
    // C=64) will need streaming or simdgroup-matrix tiling.
    //
    // Sizes (4 bytes each): state 1024 + q/k 512+512 + v 512 + kkt 256
    // + bigG/g/beta 16+16+16 + p 512 + uv 512 + qkt 256 = 4144 floats = 17 KB.
    threadgroup_alloc("tg_state", 1024u32, f32); // up to 32*32
    // Per-lane stack staging for the chunk-end state update — replaces the
    // tg_state_new TG buffer (saved 4 KB). Each lane handles (dv*dk/32)
    // iterations; stash new values here, barrier once, then write back.
    // 128 supports Dv*Dk ≤ 4096 (e.g. 64×64).
    stack_alloc("new_state", 128u32, "f32");
    threadgroup_alloc("tg_q", 512u32, f32); // C × Dk
    threadgroup_alloc("tg_k", 512u32, f32);
    threadgroup_alloc("tg_v", 512u32, f32); // C × Dv
    threadgroup_alloc("tg_g", 16u32, f32);
    threadgroup_alloc("tg_beta", 16u32, f32);
    threadgroup_alloc("tg_bigG", 16u32, f32);
    threadgroup_alloc("tg_kkt", 256u32, f32); // [c, c]
    threadgroup_alloc("tg_p", 512u32, f32);
    threadgroup_alloc("tg_uv", 512u32, f32);
    threadgroup_alloc("tg_qkt", 256u32, f32);
    // S0_p[d_v, i] = Σ_d state[d_v, d] · p[i, d]  ∈ R^{Dv × C}
    // Precomputed once per chunk; both y_pass and S_end reuse it.
    // Eliminates ~256K redundant TG state reads per chunk at Dv=32 C=16.
    threadgroup_alloc("tg_s0p", 512u32, f32); // Dv × C, max 32 × 16 = 512
    // ── State init: load [Dv, Dk] from state_in[n] ────────────────────
    let state_base = n * dv * dk;
    let total_state = dv * dk;
    for ii in range(lane, total_state, 32u32) {
        let v_in = load(state_in[state_base + ii]).cast::<f32>();
        threadgroup_store("tg_state", ii, v_in);
    }
    threadgroup_barrier();
    // ── Chunk loop ────────────────────────────────────────────────────
    //
    // Precondition: t_len % c == 0. Caller must pad shorter prefills up to
    // a multiple of `c` with zero-init tokens (g=1, β=0 → no-op recurrence).
    // This keeps the kernel body free of branching on c at chunk
    // boundaries — significant codegen win at long context.
    let num_chunks = t_len / c;
    for chunk_idx in range(0u32, num_chunks, 1u32) {
        let chunk_start = chunk_idx * c;
        // Step 1: gather Q, K, V, g, β for this chunk into TG.
        for i in range(0u32, c, 1u32) {
            let t_abs = chunk_start + i;
            for d in range(lane, dk, 32u32) {
                let qkv_off = (t_abs * hk + hk_idx) * dk + d;
                threadgroup_store("tg_q", i * dk + d, load(q[qkv_off]).cast::<f32>());
                threadgroup_store("tg_k", i * dk + d, load(k[qkv_off]).cast::<f32>());
            }
            for d in range(lane, dv, 32u32) {
                let v_off = (t_abs * hv + hv_idx) * dv + d;
                threadgroup_store("tg_v", i * dv + d, load(v[v_off]).cast::<f32>());
            }
            if lane == 0u32 {
                let gb_off = t_abs * hv + hv_idx;
                threadgroup_store("tg_g", i, load(g[gb_off]).cast::<f32>());
                threadgroup_store("tg_beta", i, load(beta[gb_off]).cast::<f32>());
            }
        }
        threadgroup_barrier();
        // Step 2: prefix gates G_t (one lane, scalar — small C).
        if lane == 0u32 {
            let mut g_acc = 1.0f32;
            for i in range(0u32, c, 1u32) {
                g_acc = g_acc * threadgroup_load("tg_g", i);
                threadgroup_store("tg_bigG", i, g_acc);
            }
        }
        threadgroup_barrier();
        // Step 3: KKT[i, j] = k_i · k_j  (lane-parallel over (i, j) pairs).
        for ij in range(lane, c * c, 32u32) {
            let i = ij / c;
            let j = ij % c;
            let mut s = 0.0f32;
            for d in range(0u32, dk, 1u32) {
                let ki = threadgroup_load("tg_k", i * dk + d);
                let kj = threadgroup_load("tg_k", j * dk + d);
                s = s + ki * kj;
            }
            threadgroup_store("tg_kkt", i * c + j, s);
        }
        threadgroup_barrier();
        // Step 4: solve (I + L) p = K via forward substitution.
        //   L[t, j] = β_j · KKT[t, j] for j < t; else 0.
        //   p[0] = K[0]
        //   p[t] = K[t] - Σ_{j<t} L[t,j] * p[j]
        // Lane-parallelism over Dk for each iteration.
        // Forward-sub iteration: outer loop over t, inner work over Dk lane-parallel.
        for t in range(0u32, c, 1u32) {
            // Compute p[t, d] for all d in parallel.
            for d in range(lane, dk, 32u32) {
                let mut accum = threadgroup_load("tg_k", t * dk + d);
                // Subtract sum_{j<t} L[t, j] * p[j, d]
                for j in range(0u32, t, 1u32) {
                    let beta_j = threadgroup_load("tg_beta", j);
                    let kkt_tj = threadgroup_load("tg_kkt", t * c + j);
                    let p_jd = threadgroup_load("tg_p", j * dk + d);
                    accum = accum - beta_j * kkt_tj * p_jd;
                }
                threadgroup_store("tg_p", t * dk + d, accum);
            }
            threadgroup_barrier();
        }
        // Step 5: solve (I + A) u^v = β ⊙ V.
        //   A[t, j] = β_t · Γ[t,j] · KKT[t, j]  for j < t
        //   u^v[0]  = β_0 · v_0
        //   u^v[t]  = β_t · v_t  -  Σ_{j<t} A[t,j] · u^v[j]
        for t in range(0u32, c, 1u32) {
            let beta_t = threadgroup_load("tg_beta", t);
            let big_g_t = threadgroup_load("tg_bigG", t);
            for d in range(lane, dv, 32u32) {
                let v_td = threadgroup_load("tg_v", t * dv + d);
                let mut accum = beta_t * v_td;
                for j in range(0u32, t, 1u32) {
                    let big_g_j = threadgroup_load("tg_bigG", j);
                    let gamma_tj = big_g_t / big_g_j;
                    let kkt_tj = threadgroup_load("tg_kkt", t * c + j);
                    let a_tj = beta_t * gamma_tj * kkt_tj;
                    let uv_jd = threadgroup_load("tg_uv", j * dv + d);
                    accum = accum - a_tj * uv_jd;
                }
                threadgroup_store("tg_uv", t * dv + d, accum);
            }
            threadgroup_barrier();
        }
        // Step 6 prep: QKT[t, j] = Σ_d q[t,d] · k[j,d]
        for tj in range(lane, c * c, 32u32) {
            let t = tj / c;
            let j = tj % c;
            let mut s = 0.0f32;
            for d in range(0u32, dk, 1u32) {
                let qt = threadgroup_load("tg_q", t * dk + d);
                let kj = threadgroup_load("tg_k", j * dk + d);
                s = s + qt * kj;
            }
            threadgroup_store("tg_qkt", t * c + j, s);
        }
        threadgroup_barrier();
        // Precompute S0_p[d_v, i] = Σ_d state[d_v, d] · p[i, d] (∈ R^{Dv × C}).
        // Reused by both the y_pass correction term AND the chunk-end state
        // update. Lane-parallel over (d_v, i) pairs.
        for vi in range(lane, dv * c, 32u32) {
            let d_v = vi / c;
            let i = vi % c;
            let mut acc = 0.0f32;
            for d in range(0u32, dk, 1u32) {
                let st = threadgroup_load("tg_state", d_v * dk + d);
                let pi = threadgroup_load("tg_p", i * dk + d);
                acc = acc + st * pi;
            }
            threadgroup_store("tg_s0p", d_v * c + i, acc);
        }
        threadgroup_barrier();
        // Steps 6–8: per (t, d_v) compute y[t, d_v] = y_pass + y_local.
        //   y_local[t, dv]  = Σ_{j≤t} Γ[t,j] · QKT[t,j] · u^v[j, dv]
        //   S0_q[t, dv]     = Σ_d  state[dv, d] · q[t, d]
        //   y_pass_corr     = Σ_{i≤t} β_i · QKT[t, i] · S0_p[dv, i]
        //   y[t, dv]        = big_g[t] · (S0_q - y_pass_corr) + y_local
        for tdv in range(lane, c * dv, 32u32) {
            let t = tdv / dv;
            let d_v = tdv % dv;
            let big_g_t = threadgroup_load("tg_bigG", t);
            // y_local
            let mut y_loc = 0.0f32;
            for j in range(0u32, t + 1u32, 1u32) {
                let big_g_j = threadgroup_load("tg_bigG", j);
                let gamma_tj = big_g_t / big_g_j;
                let qkt_tj = threadgroup_load("tg_qkt", t * c + j);
                let uv_jd = threadgroup_load("tg_uv", j * dv + d_v);
                y_loc = y_loc + gamma_tj * qkt_tj * uv_jd;
            }
            // S0_q[t, dv] = Σ_d state[dv, d] · q[t, d]
            let mut s0q = 0.0f32;
            for d in range(0u32, dk, 1u32) {
                let st = threadgroup_load("tg_state", d_v * dk + d);
                let qt = threadgroup_load("tg_q", t * dk + d);
                s0q = s0q + st * qt;
            }
            // correction = Σ_{i≤t} β_i · QKT[t,i] · S0_p[d_v, i]
            let mut corr = 0.0f32;
            for i in range(0u32, t + 1u32, 1u32) {
                let beta_i = threadgroup_load("tg_beta", i);
                let qkt_ti = threadgroup_load("tg_qkt", t * c + i);
                let s0p_vi = threadgroup_load("tg_s0p", d_v * c + i);
                corr = corr + beta_i * qkt_ti * s0p_vi;
            }
            let y_pass = big_g_t * (s0q - corr);
            let t_abs = chunk_start + t;
            let y_off = (t_abs * hv + hv_idx) * dv + d_v;
            store(y[y_off], (y_pass + y_loc).cast::<T>());
        }
        threadgroup_barrier();
        // Step 9: end-of-chunk state update.
        //   S_through[v, d] = G_C · (S_0[v, d] - Σ_i β_i · p[i, d] · (S_0[v, *] · p[i, *]^T))
        //   U_end[v, d]     = Σ_j (G_C/G_j) · u^v[j, v] · k[j, d]
        //   S_end[v, d]     = S_through + U_end
        let big_g_c = threadgroup_load("tg_bigG", c - 1u32);
        // Per-lane iteration counter for stack staging (0..(dv*dk/32)).
        let mut iter_idx = 0u32;
        for vd in range(lane, dv * dk, 32u32) {
            let d_v = vd / dk;
            let d_k = vd % dk;
            let s0_old = threadgroup_load("tg_state", d_v * dk + d_k);
            // S0_bp_t_K [d_v, d_k] = Σ_i β_i · p[i, d_k] · S0_p[d_v, i]
            // S0_p was precomputed before y_pass — reuse it here.
            let mut s_corr = 0.0f32;
            for i in range(0u32, c, 1u32) {
                let beta_i = threadgroup_load("tg_beta", i);
                let p_ik = threadgroup_load("tg_p", i * dk + d_k);
                let s0p_vi = threadgroup_load("tg_s0p", d_v * c + i);
                s_corr = s_corr + beta_i * p_ik * s0p_vi;
            }
            let s_through = big_g_c * (s0_old - s_corr);
            // U_end[d_v, d_k] = Σ_j (G_C/G_j) · u^v[j, d_v] · k[j, d_k]
            let mut u_end = 0.0f32;
            for j in range(0u32, c, 1u32) {
                let big_g_j = threadgroup_load("tg_bigG", j);
                let rw = big_g_c / big_g_j;
                let uv_jv = threadgroup_load("tg_uv", j * dv + d_v);
                let k_jd = threadgroup_load("tg_k", j * dk + d_k);
                u_end = u_end + rw * uv_jv * k_jd;
            }
            // Stash in per-lane stack; flush to tg_state after a barrier.
            stack_store("new_state", iter_idx, s_through + u_end);
            iter_idx = iter_idx + 1u32;
        }
        threadgroup_barrier();
        // Flush staged values back into tg_state for the next chunk's reads.
        let mut flush_idx = 0u32;
        for vd in range(lane, dv * dk, 32u32) {
            threadgroup_store("tg_state", vd, stack_load("new_state", flush_idx));
            flush_idx = flush_idx + 1u32;
        }
        threadgroup_barrier();
    }
    // ── Write final state out ──────────────────────────────────────────
    for ii in range(lane, total_state, 32u32) {
        let s = threadgroup_load("tg_state", ii);
        store(state_out[state_base + ii], s.cast::<T>());
    }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::mt_gated_delta_wy_chunk;

    // Single-chunk shape (T = C) that fits the scalar TG-memory budget:
    // Dk=32, Dv=16, C=16, Hk=Hv=1. `t_len` must be a multiple of `c`.
    #[bench(name = "ffai/gated_delta_wy_chunk", dtypes = [f32, f16, bf16])]
    fn bench_gated_delta_wy_chunk(dt: DType) -> BenchSetup {
        let (b, t, hk, hv, dk, dv, c) =
            (1usize, 16usize, 1usize, 1usize, 32usize, 16usize, 16usize);
        let n_total = b * hv;
        BenchSetup::new(mt_gated_delta_wy_chunk::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("q", t * hk * dk, dt))
            .buffer(BenchBuffer::random("k", t * hk * dk, dt))
            .buffer(BenchBuffer::random("v", t * n_total * dv, dt))
            .buffer(BenchBuffer::random("g", t * n_total, dt))
            .buffer(BenchBuffer::random("beta", t * n_total, dt))
            .buffer(BenchBuffer::random("state_in", n_total * dv * dk, dt))
            .buffer(BenchBuffer::zeros("state_out", n_total * dv * dk, dt).output())
            .buffer(BenchBuffer::zeros("y", t * n_total * dv, dt).output())
            .constexpr("dk", dk as u32)
            .constexpr("dv", dv as u32)
            .constexpr("hv", hv as u32)
            .constexpr("hk", hk as u32)
            .constexpr("c", c as u32)
            .constexpr("t_len", t as u32)
            .grid_3d(1, n_total as u32, 1, [32, 1, 1])
            .bytes_moved((t * n_total * dv * dt.size_bytes()) as u64)
    }
}
