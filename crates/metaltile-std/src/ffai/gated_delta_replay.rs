//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! GatedDeltaNet innovation-tape capture + replay — port of
//! `gated_delta_replay.metal` (spec 020 phase 2). Companion to
//! `gated_delta.rs`; the speculative-decode rollback path for
//! GDN-bearing models (Qwen 3.5 / 3.6).
//!
//! Two kernels:
//!   - `gated_delta_step_record` — the standard GatedDelta forward step
//!     that *also* writes each step's `delta_t` to a `delta_log` tape.
//!   - `state_replay` — re-folds the accepted prefix `[0, accepted)` of
//!     an innovation tape onto a pre-record state snapshot:
//!     `state ← select(do_step, state·g_t + k_t·delta_t, state)`,
//!     branchless via `select` (good SIMD occupancy when the timestep
//!     mask is non-uniform within a simdgroup).
//!
//! Tape layout: `delta_log` [B, T, Hv, Dv], `k_log` [B, T, Hv, Dk]
//! (GQA-expanded by the cache), `g_log` [B, T, Hv].
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid3D**, `grid = [1, Dv, batch*Hv]`, `tg = [32, 1, 1]`.
//! - `Dk` a multiple of 32.
//!
//! Codegen-only; correctness pinned by
//! `tests/gated_delta_replay_gpu_correctness.rs`.

use metaltile::kernel;

// ── Forward GatedDelta step with per-step delta-tape capture ────────────────
#[rustfmt::skip]
macro_rules! gated_delta_record {
    ($name:ident, $dk:literal, $dv:literal, $hk:literal, $hv:literal, $n_per_t:literal, $subop:literal) => {
        #[kernel(
            bench(
                op="gated_delta_replay",
                subop=$subop,
                class=GenericEmpty,
                tol=1e-3,
                kernel_mode=Grid3D,
            )
        )]
        pub fn $name<T>(
            q: Tensor<T>,
            k: Tensor<T>,
            v: Tensor<T>,
            g: Tensor<T>,
            beta: Tensor<T>,
            state_in: Tensor<T>,
            mask: Tensor<u32>,
            mut y: Tensor<T>,
            mut state_out: Tensor<T>,
            mut delta_log: Tensor<T>,
            #[constexpr] t_val: u32,
            #[constexpr] has_mask: u32,
        ) {
            let lane = program_id::<0>();
            let dv_idx = program_id::<1>();
            let n = program_id::<2>();
            let b_idx = n / $hv;
            let hv_idx = n - b_idx * $hv;
            let hk_idx = hv_idx / ($hv / $hk);
            let i_state_base = (n * $dv + dv_idx) * $dk;

            stack_alloc("state", $n_per_t, "f32");
            for i in range(0u32, $n_per_t, 1u32) {
                let v = load(state_in[i_state_base + $n_per_t * lane + i]).cast::<f32>();
                stack_store("state", i, v);
            }

            for t in range(0u32, t_val, 1u32) {
                let m = select(has_mask == 0u32, 1u32, load(mask[b_idx * t_val + t]));
                if m > 0u32 {
                    let qk_base = (b_idx * t_val + t) * $hk * $dk + hk_idx * $dk;
                    let v_base = (b_idx * t_val + t) * $hv * $dv + hv_idx * $dv;
                    let gb_idx = (b_idx * t_val + t) * $hv + hv_idx;
                    let g_val = load(g[gb_idx]).cast::<f32>();
                    let beta_val = load(beta[gb_idx]).cast::<f32>();

                    let mut kv_mem = 0.0f32;
                    for i in range(0u32, $n_per_t, 1u32) {
                        let s_idx = $n_per_t * lane + i;
                        let st = stack_load("state", i) * g_val;
                        stack_store("state", i, st);
                        kv_mem = kv_mem + st * load(k[qk_base + s_idx]).cast::<f32>();
                    }
                    let kv = simd_sum(kv_mem);
                    let delta = (load(v[v_base + dv_idx]).cast::<f32>() - kv) * beta_val;

                    // Tape write: surface delta_t for the replay kernel.
                    if lane == 0u32 {
                        store(delta_log[v_base + dv_idx], delta.cast::<T>());
                    }

                    let mut out_acc = 0.0f32;
                    for i in range(0u32, $n_per_t, 1u32) {
                        let s_idx = $n_per_t * lane + i;
                        let st =
                            stack_load("state", i) + load(k[qk_base + s_idx]).cast::<f32>() * delta;
                        stack_store("state", i, st);
                        out_acc = out_acc + st * load(q[qk_base + s_idx]).cast::<f32>();
                    }
                    let out_red = simd_sum(out_acc);
                    if lane == 0u32 {
                        store(y[v_base + dv_idx], out_red.cast::<T>());
                    }
                }
            }

            for i in range(0u32, $n_per_t, 1u32) {
                let st = stack_load("state", i);
                store(state_out[i_state_base + $n_per_t * lane + i], st.cast::<T>());
            }
        }
    };
}

// ── Tape replay: re-fold the accepted prefix onto a snapshot ────────────────
#[rustfmt::skip]
macro_rules! state_replay {
    ($name:ident, $dk:literal, $dv:literal, $hv:literal, $n_per_t:literal, $subop:literal) => {
        #[kernel(
            bench(
                op="gated_delta_replay",
                subop=$subop,
                class=GenericEmpty,
                tol=1e-3,
                kernel_mode=Grid3D,
            )
        )]
        pub fn $name<T>(
            delta_log: Tensor<T>,
            k_log: Tensor<T>,
            g_log: Tensor<T>,
            state_in: Tensor<T>,
            mask: Tensor<u32>,
            mut state_out: Tensor<T>,
            #[constexpr] t_log: u32,
            #[constexpr] accepted: u32,
            #[constexpr] has_mask: u32,
        ) {
            let lane = program_id::<0>();
            let dv_idx = program_id::<1>();
            let n = program_id::<2>();
            let b_idx = n / $hv;
            let hv_idx = n - b_idx * $hv;
            let i_state_base = (n * $dv + dv_idx) * $dk;

            stack_alloc("state", $n_per_t, "f32");
            for i in range(0u32, $n_per_t, 1u32) {
                let v = load(state_in[i_state_base + $n_per_t * lane + i]).cast::<f32>();
                stack_store("state", i, v);
            }

            for t in range(0u32, t_log, 1u32) {
                let mask_v = select(has_mask == 0u32, 1u32, load(mask[b_idx * t_log + t]));
                // do_step = (t < accepted) && mask_passes — branchless.
                let do_step = select(t < accepted, mask_v, 0u32);

                let delta_row = (b_idx * t_log + t) * $hv * $dv + hv_idx * $dv;
                let k_row = (b_idx * t_log + t) * $hv * $dk + hv_idx * $dk;
                let g_idx = (b_idx * t_log + t) * $hv + hv_idx;
                let g_val = load(g_log[g_idx]).cast::<f32>();
                let d_val = load(delta_log[delta_row + dv_idx]).cast::<f32>();

                for i in range(0u32, $n_per_t, 1u32) {
                    let s_idx = $n_per_t * lane + i;
                    let old = stack_load("state", i);
                    let new_val = old * g_val + load(k_log[k_row + s_idx]).cast::<f32>() * d_val;
                    stack_store("state", i, select(do_step > 0u32, new_val, old));
                }
            }

            for i in range(0u32, $n_per_t, 1u32) {
                let st = stack_load("state", i);
                store(state_out[i_state_base + $n_per_t * lane + i], st.cast::<T>());
            }
        }
    };
}

// Qwen 3.5/3.6 A3B: Dk=192, Dv=128, Hk=4, Hv=4.
gated_delta_record!(
    gated_delta_step_record_d192_128_4_4,
    192u32,
    128u32,
    4u32,
    4u32,
    6u32,
    "record_d192_128_4_4"
);
state_replay!(state_replay_d192_128_4_4, 192u32, 128u32, 4u32, 6u32, "replay_d192_128_4_4");
// Small unit-test cell: Dk=64, Dv=32, Hk=2, Hv=2.
gated_delta_record!(
    gated_delta_step_record_d64_32_2_2,
    64u32,
    32u32,
    2u32,
    2u32,
    2u32,
    "record_d64_32_2_2"
);
state_replay!(state_replay_d64_32_2_2, 64u32, 32u32, 2u32, 2u32, "replay_d64_32_2_2");

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{
        gated_delta_step_record_d64_32_2_2,
        gated_delta_step_record_d192_128_4_4,
        state_replay_d64_32_2_2,
        state_replay_d192_128_4_4,
    };

    const DK: usize = 64;
    const DV: usize = 32;
    const HK: usize = 2;
    const HV: usize = 2;

    // Production Qwen 3.5/3.6 A3B cell: Dk=192, Dv=128, Hk=4, Hv=4.
    const P_DK: usize = 192;
    const P_DV: usize = 128;
    const P_HK: usize = 4;
    const P_HV: usize = 4;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    // Forward recurrence with per-step delta-tape capture.
    #[bench(name = "ffai/gated_delta_record", dtypes = [f32, f16, bf16])]
    fn bench_gated_delta_record(dt: DType) -> BenchSetup {
        let (batch, t_val) = (1usize, 8usize);
        let n_total = batch * HV;
        BenchSetup::new(gated_delta_step_record_d64_32_2_2::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("q", batch * t_val * HK * DK, dt))
            .buffer(BenchBuffer::random("k", batch * t_val * HK * DK, dt))
            .buffer(BenchBuffer::random("v", batch * t_val * HV * DV, dt))
            .buffer(BenchBuffer::random("g", batch * t_val * HV, dt))
            .buffer(BenchBuffer::random("beta", batch * t_val * HV, dt))
            .buffer(BenchBuffer::random("state_in", n_total * DV * DK, dt))
            .buffer(BenchBuffer::from_vec(
                "mask",
                u32_bytes(&vec![1u32; batch * t_val]),
                DType::U32,
            ))
            .buffer(BenchBuffer::zeros("y", batch * t_val * HV * DV, dt).output())
            .buffer(BenchBuffer::zeros("state_out", n_total * DV * DK, dt).output())
            .buffer(BenchBuffer::zeros("delta_log", batch * t_val * HV * DV, dt).output())
            .constexpr("t_val", t_val as u32)
            .constexpr("has_mask", 0u32)
            .grid_3d(1, DV as u32, (batch * HV) as u32, [32, 1, 1])
            .bytes_moved((n_total * DV * DK * 2 * dt.size_bytes()) as u64)
    }

    // Branchless tape re-fold of the accepted prefix onto a snapshot.
    #[bench(name = "ffai/gated_delta_replay", dtypes = [f32, f16, bf16])]
    fn bench_gated_delta_replay(dt: DType) -> BenchSetup {
        let (batch, t_log, accepted) = (1usize, 8usize, 8usize);
        let n_total = batch * HV;
        BenchSetup::new(state_replay_d64_32_2_2::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("delta_log", batch * t_log * HV * DV, dt))
            .buffer(BenchBuffer::random("k_log", batch * t_log * HV * DK, dt))
            .buffer(BenchBuffer::random("g_log", batch * t_log * HV, dt))
            .buffer(BenchBuffer::random("state_in", n_total * DV * DK, dt))
            .buffer(BenchBuffer::from_vec(
                "mask",
                u32_bytes(&vec![1u32; batch * t_log]),
                DType::U32,
            ))
            .buffer(BenchBuffer::zeros("state_out", n_total * DV * DK, dt).output())
            .constexpr("t_log", t_log as u32)
            .constexpr("accepted", accepted as u32)
            .constexpr("has_mask", 0u32)
            .grid_3d(1, DV as u32, (batch * HV) as u32, [32, 1, 1])
            .bytes_moved((n_total * DV * DK * 2 * dt.size_bytes()) as u64)
    }

    // Production cell (Dk=192, Dv=128, Hk=4, Hv=4): forward recurrence with
    // per-step delta-tape capture.
    #[bench(name = "ffai/gated_delta_record_d192_128_4_4", dtypes = [f32, f16, bf16])]
    fn bench_gated_delta_record_d192_128_4_4(dt: DType) -> BenchSetup {
        let (batch, t_val) = (1usize, 8usize);
        let n_total = batch * P_HV;
        BenchSetup::new(gated_delta_step_record_d192_128_4_4::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("q", batch * t_val * P_HK * P_DK, dt))
            .buffer(BenchBuffer::random("k", batch * t_val * P_HK * P_DK, dt))
            .buffer(BenchBuffer::random("v", batch * t_val * P_HV * P_DV, dt))
            .buffer(BenchBuffer::random("g", batch * t_val * P_HV, dt))
            .buffer(BenchBuffer::random("beta", batch * t_val * P_HV, dt))
            .buffer(BenchBuffer::random("state_in", n_total * P_DV * P_DK, dt))
            .buffer(BenchBuffer::from_vec(
                "mask",
                u32_bytes(&vec![1u32; batch * t_val]),
                DType::U32,
            ))
            .buffer(BenchBuffer::zeros("y", batch * t_val * P_HV * P_DV, dt).output())
            .buffer(BenchBuffer::zeros("state_out", n_total * P_DV * P_DK, dt).output())
            .buffer(BenchBuffer::zeros("delta_log", batch * t_val * P_HV * P_DV, dt).output())
            .constexpr("t_val", t_val as u32)
            .constexpr("has_mask", 0u32)
            .grid_3d(1, P_DV as u32, (batch * P_HV) as u32, [32, 1, 1])
            .bytes_moved((n_total * P_DV * P_DK * 2 * dt.size_bytes()) as u64)
    }

    // Production cell (Dk=192, Dv=128, Hv=4): branchless tape re-fold of the
    // accepted prefix onto a snapshot.
    #[bench(name = "ffai/gated_delta_replay_d192_128_4_4", dtypes = [f32, f16, bf16])]
    fn bench_gated_delta_replay_d192_128_4_4(dt: DType) -> BenchSetup {
        let (batch, t_log, accepted) = (1usize, 8usize, 8usize);
        let n_total = batch * P_HV;
        BenchSetup::new(state_replay_d192_128_4_4::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("delta_log", batch * t_log * P_HV * P_DV, dt))
            .buffer(BenchBuffer::random("k_log", batch * t_log * P_HV * P_DK, dt))
            .buffer(BenchBuffer::random("g_log", batch * t_log * P_HV, dt))
            .buffer(BenchBuffer::random("state_in", n_total * P_DV * P_DK, dt))
            .buffer(BenchBuffer::from_vec(
                "mask",
                u32_bytes(&vec![1u32; batch * t_log]),
                DType::U32,
            ))
            .buffer(BenchBuffer::zeros("state_out", n_total * P_DV * P_DK, dt).output())
            .constexpr("t_log", t_log as u32)
            .constexpr("accepted", accepted as u32)
            .constexpr("has_mask", 0u32)
            .grid_3d(1, P_DV as u32, (batch * P_HV) as u32, [32, 1, 1])
            .bytes_moved((n_total * P_DV * P_DK * 2 * dt.size_bytes()) as u64)
    }
}
