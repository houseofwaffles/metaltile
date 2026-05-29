//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Mamba / Mamba 2 state replay — port of `ssm_replay.metal`
//! (spec 040). The speculative-decode rollback companion to
//! `ssm.rs`'s `ssm_step`. Two kernels:
//!
//!   - `ssm_step_record` — the sequential SSD forward over `t_total`
//!     steps, capturing each step's `(dA, dBx)` into delta logs
//!     alongside the standard `(y, state_out)`.
//!   - `ssm_replay` — re-folds the first `k` log entries onto a
//!     recurrent-state snapshot to recover state-after-k.
//!
//! Threading (matches `ssm.metal`): a 32-lane simdgroup splits the
//! `Ds` state axis (`n_per_t = Ds/32` per lane); `program_id::<1>()`
//! = `Dh` index, `program_id::<2>()` = `batch*H + h`. `simd_sum`
//! reduces `y = C·state` across the `Ds` lanes.
//!
//! Layouts: `x` / `y` / `dt` [B,T,H,Dh|H]; `B` / `C` [B,T,G,Ds];
//! `state` [B,H,Dh,Ds]; `dA_log` [B,T,H,Ds]; `dBx_log` [B,T,H,Dh,Ds].
//! `mask` (u32, 0/1) makes a masked timestep identity (`dA=1, dBx=0`)
//! so rollback past it is order-preserving.
//!
//! ## DISPATCH INVARIANTS
//!
//! - **Grid3D**, `grid = [1, Dh, batch*H]`, `tg = [32, 1, 1]`.
//! - `Ds` a multiple of 32.
//!
//! Codegen-only; correctness pinned by
//! `tests/ssm_replay_gpu_correctness.rs`.

use metaltile::kernel;

// ── SSD forward step with (dA, dBx) tape capture ────────────────────────────
#[rustfmt::skip]
macro_rules! ssm_step_record {
    ($name:ident, $dh:literal, $ds:literal, $h:literal, $g:literal, $n_per_t:literal, $subop:literal) => {
        #[kernel(
            bench(
                op="ssm_replay",
                subop=$subop,
                class=GenericEmpty,
                tol=1e-3,
                kernel_mode=Grid3D,
            )
        )]
        pub fn $name<T>(
            x: Tensor<T>,
            a_log: Tensor<T>,
            b: Tensor<T>,
            c: Tensor<T>,
            d: Tensor<T>,
            dt: Tensor<T>,
            state_in: Tensor<T>,
            mask: Tensor<u32>,
            mut y: Tensor<T>,
            mut state_out: Tensor<T>,
            mut da_log: Tensor<T>,
            mut dbx_log: Tensor<T>,
            #[constexpr] t_total: u32,
            #[constexpr] has_mask: u32,
        ) {
            let ds_lane = program_id::<0>();
            let d_idx = program_id::<1>();
            let n = program_id::<2>();
            let h_idx = n - (n / $h) * $h;
            let b_idx = n / $h;
            let g_idx = h_idx / ($h / $g);
            let state_base = (n * $dh + d_idx) * $ds;

            stack_alloc("state", $n_per_t, "f32");
            for i in range(0u32, $n_per_t, 1u32) {
                let v = load(state_in[state_base + $n_per_t * ds_lane + i]).cast::<f32>();
                stack_store("state", i, v);
            }

            // A = -exp(A_log[h]).
            let a_neg = 0.0f32 - exp(load(a_log[h_idx]).cast::<f32>());

            for t in range(0u32, t_total, 1u32) {
                let bt = b_idx * t_total + t;
                let bt_h = bt * $h + h_idx;
                let bt_g = bt * $g + g_idx;
                let active = select(has_mask == 0u32, 1u32, load(mask[bt]));

                let dt_raw = load(dt[bt_h]).cast::<f32>();
                // Masked step: dA=1, dt_eff=0 → identity recurrence.
                let dt_eff = select(active > 0u32, dt_raw, 0.0f32);
                let d_a = select(active > 0u32, exp(a_neg * dt_raw), 1.0f32);

                // Capture dA (same scalar in every Ds slot for this lane).
                for i in range(0u32, $n_per_t, 1u32) {
                    store(da_log[bt_h * $ds + $n_per_t * ds_lane + i], d_a.cast::<T>());
                }

                let x_v = load(x[bt_h * $dh + d_idx]).cast::<f32>();
                let dbx_base = (bt_h * $dh + d_idx) * $ds;
                let mut y_acc = 0.0f32;
                for i in range(0u32, $n_per_t, 1u32) {
                    let s_idx = $n_per_t * ds_lane + i;
                    let b_v = load(b[bt_g * $ds + s_idx]).cast::<f32>();
                    let dbx = x_v * dt_eff * b_v;
                    store(dbx_log[dbx_base + s_idx], dbx.cast::<T>());
                    let st = d_a * stack_load("state", i) + dbx;
                    stack_store("state", i, st);
                    y_acc = y_acc + st * load(c[bt_g * $ds + s_idx]).cast::<f32>();
                }
                let y_sum = simd_sum(y_acc);
                if ds_lane == 0u32 {
                    let y_d = y_sum + x_v * load(d[h_idx]).cast::<f32>();
                    store(y[bt_h * $dh + d_idx], y_d.cast::<T>());
                }
            }

            for i in range(0u32, $n_per_t, 1u32) {
                let st = stack_load("state", i);
                store(state_out[state_base + $n_per_t * ds_lane + i], st.cast::<T>());
            }
        }
    };
}

// ── Tape replay: re-fold the first k log entries onto a snapshot ────────────
#[rustfmt::skip]
macro_rules! ssm_replay {
    ($name:ident, $dh:literal, $ds:literal, $h:literal, $n_per_t:literal, $subop:literal) => {
        #[kernel(
            bench(
                op="ssm_replay",
                subop=$subop,
                class=GenericEmpty,
                tol=1e-3,
                kernel_mode=Grid3D,
            )
        )]
        pub fn $name<T>(
            state_snapshot: Tensor<T>,
            da_log: Tensor<T>,
            dbx_log: Tensor<T>,
            mask: Tensor<u32>,
            mut state_after_k: Tensor<T>,
            #[constexpr] k_steps: u32,
            #[constexpr] t_total: u32,
            #[constexpr] has_mask: u32,
        ) {
            let ds_lane = program_id::<0>();
            let d_idx = program_id::<1>();
            let n = program_id::<2>();
            let h_idx = n - (n / $h) * $h;
            let b_idx = n / $h;
            let state_base = (n * $dh + d_idx) * $ds;

            stack_alloc("state", $n_per_t, "f32");
            for i in range(0u32, $n_per_t, 1u32) {
                let v = load(state_snapshot[state_base + $n_per_t * ds_lane + i]).cast::<f32>();
                stack_store("state", i, v);
            }

            for t in range(0u32, k_steps, 1u32) {
                let bt = b_idx * t_total + t;
                let bt_h = bt * $h + h_idx;
                let active = select(has_mask == 0u32, 1u32, load(mask[bt]));
                let dbx_base = (bt_h * $dh + d_idx) * $ds;
                for i in range(0u32, $n_per_t, 1u32) {
                    let s_idx = $n_per_t * ds_lane + i;
                    let old = stack_load("state", i);
                    let d_a = load(da_log[bt_h * $ds + s_idx]).cast::<f32>();
                    let dbx = load(dbx_log[dbx_base + s_idx]).cast::<f32>();
                    let new_val = d_a * old + dbx;
                    // Masked steps were recorded as dA=1, dBx=0 (identity),
                    // but guard anyway so a stale tape entry can't perturb.
                    stack_store("state", i, select(active > 0u32, new_val, old));
                }
            }

            for i in range(0u32, $n_per_t, 1u32) {
                let st = stack_load("state", i);
                store(state_after_k[state_base + $n_per_t * ds_lane + i], st.cast::<T>());
            }
        }
    };
}

// Small unit-test cell: Dh=16, Ds=64, H=4, G=2.
ssm_step_record!(ssm_step_record_d16_64_4_2, 16u32, 64u32, 4u32, 2u32, 2u32, "record_d16_64_4_2");
ssm_replay!(ssm_replay_d16_64_4, 16u32, 64u32, 4u32, 2u32, "replay_d16_64_4");
// Production cell: Dh=128, Ds=128, H=32, G=2 (Jamba / Nemotron class).
ssm_step_record!(
    ssm_step_record_d128_128_32_2,
    128u32,
    128u32,
    32u32,
    2u32,
    4u32,
    "record_d128_128_32_2"
);
ssm_replay!(ssm_replay_d128_128_32, 128u32, 128u32, 32u32, 4u32, "replay_d128_128_32");

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{
        ssm_replay_d16_64_4,
        ssm_replay_d128_128_32,
        ssm_step_record_d16_64_4_2,
        ssm_step_record_d128_128_32_2,
    };

    const DH: usize = 16;
    const DS: usize = 64;
    const H: usize = 4;
    const G: usize = 2;

    // Production cell: Dh=128, Ds=128, H=32, G=2 (Jamba / Nemotron class).
    const P_DH: usize = 128;
    const P_DS: usize = 128;
    const P_H: usize = 32;
    const P_G: usize = 2;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    // Sequential SSD forward with (dA, dBx) tape capture over `t_total` steps.
    #[bench(name = "ffai/ssm_record", dtypes = [f32, f16, bf16])]
    fn bench_ssm_record(dt: DType) -> BenchSetup {
        let (batch, t) = (1usize, 8usize);
        let n_total = batch * H;
        BenchSetup::new(ssm_step_record_d16_64_4_2::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("x", batch * t * H * DH, dt))
            .buffer(BenchBuffer::random("a_log", H, dt))
            .buffer(BenchBuffer::random("b", batch * t * G * DS, dt))
            .buffer(BenchBuffer::random("c", batch * t * G * DS, dt))
            .buffer(BenchBuffer::random("d", H, dt))
            .buffer(BenchBuffer::random("dt", batch * t * H, dt))
            .buffer(BenchBuffer::random("state_in", n_total * DH * DS, dt))
            .buffer(BenchBuffer::from_vec("mask", u32_bytes(&vec![1u32; batch * t]), DType::U32))
            .buffer(BenchBuffer::zeros("y", batch * t * H * DH, dt).output())
            .buffer(BenchBuffer::zeros("state_out", n_total * DH * DS, dt).output())
            .buffer(BenchBuffer::zeros("da_log", batch * t * H * DS, dt).output())
            .buffer(BenchBuffer::zeros("dbx_log", batch * t * H * DH * DS, dt).output())
            .constexpr("t_total", t as u32)
            .constexpr("has_mask", 0u32)
            .grid_3d(1, DH as u32, (batch * H) as u32, [32, 1, 1])
            .bytes_moved((batch * t * H * DH * DS * dt.size_bytes()) as u64)
    }

    // Re-fold the first `k_steps` tape entries onto a recurrent-state snapshot.
    #[bench(name = "ffai/ssm_replay", dtypes = [f32, f16, bf16])]
    fn bench_ssm_replay(dt: DType) -> BenchSetup {
        let (batch, t, k_steps) = (1usize, 8usize, 8usize);
        let n_total = batch * H;
        BenchSetup::new(ssm_replay_d16_64_4::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("state_snapshot", n_total * DH * DS, dt))
            .buffer(BenchBuffer::random("da_log", batch * t * H * DS, dt))
            .buffer(BenchBuffer::random("dbx_log", batch * t * H * DH * DS, dt))
            .buffer(BenchBuffer::from_vec("mask", u32_bytes(&vec![1u32; batch * t]), DType::U32))
            .buffer(BenchBuffer::zeros("state_after_k", n_total * DH * DS, dt).output())
            .constexpr("k_steps", k_steps as u32)
            .constexpr("t_total", t as u32)
            .constexpr("has_mask", 0u32)
            .grid_3d(1, DH as u32, (batch * H) as u32, [32, 1, 1])
            .bytes_moved((batch * t * H * DH * DS * dt.size_bytes()) as u64)
    }

    // Production cell (Dh=128, Ds=128, H=32, G=2): sequential SSD forward
    // with (dA, dBx) tape capture over `t_total` steps.
    #[bench(name = "ffai/ssm_record_d128_128_32", dtypes = [f32, f16, bf16])]
    fn bench_ssm_record_d128_128_32(dt: DType) -> BenchSetup {
        let (batch, t) = (1usize, 8usize);
        let n_total = batch * P_H;
        BenchSetup::new(ssm_step_record_d128_128_32_2::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("x", batch * t * P_H * P_DH, dt))
            .buffer(BenchBuffer::random("a_log", P_H, dt))
            .buffer(BenchBuffer::random("b", batch * t * P_G * P_DS, dt))
            .buffer(BenchBuffer::random("c", batch * t * P_G * P_DS, dt))
            .buffer(BenchBuffer::random("d", P_H, dt))
            .buffer(BenchBuffer::random("dt", batch * t * P_H, dt))
            .buffer(BenchBuffer::random("state_in", n_total * P_DH * P_DS, dt))
            .buffer(BenchBuffer::from_vec("mask", u32_bytes(&vec![1u32; batch * t]), DType::U32))
            .buffer(BenchBuffer::zeros("y", batch * t * P_H * P_DH, dt).output())
            .buffer(BenchBuffer::zeros("state_out", n_total * P_DH * P_DS, dt).output())
            .buffer(BenchBuffer::zeros("da_log", batch * t * P_H * P_DS, dt).output())
            .buffer(BenchBuffer::zeros("dbx_log", batch * t * P_H * P_DH * P_DS, dt).output())
            .constexpr("t_total", t as u32)
            .constexpr("has_mask", 0u32)
            .grid_3d(1, P_DH as u32, (batch * P_H) as u32, [32, 1, 1])
            .bytes_moved((batch * t * P_H * P_DH * P_DS * dt.size_bytes()) as u64)
    }

    // Production cell (Dh=128, Ds=128, H=32): re-fold the first `k_steps`
    // tape entries onto a recurrent-state snapshot.
    #[bench(name = "ffai/ssm_replay_d128_128_32", dtypes = [f32, f16, bf16])]
    fn bench_ssm_replay_d128_128_32(dt: DType) -> BenchSetup {
        let (batch, t, k_steps) = (1usize, 8usize, 8usize);
        let n_total = batch * P_H;
        BenchSetup::new(ssm_replay_d128_128_32::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .buffer(BenchBuffer::random("state_snapshot", n_total * P_DH * P_DS, dt))
            .buffer(BenchBuffer::random("da_log", batch * t * P_H * P_DS, dt))
            .buffer(BenchBuffer::random("dbx_log", batch * t * P_H * P_DH * P_DS, dt))
            .buffer(BenchBuffer::from_vec("mask", u32_bytes(&vec![1u32; batch * t]), DType::U32))
            .buffer(BenchBuffer::zeros("state_after_k", n_total * P_DH * P_DS, dt).output())
            .constexpr("k_steps", k_steps as u32)
            .constexpr("t_total", t as u32)
            .constexpr("has_mask", 0u32)
            .grid_3d(1, P_DH as u32, (batch * P_H) as u32, [32, 1, 1])
            .bytes_moved((batch * t * P_H * P_DH * P_DS * dt.size_bytes()) as u64)
    }
}
