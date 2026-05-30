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
        #[kernel]
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
        #[kernel]
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

/// New-syntax correctness for the GDN innovation-tape record + replay kernels
/// on the small `d64_32_2_2` cell (Dk=64, Dv=32, Hk=2, Hv=2). Oracles are
/// ported verbatim from `tests/gated_delta_replay_gpu_correctness.rs`:
///
///   - `record` runs the plain GatedDelta forward step and surfaces `delta_t`
///     to the tape; we check its `y` and `state_out`.
///   - `replay` re-folds the accepted prefix `[0, accepted)` of the tape onto
///     a state snapshot via the branchless `select`-gate; we check `state_out`.
///
/// Both kernels are single-dispatch Grid3D, `grid = [1, Dv, batch·Hv]`,
/// `tg = [32,1,1]`. We run with `has_mask=0` and a full (all-ones) mask buffer.
/// The record forward folds an `accepted`-step recurrence, so tolerances follow
/// the legacy 2e-3 f32 bar widened for f16/bf16: `[1e-3, 5e-3, 2e-2]`.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::{gated_delta_step_record_d64_32_2_2, state_replay_d64_32_2_2};
    use crate::utils::{pack_f32, unpack_f32};

    // d64_32_2_2 cell dims.
    const DK: usize = 64;
    const DV: usize = 32;
    const HK: usize = 2;
    const HV: usize = 2;

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// Forward recurrence + delta capture — ported `naive_record`.
    #[allow(clippy::too_many_arguments)]
    fn naive_record(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        state_in: &[f32],
        batch: usize,
        t_val: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut y = vec![0.0_f32; batch * t_val * HV * DV];
        let mut state = state_in.to_vec();
        for n in 0..batch * HV {
            let b = n / HV;
            let hvh = n % HV;
            let hkh = hvh / (HV / HK);
            for t in 0..t_val {
                let qk = (b * t_val + t) * HK * DK + hkh * DK;
                let vb = (b * t_val + t) * HV * DV + hvh * DV;
                let gb = (b * t_val + t) * HV + hvh;
                for dv in 0..DV {
                    let s0 = (n * DV + dv) * DK;
                    let mut kv = 0.0_f32;
                    for dk in 0..DK {
                        state[s0 + dk] *= g[gb];
                        kv += state[s0 + dk] * k[qk + dk];
                    }
                    let delta = (v[vb + dv] - kv) * beta[gb];
                    let mut out = 0.0_f32;
                    for dk in 0..DK {
                        state[s0 + dk] += k[qk + dk] * delta;
                        out += state[s0 + dk] * q[qk + dk];
                    }
                    y[vb + dv] = out;
                }
            }
        }
        (y, state)
    }

    /// Branchless tape re-fold of the accepted prefix — ported `naive_replay`.
    #[allow(clippy::too_many_arguments)]
    fn naive_replay(
        delta_log: &[f32],
        k_log: &[f32],
        g_log: &[f32],
        state_in: &[f32],
        batch: usize,
        t_log: usize,
        accepted: usize,
    ) -> Vec<f32> {
        let mut state = state_in.to_vec();
        for n in 0..batch * HV {
            let b = n / HV;
            let hvh = n % HV;
            for t in 0..t_log {
                if t >= accepted {
                    continue;
                }
                let dr = (b * t_log + t) * HV * DV + hvh * DV;
                let kr = (b * t_log + t) * HV * DK + hvh * DK;
                let g = g_log[(b * t_log + t) * HV + hvh];
                for dv in 0..DV {
                    let s0 = (n * DV + dv) * DK;
                    for dk in 0..DK {
                        state[s0 + dk] = state[s0 + dk] * g + k_log[kr + dk] * delta_log[dr + dv];
                    }
                }
            }
        }
        state
    }

    /// Deterministic xorshift fixture (mirrors the legacy `src`).
    fn src(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 20_000) as f32 / 20_000.0 * scale - scale * 0.5
            })
            .collect()
    }

    /// Record cell: T=3 forward steps, verify `y` and `state_out`.
    fn record_setup(dt: DType) -> TestSetup {
        let (batch, t_val) = (1usize, 3usize);
        let n_total = batch * HV;
        let q = src(batch * t_val * HK * DK, 0x1, 0.4);
        let k = src(batch * t_val * HK * DK, 0x2, 0.4);
        let v = src(batch * t_val * HV * DV, 0x3, 1.0);
        let g: Vec<f32> = src(batch * t_val * HV, 0x4, 0.1).iter().map(|x| 0.9 + x).collect();
        let beta: Vec<f32> = src(batch * t_val * HV, 0x5, 0.1).iter().map(|x| 0.5 + x).collect();
        let state_in = src(n_total * DV * DK, 0x6, 0.2);

        // Dtype-round inputs so the oracle matches the GPU's load precision.
        let r = |xs: &[f32]| unpack_f32(&pack_f32(xs, dt), dt);
        let (y_exp, s_exp) =
            naive_record(&r(&q), &r(&k), &r(&v), &r(&g), &r(&beta), &r(&state_in), batch, t_val);

        TestSetup::new(gated_delta_step_record_d64_32_2_2::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("q", pack_f32(&q, dt), dt))
            .input(TestBuffer::from_vec("k", pack_f32(&k, dt), dt))
            .input(TestBuffer::from_vec("v", pack_f32(&v, dt), dt))
            .input(TestBuffer::from_vec("g", pack_f32(&g, dt), dt))
            .input(TestBuffer::from_vec("beta", pack_f32(&beta, dt), dt))
            .input(TestBuffer::from_vec("state_in", pack_f32(&state_in, dt), dt))
            .input(TestBuffer::from_vec("mask", u32_bytes(&vec![1u32; batch * t_val]), DType::U32))
            .input(TestBuffer::zeros("y", batch * t_val * HV * DV, dt))
            .input(TestBuffer::zeros("state_out", n_total * DV * DK, dt))
            .input(TestBuffer::zeros("delta_log", batch * t_val * HV * DV, dt))
            .constexpr("t_val", t_val as u32)
            .constexpr("has_mask", 0u32)
            .expect(TestBuffer::from_vec("y", pack_f32(&y_exp, dt), dt))
            .expect(TestBuffer::from_vec("state_out", pack_f32(&s_exp, dt), dt))
            .grid_3d(1, DV as u32, (batch * HV) as u32, [32, 1, 1])
    }

    /// Replay cell: T=5 tape, accept the first 3 steps; verify `state_out`.
    fn replay_setup(dt: DType) -> TestSetup {
        let (batch, t_log, accepted) = (1usize, 5usize, 3usize);
        let n_total = batch * HV;
        let delta_log = src(batch * t_log * HV * DV, 0x21, 0.5);
        let k_log = src(batch * t_log * HV * DK, 0x22, 0.4);
        let g_log: Vec<f32> = src(batch * t_log * HV, 0x23, 0.1).iter().map(|x| 0.9 + x).collect();
        let state_in = src(n_total * DV * DK, 0x24, 0.3);

        let r = |xs: &[f32]| unpack_f32(&pack_f32(xs, dt), dt);
        let s_exp = naive_replay(
            &r(&delta_log),
            &r(&k_log),
            &r(&g_log),
            &r(&state_in),
            batch,
            t_log,
            accepted,
        );

        TestSetup::new(state_replay_d64_32_2_2::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("delta_log", pack_f32(&delta_log, dt), dt))
            .input(TestBuffer::from_vec("k_log", pack_f32(&k_log, dt), dt))
            .input(TestBuffer::from_vec("g_log", pack_f32(&g_log, dt), dt))
            .input(TestBuffer::from_vec("state_in", pack_f32(&state_in, dt), dt))
            .input(TestBuffer::from_vec("mask", u32_bytes(&vec![1u32; batch * t_log]), DType::U32))
            .input(TestBuffer::zeros("state_out", n_total * DV * DK, dt))
            .constexpr("t_log", t_log as u32)
            .constexpr("accepted", accepted as u32)
            .constexpr("has_mask", 0u32)
            .expect(TestBuffer::from_vec("state_out", pack_f32(&s_exp, dt), dt))
            .grid_3d(1, DV as u32, (batch * HV) as u32, [32, 1, 1])
    }

    // Forward record: y + state_out match the plain GDN recurrence.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-3, 2e-2])]
    fn test_gated_delta_record_d64_32_2_2(dt: DType) -> TestSetup { record_setup(dt) }

    // Replay: state_out matches the accepted-prefix re-fold.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-3, 5e-3, 2e-2])]
    fn test_state_replay_d64_32_2_2(dt: DType) -> TestSetup { replay_setup(dt) }
}

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
