//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! AURA fused encode — full per-vector encode pipeline:
//!
//!   1. L2 norm via `simd_sum` + cross-simdgroup reduction in
//!      threadgroup memory.
//!   2. Rotation Π via shared-memory matmul (each thread emits one
//!      rotated component).
//!   3. Quantize via branchless boundary comparison against the
//!      Lloyd-Max boundaries.
//!   4. Pack the codebook indices into a u32 stream via
//!      `atomic_or_tg` so threads racing on the same packed word
//!      (bit slots `[d*Bits, (d+1)*Bits)` may share a u32 word) are
//!      properly serialised.
//!   5. Norm correction — re-run the reduction over the dequantised
//!      centroid values to derive the per-vector norm-correction
//!      factor that the decoder multiplies back through.
//!
//! Port of `turbo_fused_encode` from
//! `ekryski/mlx@alpha:mlx/backend/metal/kernels/turbo_quant.metal`.
//! One threadgroup per input row; one thread per dim slot.
//!
//! ## Layout
//!
//! Inputs:
//! - `input      [rows, dim]`           T    — model dtype (bf16/f16/f32).
//! - `rotation   [dim, dim]`            T    — Π; the dim×dim read dominates
//!   this kernel's bandwidth, so it follows the model dtype (cast-to-f32 at
//!   load, f32 accumulation). f16/bf16 Π rounding (~1e-3) is far below the
//!   2–4-bit quantization bin width that consumes the rotated value.
//! - `boundaries [2**bits - 1]`         f32  — Lloyd-Max thresholds. Kept f32
//!   (not T) because the buffer is `2^bits - 1` floats per scheme — `60
//!   bytes` at 4-bit, ~`1 KB` worst-case at 8-bit. The bandwidth argument that
//!   moves Π to T does not apply here, and at 4-bit aura the bf16 boundary
//!   rounding flips a small but measurable fraction of borderline bin
//!   assignments — costs ~0.3 nats KLD on Qwen3-0.6B-4bit aura4v4 with no
//!   bandwidth payback. See FFAI's `AuraKLDIntegrationTests` for the gate.
//! - `codebook   [2**bits]`             T    — centroid values, dtype matched
//!   to the decoder cache so the same buffer feeds both paths with no
//!   per-call cast.
//!
//! Outputs:
//! - `packed_out [rows, packed_width]`  u32
//! - `norms_out  [rows]`                T    — norm-correction factor; cast at
//!   the final store (the internal reduction stays f32).
//!
//! ## Constexpr params
//!
//! - `bits`         — quant bit-width (2 / 3 / 4 / 8).
//! - `dim`          — vector length (64 / 80 / 96 / 128 / 256 / 512).
//! - `packed_width` — `ceil(dim * bits / 32)`.
//! - `levels`       — `1 << bits`.
//!
//! ## DISPATCH INVARIANTS
//!
//! This kernel is reduction-mode and has STRICT threadgroup-geometry
//! requirements. Violating any of these silently miscomputes the
//! encoded output (best case) or pins the GPU in an infinite loop
//! (worst case — see FFAI post-mortem 2026-05-19). Consumers MUST
//! encode these as preconditions in their wrappers.
//!
//! - **TPG = `dim`.** One thread per rotated coordinate. Each thread's
//!   slot in `shared_unit` is `tid`; loads/stores are unconditional.
//! - **`dim` must be a multiple of 32** (one full Apple simdgroup).
//!   The L2-norm `simd_sum` is only well-defined over full simdgroups;
//!   `dim < 32` produces undefined behaviour across lanes.
//! - **`dim ≤ 1024`** (Apple's max-threads-per-threadgroup cap, and
//!   matches the static `threadgroup_alloc("shared_unit", 1024)`).
//! - **`shared_norm[16]`** holds one partial per simdgroup. Adequate
//!   for `dim ≤ 16 * 32 = 512`; dims 513..1024 would overflow this
//!   buffer (currently bench-only — production AURA dims are 64/96/
//!   128/192/256).
//! - **Grid: 1 threadgroup per row.** Wrapper uses
//!   `grid = (dim, rows, 1)`, `tg = (dim, 1, 1)` so Metal slices that
//!   into `rows` threadgroups of `dim` threads.
//!
//! ## Macro structure
//!
//! `aura_encode_kernel!` wraps a single `#[kernel(bench(...))] pub fn …`
//! at module scope.  Bit-widths get separate invocations so the compiler
//! expands the outer macro before the `#[kernel]` proc-macro sees it —
//! required because the proc-macro does not expand inner declarative macros.

use metaltile::kernel;

macro_rules! aura_encode_kernel {
    ($name:ident, $bits:literal, $levels:literal, $subop:literal) => {
        // `input` / `rotation` / `codebook` / `norms_out` are model dtype
        // T (bf16/f16 in production, f32 in tests) — we cast each load
        // to f32 and accumulate in f32.
        //
        // The f32 *accumulation* is load-bearing and stays: the L2-norm
        // reductions (Stages 1 & 5) and the dim-length rotation matmul
        // (Stage 2) each sum up to `dim` (≤512) terms, and the resulting
        // norm-correction factor scales the *entire* dequantized vector
        // in the decoder — f16 accumulation there drifts enough to hurt.
        // The *storage* precision is what was overkill: the dim×dim
        // rotation matrix dominates this kernel's bandwidth, so storing
        // it (and the codebook + norms_out) in T halves the dominant
        // read. Its f16/bf16 rounding (~1e-3) is far below the 2–4-bit
        // quant bin the rotated value lands in, and the decoder's
        // inverse rotation is already a model-dtype gemm — so f32 Π
        // here was strictly more precise than the round-trip it feeds.
        //
        // `boundaries` stays f32. The buffer is `2^bits - 1` floats per
        // scheme — 60 bytes at 4-bit, ~1 KB at 8-bit — so the bandwidth
        // argument for narrowing Π does not apply, and at 4-bit aura
        // the bf16 rounding flips a small fraction of borderline bin
        // assignments at the Stage-3 branchless compare. Measured on
        // FFAI's KLD harness (Qwen3-0.6B-4bit, 61-position): T-typed
        // boundaries → aura4v4 compressed-flash KLD 1.76; f32 boundaries
        // → 1.40 (mirror baseline 1.41). The 0.3-nat gap maps cleanly
        // to the bf16 boundary rounding flipping borderline bins.
        #[kernel(
            bench(op="aura", subop=$subop, class=GenericEmpty, tol=0.0, kernel_mode=Reduction,)
        )]
        pub fn $name<T>(
            input: Tensor<T>,
            rotation: Tensor<T>,
            boundaries: Tensor<f32>,
            codebook: Tensor<T>,
            mut packed_out: Tensor<u32>,
            mut norms_out: Tensor<T>,
            #[constexpr] dim: u32,
            #[constexpr] packed_width: u32,
        ) {
            let d = tid;
            let row = tgid_x;

            // ── Stage 1: per-thread L2 norm via simd_sum + cross-simdgroup
            // reduction through threadgroup memory.  `shared_norm[16]`
            // holds one partial per simdgroup (16 is enough for dim ≤ 512).
            let val = load(input[row * dim + d]).cast::<f32>();
            let sq = val * val;
            let simd_norm_sq = simd_sum(sq);
            threadgroup_alloc("shared_norm", 16);
            let sg_id = d / 32u32;
            let lane = d & 31u32;
            if lane == 0u32 {
                threadgroup_store("shared_norm", sg_id, simd_norm_sq);
            }
            threadgroup_barrier();

            let mut total_norm_sq = 0.0f32;
            let num_groups = (dim + 31u32) / 32u32;
            for i in range(0u32, num_groups, 1u32) {
                total_norm_sq = total_norm_sq + threadgroup_load("shared_norm", i);
            }
            let norm_val = sqrt(total_norm_sq);
            let inv_norm = select(norm_val > 1.0e-8f32, 1.0f32 / norm_val, 0.0f32);
            let unit_val = val * inv_norm;

            // ── Stage 2: rotation Π via shared-memory matmul.  Each
            // thread stores its unit value, barriers, then reads the
            // full row to compute its rotated component.  `shared_unit`
            // sized at 1024 to match the MLX upstream's max dim.
            threadgroup_alloc("shared_unit", 1024);
            threadgroup_store("shared_unit", d, unit_val);
            threadgroup_barrier();
            let mut rotated = 0.0f32;
            for j in range(0u32, dim, 1u32) {
                rotated = rotated
                    + load(rotation[d * dim + j]).cast::<f32>()
                        * threadgroup_load("shared_unit", j);
            }

            // ── Stage 3: branchless boundary comparison → codebook
            // index.  For LEVELS levels we have LEVELS-1 boundaries;
            // the index is the count of boundaries the rotated value
            // exceeds.
            let mut idx = 0u32;
            for b in range(0u32, $levels - 1u32, 1u32) {
                idx = idx + (rotated > load(boundaries[b])).cast::<u32>();
            }

            // ── Stage 4: pack the index into the u32 stream via
            // `atomic_or_tg` on threadgroup memory.  `shared_packed`
            // sized for the worst-case PackedWidth (D=512, bits=8 →
            // 128).  Other (dim, bits) combos use the prefix only.
            let bit_offset = d * $bits;
            let word_idx = bit_offset / 32u32;
            let shift = bit_offset & 31u32;
            let masked = idx & ((1u32 << $bits) - 1u32);

            threadgroup_alloc("shared_packed", 128, "u32");
            if d < packed_width {
                threadgroup_store("shared_packed", d, 0u32);
            }
            threadgroup_barrier();

            atomic_or_tg("shared_packed", word_idx, masked << shift);
            // Cross-word spill — write the high bits into the next u32
            // if the index straddles a word boundary.
            //
            // Use unsigned-only arithmetic.  The original formulation
            // `(shift + bits).cast::<i32>() - 32i32 > 0i32` lost its
            // signedness in codegen (lowered to `int v_spill_bits =
            // (int)(shift + bits) - 32; bool = v_spill_bits > 0u`),
            // making the comparison int-vs-uint.  C/MSL promotes the
            // int to uint, so -28 becomes ~4e9 and the spill branch
            // ran for EVERY thread, polluting `shared_packed[word+1]`
            // with garbage shifted by `masked >> 32` (which Apple
            // Silicon evaluates as `masked >> 0 = masked`).  Symptom:
            // low nibbles of subsequent words OR'd with unrelated dim
            // indices.  Caught by the aura_encode GPU correctness
            // test.  See FFAI post-mortem 2026-05-19 — a metaltile
            // codegen follow-up should emit i32 comparisons
            // faithfully when the DSL requests them.
            let total_bits = shift + $bits;
            if total_bits > 32u32 {
                let spill_u = total_bits - 32u32;
                atomic_or_tg("shared_packed", word_idx + 1u32, masked >> ($bits - spill_u));
            }
            threadgroup_barrier();

            if d < packed_width {
                store(packed_out[row * packed_width + d], threadgroup_load("shared_packed", d));
            }

            // ── Stage 5: norm correction.  Re-run the reduction over
            // the dequantised centroid values; `recon_norm` gives the
            // L2 norm of the reconstructed vector, and
            // `corrected = norm_val / recon_norm` is what the decoder
            // multiplies back through.
            let centroid_val = load(codebook[idx]).cast::<f32>();
            let recon_sq = centroid_val * centroid_val;
            let simd_recon_sq = simd_sum(recon_sq);
            if lane == 0u32 {
                threadgroup_store("shared_norm", sg_id, simd_recon_sq);
            }
            threadgroup_barrier();
            let mut total_recon_sq = 0.0f32;
            for i in range(0u32, num_groups, 1u32) {
                total_recon_sq = total_recon_sq + threadgroup_load("shared_norm", i);
            }
            let recon_norm = sqrt(total_recon_sq);
            let corrected_norm = select(recon_norm > 1.0e-8f32, norm_val / recon_norm, norm_val);

            if d == 0u32 {
                store(norms_out[row], corrected_norm.cast::<T>());
            }
        }
    };
}

aura_encode_kernel!(aura_encode_int2, 2u32, 4u32, "encode_int2");
aura_encode_kernel!(aura_encode_int3, 3u32, 8u32, "encode_int3");
aura_encode_kernel!(aura_encode_int4, 4u32, 16u32, "encode_int4");
aura_encode_kernel!(aura_encode_int6, 6u32, 64u32, "encode_int6");
aura_encode_kernel!(aura_encode_int8, 8u32, 256u32, "encode_int8");

/// New-syntax correctness for the AURA fused encode kernel. Mirrors the
/// affine-quantize tests' strategy: the **packed codes** go through a
/// branchless boundary-count whose last bit and bit-packing are sensitive to
/// Metal fast-math FMA fusion, so they stay covered by the legacy bit-exact
/// `aura_encode_gpu_correctness.rs` A/B test. Here we pin the part that is
/// robustly checkable — the per-vector **norm-correction factor**
/// (`norms_out`), which the decoder multiplies back through.
///
/// Setup uses an **identity rotation** so the Stage-2 matmul is `rotated =
/// unit_val` exactly (no reorder ambiguity in the quant index), and a smooth
/// input chosen to land each rotated coordinate comfortably mid-bin — so the
/// codebook index feeding `norms_out` is stable under input dtype rounding.
/// `input` / `rotation` / `codebook` / `norms_out` are `Tensor<T>` and
/// dtype-rounded here to match the kernel's cast-at-load. The identity rotation
/// is exact in all dtypes; `norms_out` is compared at the per-dtype tol.
/// `boundaries` stays `Tensor<f32>` — the 60-byte 4-bit (~1 KB worst-case
/// 8-bit) buffer isn't bandwidth-bound, and the bf16-boundary rounding
/// measurably degrades 4-bit AURA quality at decode time. `packed_out` is
/// provided but NOT expected (fast-math packing — the legacy f32 A/B test owns
/// the bit-exact check).
///
/// Grid (Reduction, TPG = dim): `grid_3d(rows, 1, 1, [dim,1,1])`.
pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::aura_encode_int4;
    use crate::utils::{pack_f32, unpack_f32};

    fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    /// Identity rotation `[dim, dim]` — row d is the unit vector e_d.
    fn identity_rotation(dim: usize) -> Vec<f32> {
        let mut r = vec![0.0_f32; dim * dim];
        for d in 0..dim {
            r[d * dim + d] = 1.0;
        }
        r
    }

    /// Symmetric int4 codebook: 16 evenly-spaced centroids in [-1, 1], with the
    /// 15 boundaries at the midpoints (mirrors `int4_uniform_codebook` in the
    /// legacy test).
    fn int4_uniform_codebook() -> (Vec<f32>, Vec<f32>) {
        let levels = 16usize;
        let codebook: Vec<f32> =
            (0..levels).map(|i| -1.0 + 2.0 * (i as f32) / (levels as f32 - 1.0)).collect();
        let boundaries: Vec<f32> =
            (0..levels - 1).map(|i| 0.5 * (codebook[i] + codebook[i + 1])).collect();
        (codebook, boundaries)
    }

    /// CPU oracle for `norms_out` only. Replicates Stages 1/3/5 of the kernel
    /// under an identity rotation (so `rotated = unit_val`): L2-normalise the
    /// row, boundary-count each coordinate into a codebook index, then
    /// `corrected = norm / ‖centroids‖`.
    fn norms_oracle(
        input: &[f32],
        boundaries: &[f32],
        codebook: &[f32],
        rows: usize,
        dim: usize,
    ) -> Vec<f32> {
        let mut norms = vec![0.0_f32; rows];
        for r in 0..rows {
            let row = &input[r * dim..(r + 1) * dim];
            let norm_sq: f32 = row.iter().map(|&v| v * v).sum();
            let norm_val = norm_sq.sqrt();
            let inv_norm = if norm_val > 1.0e-8 { 1.0 / norm_val } else { 0.0 };
            let mut recon_sq = 0.0_f32;
            for &v in row {
                let rotated = v * inv_norm; // identity rotation
                // Index = count of boundaries the value exceeds.
                let mut idx = 0usize;
                for &bnd in boundaries {
                    if rotated > bnd {
                        idx += 1;
                    }
                }
                let centroid = codebook[idx];
                recon_sq += centroid * centroid;
            }
            let recon_norm = recon_sq.sqrt();
            norms[r] = if recon_norm > 1.0e-8 { norm_val / recon_norm } else { norm_val };
        }
        norms
    }

    /// Small AURA-encode shape: dim a multiple of 32, identity rotation. int4.
    fn setup(dim: usize, rows: usize, dt: DType) -> TestSetup {
        const BITS: usize = 4;
        let packed_width = (dim * BITS).div_ceil(32);
        let (codebook, boundaries) = int4_uniform_codebook();
        let rotation = identity_rotation(dim);
        // Smooth input bounded so unit-normed coordinates land mid-bin (away
        // from the 15 midpoint boundaries) — keeps the quant index stable under
        // input dtype rounding.
        let input: Vec<f32> = (0..rows * dim).map(|i| ((i as f32) * 0.013).sin() * 0.6).collect();
        // `input` and `codebook` are dtype-rounded (both `Tensor<T>` now); the
        // oracle then matches the GPU's cast-at-load. `norms_out` is stored
        // through `dt`, so round the expectation through `dt` too.
        let input_r = unpack_f32(&pack_f32(&input, dt), dt);
        let codebook_r = unpack_f32(&pack_f32(&codebook, dt), dt);
        // rotation rounds through dt at cast-at-load (identity is exact in
        // every dtype so it's a no-op here). `boundaries` is Tensor<f32>
        // — no rounding, the GPU loads the same float the oracle reads.
        let expected_norms_f32 = norms_oracle(&input_r, &boundaries, &codebook_r, rows, dim);
        let expected_norms = unpack_f32(&pack_f32(&expected_norms_f32, dt), dt);

        TestSetup::new(aura_encode_int4::kernel_ir_for(dt))
            .mode(KernelMode::Reduction)
            .input(TestBuffer::from_vec("input", pack_f32(&input, dt), dt))
            .input(TestBuffer::from_vec("rotation", pack_f32(&rotation, dt), dt))
            .input(TestBuffer::from_vec(
                "boundaries",
                pack_f32(&boundaries, DType::F32),
                DType::F32,
            ))
            .input(TestBuffer::from_vec("codebook", pack_f32(&codebook, dt), dt))
            .input(TestBuffer::from_vec(
                "packed_out",
                u32_bytes(&vec![0u32; rows * packed_width]),
                DType::U32,
            ))
            .input(TestBuffer::zeros("norms_out", rows, dt))
            .constexpr("dim", dim as u32)
            .constexpr("packed_width", packed_width as u32)
            // Only verify the norm-correction factor (now `T`). `packed_out` is
            // fast-math-sensitive — covered bit-exact by the legacy A/B test.
            .expect(TestBuffer::from_vec("norms_out", pack_f32(&expected_norms, dt), dt))
            .grid_3d(rows as u32, 1, 1, [dim as u32, 1, 1])
    }

    // dim=128 (4 simdgroups — exercises the cross-simdgroup `shared_norm`
    // combine), 2 rows. norms_out is f32; the simd_sum reorder vs the CPU
    // left-fold drifts a few ulp, so even the f32 cell uses a small absolute
    // band; f16/bf16 widen only from input rounding.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-3, 5e-3])]
    fn test_aura_encode_int4_norms(dt: DType) -> TestSetup { setup(128, 2, dt) }

    // dim=32 (exactly one simdgroup — the n_simd=1 path), single row.
    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 1e-3, 5e-3])]
    fn test_aura_encode_int4_norms_min_dim(dt: DType) -> TestSetup { setup(32, 1, dt) }
}

pub mod kernel_benches {
    use metaltile::{bench, test::*};

    use super::{
        aura_encode_int2,
        aura_encode_int3,
        aura_encode_int4,
        aura_encode_int6,
        aura_encode_int8,
    };

    fn setup(s: BenchSetup, dim: usize, bits: usize, rows: usize, dt: DType) -> BenchSetup {
        let packed_width = (dim * bits).div_ceil(32);
        let levels = 1usize << bits;
        s.mode(KernelMode::Reduction)
            .buffer(BenchBuffer::random("input", rows * dim, dt))
            .buffer(BenchBuffer::random("rotation", dim * dim, dt))
            .buffer(BenchBuffer::random("boundaries", levels - 1, dt))
            .buffer(BenchBuffer::random("codebook", levels, dt))
            .buffer(BenchBuffer::zeros("packed_out", rows * packed_width, DType::U32).output())
            .buffer(BenchBuffer::zeros("norms_out", rows, dt).output())
            .constexpr("dim", dim as u32)
            .constexpr("packed_width", packed_width as u32)
            // Rotation matmul dominates: rows reads of a dim×dim T matrix.
            .bytes_moved((rows * dim * dim * dt.size_bytes()) as u64)
            .grid_3d(rows as u32, 1, 1, [dim as u32, 1, 1])
    }

    #[bench(name = "ffai/aura_encode_int2", dtypes = [f32, f16, bf16])]
    fn bench_int2(dt: DType) -> BenchSetup {
        setup(BenchSetup::new(aura_encode_int2::kernel_ir_for(dt)), 128, 2, 256, dt)
    }

    #[bench(name = "ffai/aura_encode_int3", dtypes = [f32, f16, bf16])]
    fn bench_int3(dt: DType) -> BenchSetup {
        setup(BenchSetup::new(aura_encode_int3::kernel_ir_for(dt)), 128, 3, 256, dt)
    }

    #[bench(name = "ffai/aura_encode_int4", dtypes = [f32, f16, bf16])]
    fn bench_int4(dt: DType) -> BenchSetup {
        setup(BenchSetup::new(aura_encode_int4::kernel_ir_for(dt)), 128, 4, 256, dt)
    }

    #[bench(name = "ffai/aura_encode_int6", dtypes = [f32, f16, bf16])]
    fn bench_int6(dt: DType) -> BenchSetup {
        setup(BenchSetup::new(aura_encode_int6::kernel_ir_for(dt)), 128, 6, 256, dt)
    }

    #[bench(name = "ffai/aura_encode_int8", dtypes = [f32, f16, bf16])]
    fn bench_int8(dt: DType) -> BenchSetup {
        setup(BenchSetup::new(aura_encode_int8::kernel_ir_for(dt)), 128, 8, 256, dt)
    }
}
