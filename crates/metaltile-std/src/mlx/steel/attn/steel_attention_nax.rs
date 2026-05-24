//! `mt_sdpa_prefill_nax` — flash attention via `mpp::tensor_ops::matmul2d`.
//!
//! NAX (Apple tensor-core) port of the steel flash-attention prefill
//! kernel. Requires Metal 4 / macOS 26+ and Apple10+ hardware (M4 family
//! or newer); runtime-gated via `Context::chip_family()`.
//!
//! Expressed in the `#[kernel]` DSL via the `coop_tile_*` intrinsics —
//! no `Op::InlineMsl`. The cooperative-tensor counterpart of
//! `mt_sdpa_prefill_mma`: the standard FlashAttention-2 online-softmax
//! loop, but the two matmuls — `S = Q·Kᵀ` and `O += P·V` — are each one
//! cooperative `matmul2d` instead of an 8×8 `simdgroup_matmul` ladder.
//!
//! ## Tile geometry (all variants)
//!
//! - **BQ = 16** queries/TG, **BK = 16** keys/block.
//! - **BD = 32** — the inner-contraction chunk used by every `matmul2d`
//!   call. Apple's "at least one of M/N/K = 32" rule requires K=32 for the
//!   QK contraction descriptor. When `head_dim > 32` the QK (and PV)
//!   contractions loop over `n_chunks = head_dim / 32` consecutive 32-wide
//!   D-slices, accumulating partial S scores (fp32) before the online-
//!   softmax step. This is the **D-chunk loop** that unlocks d={64,128,256}.
//! - **tpg = 32** (one simdgroup). The 16×16 S tile and 16×32 O tile are
//!   each one cooperative `matmul2d` per D-chunk.
//! - Grid: `[q_len/16, n_q_heads, batch]` — `tgid_x` Q-tile, `tgid_y`
//!   Q-head, `tgid_z` batch.
//!
//! ## D-chunk loop design
//!
//! The outer K-block loop (FlashAttention-2 online softmax) is unchanged.
//! Inside each K-block step we:
//!   1. QK contraction: for each D-chunk `dc` in `0..n_chunks`, load the
//!      16×32 Q and K slices; compute `dS += Q_dc · K_dcᵀ` via
//!      `matmul2d(16,16,32)`. Chunk 0 uses `overwrite` mode; chunks 1..N
//!      use `accumulate` mode so partial sums remain correct.
//!   2. Apply causal mask + online-softmax to the fully-accumulated S tile.
//!   3. PV contraction: for each D-chunk `dc` in `0..n_chunks`, load the
//!      16×32 V slice; compute `dO_blk += P · V_dc` via `matmul2d(16,32,16)`
//!      using the same overwrite-then-accumulate strategy.
//!   4. Add `dO_blk` into the running Os accumulator with the rescale factor.
//!
//! The S-tile is 16×16 (independent of head_dim); the O accumulator is
//! 16×head_dim. Only the 16×32 Qs/Ks/Vs scratch tiles are fixed at BD=32.
//! The Os/Obk scratch grows with head_dim (16 × head_dim, skewed).
//!
//! ## Dispatch invariants (per variant)
//!
//! - TPG 32 (1 SG); grid `[q_len/16, n_q_heads, batch]`.
//! - `q_len % 16 == 0`, `k_len % 16 == 0`, `head_dim ∈ {32, 64, 128, 256}`.
//! - `KernelMode::Reduction`.
//!
//! Correctness vs CPU oracle ≥ cos 0.999 (f32/f16), ≥ 0.997 (bf16) — see
//! `crates/metaltile-std/tests/steel_attention_nax_gpu_correctness.rs`.

use metaltile::kernel;

/// Tile geometry — keep in lock-step with the codegen-emitted MSL.
pub const BQ: u32 = 16;
pub const BK: u32 = 16;
/// Inner contraction width; fixed by Apple's matmul2d "one of M/N/K=32" rule.
pub const BD: u32 = 32;
/// Threads per group (1 SG × 32 lanes).
pub const TPG: u32 = 32;
/// Row skew past the inner extent — scatters 32-bank conflicts on the
/// column-strided frag loads inside `matmul2d`.
pub const TG_SKEW: u32 = 4;
/// Leading dim of the BQ/BK × BD tiles (BD + skew).
pub const TG_LD_D: u32 = BD + TG_SKEW; // 36
/// Leading dim of the BQ × BK S/P scratch (BK + skew).
pub const TG_LD_K: u32 = BK + TG_SKEW; // 20

/// Flash-attention prefill via cooperative `matmul2d`, head_dim=32.
///
/// Legacy variant — head_dim is fixed at 32 (no D-chunk loop). Kept for
/// back-compat with callers that call into this module directly. New code
/// should prefer `mt_sdpa_prefill_nax_d64` / `_d128` / `_d256` for
/// production head dims.
///
/// Params: `q`/`k`/`v`/`out` are `[batch, heads, len, head_dim]` slabs
/// (`q`/`out` use `n_q_heads`, `k`/`v` use `n_kv_heads`). Constexprs:
/// `q_len`, `k_len`, `gqa_factor`, `n_q_heads`, `n_kv_heads`, `scale`.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn mt_sdpa_prefill_nax<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    mut out: Tensor<T>,
    #[constexpr] q_len: u32,
    #[constexpr] k_len: u32,
    #[constexpr] gqa_factor: u32,
    #[constexpr] n_q_heads: u32,
    #[constexpr] n_kv_heads: u32,
    #[constexpr] scale: f32,
) {
    let q_tile = tgid_x;
    let q_head = tgid_y;
    let batch = tgid_z;
    let kv_head = q_head / gqa_factor;
    let lane = simd_lane;

    let head_dim = 32u32;
    let q_len_off = k_len - q_len;

    // Slab offsets — q/out: [batch, n_q_heads, q_len, D]; k/v: [batch,
    // n_kv_heads, k_len, D].
    let kv_row_base = batch * n_kv_heads * k_len * head_dim + kv_head * k_len * head_dim;
    let q_head_row_off = batch * n_q_heads * q_len * head_dim + q_head * q_len * head_dim;
    let q_tile_first = q_tile * 16u32;

    // Threadgroup tiles (skewed). Qs/Ks/Vs/Ps are T; Ss/Os/Obk are fp32.
    threadgroup_alloc("Qs", 576, coop_stage(T)); // 16 × 36
    threadgroup_alloc("Ks", 576, coop_stage(T));
    threadgroup_alloc("Vs", 576, coop_stage(T));
    threadgroup_alloc("Ps", 320, coop_stage(T)); // 16 × 20
    threadgroup_alloc("Ss", 320, f32);
    threadgroup_alloc("Os", 576, f32);
    threadgroup_alloc("Obk", 576, f32);

    // Coop-load the 16×32 Q tile (lane fills column `lane`); zero Os.
    for _r in range(0u32, 16u32, 1u32) {
        let q_dev = q_head_row_off + (q_tile_first + _r) * head_dim + lane;
        let qv = load(q[q_dev]).cast::<f32>() * scale;
        threadgroup_store("Qs", _r * 36u32 + lane, qv);
        threadgroup_store("Os", _r * 36u32 + lane, 0.0f32);
    }

    // Per-row online-softmax state — lane `r` (r < 16) owns row r.
    let mut row_m = -1.0e30f32;
    let mut row_s = 0.0f32;
    let owns_row = lane < 16u32;
    let my_row = lane;
    let q_abs = q_tile_first + my_row + q_len_off;

    // QK: matmul2d(16, 16, 32), ta=false tb=true, overwrite (S fresh).
    coop_tile_setup(
        "qk",
        16,
        16,
        32,
        coop_stage(T),
        "overwrite",
        "simdgroup",
        f32,
        false,
        true,
        false,
    );
    // PV: matmul2d(16, 32, 16), ta=false tb=false, overwrite (per-block).
    coop_tile_setup(
        "pv",
        16,
        32,
        16,
        coop_stage(T),
        "overwrite",
        "simdgroup",
        f32,
        false,
        false,
        false,
    );

    // Causal trim — last K-block touched by the tile's last query.
    let q_tile_last_abs = q_tile_first + 15u32 + q_len_off;
    let kb_lim = q_tile_last_abs / 16u32 + 1u32;

    for kb in range(0u32, kb_lim, 1u32) {
        let kb_off = kb * 16u32;

        // 1. Coop-load the 16×32 K and V tiles.
        for _r in range(0u32, 16u32, 1u32) {
            let kv_dev = kv_row_base + (kb_off + _r) * head_dim + lane;
            threadgroup_store("Ks", _r * 36u32 + lane, load(k[kv_dev]).cast::<f32>());
            threadgroup_store("Vs", _r * 36u32 + lane, load(v[kv_dev]).cast::<f32>());
        }
        threadgroup_barrier();

        // 2. S = Q·Kᵀ — extents inner-first: tQ/tK [TG_LD_D=36, 16],
        //    tS [TG_LD_K=20, 16].
        coop_tile_load_a("qk", "Qs", true, coop_stage(T), 36, 16);
        coop_tile_load_b("qk", "Ks", true, coop_stage(T), 36, 16);
        coop_tile_run("qk");
        coop_tile_store_c("qk", "Ss", true, f32, 20, 16);
        threadgroup_barrier();

        // 3. Online softmax — each owning lane processes its S row.
        if owns_row {
            let mut blk_m = -1.0e30f32;
            for _c in range(0u32, 16u32, 1u32) {
                let k_abs = kb_off + _c;
                let raw = threadgroup_load("Ss", my_row * 20u32 + _c);
                let sc = select(k_abs > q_abs, -1.0e30f32, raw);
                threadgroup_store("Ss", my_row * 20u32 + _c, sc);
                blk_m = select(sc > blk_m, sc, blk_m);
            }
            let new_m = select(blk_m > row_m, blk_m, row_m);
            let rescale = exp(row_m - new_m);
            let mut blk_s = 0.0f32;
            for _c in range(0u32, 16u32, 1u32) {
                let p = exp(threadgroup_load("Ss", my_row * 20u32 + _c) - new_m);
                threadgroup_store("Ss", my_row * 20u32 + _c, p);
                blk_s = blk_s + p;
            }
            row_s = row_s * rescale + blk_s;
            // Rescale the running O accumulator by exp(m_old - m_new).
            for _d in range(0u32, 32u32, 1u32) {
                let o = threadgroup_load("Os", my_row * 36u32 + _d);
                threadgroup_store("Os", my_row * 36u32 + _d, o * rescale);
            }
            row_m = new_m;
        }
        threadgroup_barrier();

        // 4. Stage the fp32 exp-weights P into the T-typed Ps tile.
        if owns_row {
            for _c in range(0u32, 16u32, 1u32) {
                let p = threadgroup_load("Ss", my_row * 20u32 + _c);
                threadgroup_store("Ps", my_row * 20u32 + _c, p);
            }
        }
        threadgroup_barrier();

        // O_blk = P·V — tP [TG_LD_K=20, 16], tV [TG_LD_D=36, 16],
        // tObk [TG_LD_D=36, 16].
        coop_tile_load_a("pv", "Ps", true, coop_stage(T), 20, 16);
        coop_tile_load_b("pv", "Vs", true, coop_stage(T), 36, 16);
        coop_tile_run("pv");
        coop_tile_store_c("pv", "Obk", true, f32, 36, 16);
        threadgroup_barrier();

        // Add the per-block P·V product into the running Os accumulator.
        if owns_row {
            for _d in range(0u32, 32u32, 1u32) {
                let o = threadgroup_load("Os", my_row * 36u32 + _d);
                let ob = threadgroup_load("Obk", my_row * 36u32 + _d);
                threadgroup_store("Os", my_row * 36u32 + _d, o + ob);
            }
        }
        threadgroup_barrier();
    }

    // 5. Normalize by the softmax denominator and store O.
    if owns_row {
        let inv_s = select(row_s > 0.0f32, 1.0f32 / row_s, 0.0f32);
        for _d in range(0u32, 32u32, 1u32) {
            let o_dev = q_head_row_off + (q_tile_first + my_row) * head_dim + _d;
            let o = threadgroup_load("Os", my_row * 36u32 + _d);
            store(out[o_dev], (o * inv_s).cast::<T>());
        }
    }
}

// ── D-chunk macro: generates mt_sdpa_prefill_nax_d{64,128,256} ────────────
//
// The macro expands to a full kernel body that loops the QK and PV
// contractions over `n_chunks = $head_dim / 32` consecutive 32-wide D-slices.
//
// TG memory layout (all variants):
//   Qs/Ks/Vs  : 16 × 36  (16 rows × (BD=32 + TG_SKEW=4))  — one D-chunk at a time
//   Ps        : 16 × 20  (16 rows × (BK=16 + TG_SKEW=4))
//   Ss        : 16 × 20  (fp32 attention scores, reused per K-block)
//   Os        : 16 × ($head_dim + TG_SKEW)   fp32 running accumulator
//   Obk       : 16 × 36                       fp32 per-D-chunk PV scratch
//
// The QK D-chunk loop uses `overwrite` for the first chunk and
// `accumulate` for subsequent chunks so partial scores add up correctly.
// The PV D-chunk loop overwrites Obk per chunk and immediately adds the
// chunk into the matching 32-wide column band of Os — this keeps Obk at
// one chunk's worth (576 fp32) instead of full-width, which is what lets
// d=256 fit under Metal's 32 KB TG-memory cap.
macro_rules! sdpa_prefill_nax_wide {
    ($name:ident, $head_dim:literal, $n_chunks:literal, $os_slots:literal) => {
        #[doc = concat!(
            "Flash-attention prefill via cooperative `matmul2d`, head_dim=",
            stringify!($head_dim),
            ".\n\n",
            "Same algorithm as `mt_sdpa_prefill_nax` (d=32) but loops the QK and PV\n",
            "contractions over `", stringify!($n_chunks), "` consecutive 32-wide D-chunks.\n",
            "The S-tile (16×16) and online-softmax state are unchanged; only the\n",
            "head_dim axis of the Q/K/V loads and the Os accumulator scale up.\n\n",
            "See module-level docs for the D-chunk loop design.\n",
            "Generic `T ∈ {f32, f16, bf16}`; `coop_stage(T)` for bf16 safety.\n",
            "Runtime-gated to Apple10+ — needs macOS 26+ / Metal 4.",
        )]
        #[kernel]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<T>(
            q: Tensor<T>,
            k: Tensor<T>,
            v: Tensor<T>,
            mut out: Tensor<T>,
            #[constexpr] q_len: u32,
            #[constexpr] k_len: u32,
            #[constexpr] gqa_factor: u32,
            #[constexpr] n_q_heads: u32,
            #[constexpr] n_kv_heads: u32,
            #[constexpr] scale: f32,
        ) {
            let q_tile = tgid_x;
            let q_head = tgid_y;
            let batch = tgid_z;
            let kv_head = q_head / gqa_factor;
            let lane = simd_lane;

            let head_dim = $head_dim;
            let n_chunks = $n_chunks;
            let q_len_off = k_len - q_len;

            // Slab offsets — q/out: [batch, n_q_heads, q_len, D]; k/v: [batch, n_kv_heads, k_len, D].
            let kv_row_base = batch * n_kv_heads * k_len * head_dim + kv_head * k_len * head_dim;
            let q_head_row_off = batch * n_q_heads * q_len * head_dim + q_head * q_len * head_dim;
            let q_tile_first = q_tile * 16u32;

            // TG memory:
            //   Qs/Ks/Vs : 16 × 36 — one 32-wide D-chunk at a time (BD + TG_SKEW=4)
            //   Ps       : 16 × 20 — attention weights (BK=16 + TG_SKEW=4)
            //   Ss       : 16 × 20 — accumulated scores per K-block (fp32)
            //   Os       : 16 × ($os_slots) — running output accumulator (fp32)
            //   Obk      : 16 × 36 — per-D-chunk P·V scratch (fp32, reused across chunks)
            //
            // For d=256 the running Os already costs 16640 bytes; keeping
            // a second full-width Obk would push past Metal's 32 KB TG
            // memory cap. Instead Obk is sized for one 32-wide D-chunk
            // (576 fp32 slots) and we accumulate each PV chunk into the
            // matching slice of Os directly — eliminating the old step 5.
            threadgroup_alloc("Qs", 576, coop_stage(T)); // 16 × 36
            threadgroup_alloc("Ks", 576, coop_stage(T));
            threadgroup_alloc("Vs", 576, coop_stage(T));
            threadgroup_alloc("Ps", 320, coop_stage(T)); // 16 × 20
            threadgroup_alloc("Ss", 320, f32);           // 16 × 20
            threadgroup_alloc("Os", $os_slots, f32);
            threadgroup_alloc("Obk", 576, f32);          // 16 × 36 — one D-chunk

            // TG_LD_O = head_dim + TG_SKEW (for Os row stride).
            let tg_ld_o = head_dim + 4u32;

            // Zero the full Os accumulator (16 × head_dim fp32 cells).
            // Stripe across the 32 lanes by D-chunk: each lane writes its
            // column index across all 16 rows × all n_chunks chunks. The
            // previous combined Q-load + Os-zero loop only zeroed the
            // dc=0 D-chunk (columns 0..32); columns 32..head_dim then
            // carried whatever stale TG-memory landed there, which on
            // single-tile inputs combined with the first K-block's
            // rescale `exp(-1e30 - finite) ≈ 0` to produce 0 × ±∞ = NaN.
            // The K-block loop reloads Qs every iteration so we don't
            // need to prime it here.
            for dc in range(0u32, n_chunks, 1u32) {
                let d_off = dc * 32u32;
                for _r in range(0u32, 16u32, 1u32) {
                    threadgroup_store("Os", _r * tg_ld_o + d_off + lane, 0.0f32);
                }
            }
            threadgroup_barrier();

            // Per-row online-softmax state.
            let mut row_m = -1.0e30f32;
            let mut row_s = 0.0f32;
            let owns_row = lane < 16u32;
            let my_row = lane;
            let q_abs = q_tile_first + my_row + q_len_off;

            // QK setup: matmul2d(16,16,32), tb=true (K transposed).
            // We use overwrite mode — accumulation across D-chunks is done
            // by re-running with accumulate and the same Ss output pointer.
            coop_tile_setup(
                "qk",
                16,
                16,
                32,
                coop_stage(T),
                "overwrite",
                "simdgroup",
                f32,
                false,
                true,
                false,
            );
            // QK accumulate variant for D-chunk 1+.
            coop_tile_setup(
                "qk_acc",
                16,
                16,
                32,
                coop_stage(T),
                "accumulate",
                "simdgroup",
                f32,
                false,
                true,
                false,
            );
            // PV setup: matmul2d(16,32,16), overwrite (per-block; we sum manually).
            coop_tile_setup(
                "pv",
                16,
                32,
                16,
                coop_stage(T),
                "overwrite",
                "simdgroup",
                f32,
                false,
                false,
                false,
            );

            // Causal trim — last K-block touched by the tile's last query.
            let q_tile_last_abs = q_tile_first + 15u32 + q_len_off;
            let kb_lim = q_tile_last_abs / 16u32 + 1u32;

            for kb in range(0u32, kb_lim, 1u32) {
                let kb_off = kb * 16u32;

                // ── Step 1: accumulate S = Σ_{dc} Q_dc · K_dcᵀ ─────────────
                // For each D-chunk, load the Q slice (already computed — we re-
                // load from device since we can't cache all n_chunks in TG at once)
                // and the K slice, then run matmul2d.
                //
                // D-chunk 0: overwrite Ss (fresh).
                // D-chunk 1+: accumulate into Ss.
                for dc in range(0u32, n_chunks, 1u32) {
                    let d_off = dc * 32u32;
                    // Load Q and K slices for this D-chunk.
                    for _r in range(0u32, 16u32, 1u32) {
                        let q_dev =
                            q_head_row_off + (q_tile_first + _r) * head_dim + d_off + lane;
                        let kv_dev = kv_row_base + (kb_off + _r) * head_dim + d_off + lane;
                        threadgroup_store("Qs", _r * 36u32 + lane, load(q[q_dev]).cast::<f32>() * scale);
                        threadgroup_store("Ks", _r * 36u32 + lane, load(k[kv_dev]).cast::<f32>());
                    }
                    threadgroup_barrier();

                    if dc == 0u32 {
                        // First chunk: overwrite Ss (qk descriptor).
                        coop_tile_load_a("qk", "Qs", true, coop_stage(T), 36, 16);
                        coop_tile_load_b("qk", "Ks", true, coop_stage(T), 36, 16);
                        coop_tile_run("qk");
                        coop_tile_store_c("qk", "Ss", true, f32, 20, 16);
                    } else {
                        // Subsequent chunks: accumulate into Ss.
                        coop_tile_load_a("qk_acc", "Qs", true, coop_stage(T), 36, 16);
                        coop_tile_load_b("qk_acc", "Ks", true, coop_stage(T), 36, 16);
                        coop_tile_run("qk_acc");
                        coop_tile_store_c("qk_acc", "Ss", true, f32, 20, 16);
                    }
                    threadgroup_barrier();
                }

                // ── Step 2: online softmax ───────────────────────────────────
                if owns_row {
                    let mut blk_m = -1.0e30f32;
                    for _c in range(0u32, 16u32, 1u32) {
                        let k_abs = kb_off + _c;
                        let raw = threadgroup_load("Ss", my_row * 20u32 + _c);
                        let sc = select(k_abs > q_abs, -1.0e30f32, raw);
                        threadgroup_store("Ss", my_row * 20u32 + _c, sc);
                        blk_m = select(sc > blk_m, sc, blk_m);
                    }
                    let new_m = select(blk_m > row_m, blk_m, row_m);
                    let rescale = exp(row_m - new_m);
                    let mut blk_s = 0.0f32;
                    for _c in range(0u32, 16u32, 1u32) {
                        let p = exp(threadgroup_load("Ss", my_row * 20u32 + _c) - new_m);
                        threadgroup_store("Ss", my_row * 20u32 + _c, p);
                        blk_s = blk_s + p;
                    }
                    row_s = row_s * rescale + blk_s;
                    // Rescale the running O accumulator row.
                    for _d in range(0u32, head_dim, 1u32) {
                        let o = threadgroup_load("Os", my_row * tg_ld_o + _d);
                        threadgroup_store("Os", my_row * tg_ld_o + _d, o * rescale);
                    }
                    row_m = new_m;
                }
                threadgroup_barrier();

                // ── Step 3: stage P (exp-weights) ───────────────────────────
                if owns_row {
                    for _c in range(0u32, 16u32, 1u32) {
                        let p = threadgroup_load("Ss", my_row * 20u32 + _c);
                        threadgroup_store("Ps", my_row * 20u32 + _c, p);
                    }
                }
                threadgroup_barrier();

                // ── Step 4: Os += P · V, accumulating per D-chunk ───────────
                // Step 2 already rescaled the running Os by `exp(row_m - new_m)`
                // across all head_dim, so all that's left here is to add the
                // PV contribution. We process one 32-wide D-chunk at a time:
                // load V chunk into Vs, run PV into Obk (16 × 36 scratch),
                // then add the chunk into Os[row][d_off + 0..32]. This
                // collapses the previous two-step (full-width Obk then
                // Os += Obk) into a single per-chunk accumulation and frees
                // 16 × (head_dim + 4 - 36) fp32 cells of TG memory — the
                // savings is what gets d=256 under Metal's 32 KB cap.
                for dc in range(0u32, n_chunks, 1u32) {
                    let d_off = dc * 32u32;
                    // Load V slice for this D-chunk into Vs.
                    for _r in range(0u32, 16u32, 1u32) {
                        let kv_dev = kv_row_base + (kb_off + _r) * head_dim + d_off + lane;
                        threadgroup_store("Vs", _r * 36u32 + lane, load(v[kv_dev]).cast::<f32>());
                    }
                    threadgroup_barrier();

                    // PV produces a 16×32 block of the output for this D-chunk.
                    // Store directly into Obk (the per-chunk scratch).
                    coop_tile_load_a("pv", "Ps", true, coop_stage(T), 20, 16);
                    coop_tile_load_b("pv", "Vs", true, coop_stage(T), 36, 16);
                    coop_tile_run("pv");
                    coop_tile_store_c("pv", "Obk", true, f32, 36, 16);
                    threadgroup_barrier();

                    // Add this 32-wide slice into the matching Os column band.
                    if owns_row {
                        for _d in range(0u32, 32u32, 1u32) {
                            let ob = threadgroup_load("Obk", my_row * 36u32 + _d);
                            let o = threadgroup_load("Os", my_row * tg_ld_o + d_off + _d);
                            threadgroup_store("Os", my_row * tg_ld_o + d_off + _d, o + ob);
                        }
                    }
                    threadgroup_barrier();
                }
            }

            // ── Step 6: normalize and store output ───────────────────────────
            if owns_row {
                let inv_s = select(row_s > 0.0f32, 1.0f32 / row_s, 0.0f32);
                for _d in range(0u32, head_dim, 1u32) {
                    let o_dev = q_head_row_off + (q_tile_first + my_row) * head_dim + _d;
                    let o = threadgroup_load("Os", my_row * tg_ld_o + _d);
                    store(out[o_dev], (o * inv_s).cast::<T>());
                }
            }
        }
    };
}

// Os/Obk slot counts = 16 rows × (head_dim + TG_SKEW=4).
// d64:  16 × 68  = 1088
// d128: 16 × 132 = 2112
// d256: 16 × 260 = 4160
sdpa_prefill_nax_wide!(mt_sdpa_prefill_nax_d64, 64u32, 2u32, 1088);
sdpa_prefill_nax_wide!(mt_sdpa_prefill_nax_d128, 128u32, 4u32, 2112);
sdpa_prefill_nax_wide!(mt_sdpa_prefill_nax_d256, 256u32, 8u32, 4160);

#[cfg(test)]
mod tests {
    use metaltile_core::{dtype::DType, ir::Op};

    use super::*;

    #[test]
    fn kernel_ir_constructs_and_uses_coop_tile_ops() {
        for dt in [DType::F32, DType::F16, DType::BF16] {
            let k = mt_sdpa_prefill_nax::kernel_ir_for(dt);
            assert_eq!(k.name, "mt_sdpa_prefill_nax");
            assert_eq!(k.params.len(), 4);
            assert!(k.params[3].is_output);
            assert_eq!(k.constexprs.len(), 6);
            let all_ops =
                || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
            // No raw inline MSL — both matmuls are CoopTile* ops.
            assert!(!all_ops().any(|op| matches!(op, Op::InlineMsl { .. })));
            // Two distinct cooperative-matmul setups (qk + pv).
            let n_setup = all_ops().filter(|op| matches!(op, Op::CoopTileSetup { .. })).count();
            assert_eq!(n_setup, 2, "expected qk + pv CoopTileSetup ops");
        }
    }

    #[test]
    fn wide_kernels_ir_constructs_and_uses_coop_tile_ops() {
        // d64/d128/d256 kernels have three setups: qk (overwrite), qk_acc
        // (accumulate), pv (overwrite).
        for dt in [DType::F32, DType::F16, DType::BF16] {
            for (name, kernel_ir) in [
                ("mt_sdpa_prefill_nax_d64", mt_sdpa_prefill_nax_d64::kernel_ir_for as fn(_) -> _),
                ("mt_sdpa_prefill_nax_d128", mt_sdpa_prefill_nax_d128::kernel_ir_for),
                ("mt_sdpa_prefill_nax_d256", mt_sdpa_prefill_nax_d256::kernel_ir_for),
            ] {
                let k = kernel_ir(dt);
                assert_eq!(k.name, name);
                assert_eq!(k.params.len(), 4, "{name}: 4 tensor params");
                assert!(k.params[3].is_output, "{name}: last param is output");
                assert_eq!(k.constexprs.len(), 6, "{name}: 6 constexprs");
                let all_ops =
                    || std::iter::once(&k.body).chain(k.blocks.values()).flat_map(|b| b.ops.iter());
                assert!(
                    !all_ops().any(|op| matches!(op, Op::InlineMsl { .. })),
                    "{name}: no InlineMsl ops"
                );
                let n_setup = all_ops().filter(|op| matches!(op, Op::CoopTileSetup { .. })).count();
                assert_eq!(n_setup, 3, "{name}: expected qk + qk_acc + pv CoopTileSetup ops");
            }
        }
    }

    #[test]
    fn codegen_emits_mpp_include_and_kernel_decl() {
        use metaltile_codegen::msl::MslGenerator;
        for (dt, t_name) in [(DType::F32, "float"), (DType::F16, "half"), (DType::BF16, "half")] {
            let mut k = mt_sdpa_prefill_nax::kernel_ir_for(dt);
            let suffix = match dt {
                DType::F32 => "f32",
                DType::F16 => "f16",
                _ => "bf16",
            };
            k.name = format!("mt_sdpa_prefill_nax_{suffix}");
            let msl = MslGenerator::default().generate(&k).expect("codegen");
            assert!(msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"));
            assert!(msl.contains("mpp::tensor_ops::matmul2d_descriptor"));
            assert!(msl.contains(&format!("kernel void mt_sdpa_prefill_nax_{suffix}")));
            assert!(msl.contains(&format!("threadgroup {t_name}")));
        }
    }

    #[test]
    fn wide_codegen_emits_mpp_include_and_kernel_decl() {
        use metaltile_codegen::msl::MslGenerator;
        for (dt, t_name) in [(DType::F32, "float"), (DType::F16, "half"), (DType::BF16, "half")] {
            for (dim, kernel_ir) in [
                (64usize, mt_sdpa_prefill_nax_d64::kernel_ir_for as fn(_) -> _),
                (128, mt_sdpa_prefill_nax_d128::kernel_ir_for),
                (256, mt_sdpa_prefill_nax_d256::kernel_ir_for),
            ] {
                let suffix = match dt {
                    DType::F32 => "f32",
                    DType::F16 => "f16",
                    _ => "bf16",
                };
                let mut k = kernel_ir(dt);
                k.name = format!("mt_sdpa_prefill_nax_d{dim}_{suffix}");
                let msl = MslGenerator::default().generate(&k).expect("codegen");
                assert!(
                    msl.contains("MetalPerformancePrimitives/MetalPerformancePrimitives.h"),
                    "d{dim} {suffix}: missing MPP include"
                );
                assert!(
                    msl.contains("mpp::tensor_ops::matmul2d_descriptor"),
                    "d{dim} {suffix}: missing matmul2d_descriptor"
                );
                assert!(
                    msl.contains(&format!("kernel void mt_sdpa_prefill_nax_d{dim}_{suffix}")),
                    "d{dim} {suffix}: missing kernel declaration"
                );
                assert!(
                    msl.contains(&format!("threadgroup {t_name}")),
                    "d{dim} {suffix}: missing threadgroup type"
                );
            }
        }
    }
}
