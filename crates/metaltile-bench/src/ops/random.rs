//! Random number generation benchmarks — metal/random.metal  (MLX, Apache-2.0)
//!
//! Generates random bits using the Threefry2×32 counter-based PRNG.
//!
//! MLX reference: `rbitsc`  (contiguous key layout)
//!   Each thread handles one 64-bit chunk: 4 bytes for index.y and 4 for index.y + half_size.
//!   Dispatch: [num_keys, 1, 1] × [1, half_size, 1]
//!   where half_size = ceil(bytes_per_key / 8)
//!
//! MetalTile: `mt_random_hash` — per-element xorshift32 hash PRNG.
//!   Each thread writes one u32 value seeded by its global index.
//!   Grid: [N/TPG, 1, 1] × [TPG, 1, 1]
//!   KernelMode::Elementwise

use metaltile::kernel;
use metaltile_codegen::msl::MslGenerator;

use crate::{
    ops::{EquivResult, OpBench, OpResult, to_gbps},
    runner::GpuRunner,
};

static SRC: &str = include_str!("../metal/random.metal");

const REF_NAME: &str = "rbitsc";
const NUM_KEYS: usize = 1024;
const BYTES_PER_KEY: usize = 4096;
const HALF_SIZE: usize = BYTES_PER_KEY / 8;
const TOTAL_FLOATS: usize = NUM_KEYS * BYTES_PER_KEY / 4;
const TPG: usize = 1024;

const BENCH: OpBench = OpBench::new("random_f32", "GB/s");

// ── DSL kernel ───────────────────────────────────────────────────────────────

/// Per-element xorshift32 hash PRNG.
///
/// Seeds each output element with its global thread index (gid + 1 to avoid
/// the all-zeros fixed point). Three xorshift rounds give reasonable mixing.
/// Dispatch: [N/TPG, 1, 1] × [TPG, 1, 1]  (each thread writes one u32).
#[kernel]
pub fn mt_random_hash(out: Tensor<u32>, #[constexpr] n: u32) {
    let gid = program_id::<0>();
    let mut s = gid + 1u32;
    s = s ^ (s << 13u32);
    s = s ^ (s >> 17u32);
    s = s ^ (s << 5u32);
    store(out[gid], s);
}

fn random_msl() -> Result<String, String> {
    MslGenerator::default()
        .generate(&mt_random_hash::kernel_ir())
        .map_err(|e| format!("random codegen: {e}"))
        .and_then(|msl| if msl.trim().is_empty() { Err("empty".into()) } else { Ok(msl) })
}

// ── CPU reference (same xorshift32 algorithm) ────────────────────────────────

fn cpu_xorshift(n: usize) -> Vec<u32> {
    (0..n as u32)
        .map(|gid| {
            let mut s = gid + 1;
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        })
        .collect()
}

// ── Bench ─────────────────────────────────────────────────────────────────────

pub fn bench_random(runner: &GpuRunner) -> Vec<OpResult> {
    let rk = runner.compile(SRC, REF_NAME).ok();

    let key_data: Vec<u8> = (0..NUM_KEYS * 2 * 4).map(|i| i as u8).collect();
    let keys_buf = runner.buffer_bytes(&key_data);
    let ref_out_buf = runner.buffer_zeros(NUM_KEYS * BYTES_PER_KEY);
    let odd_buf = runner.buffer_bytes(std::slice::from_ref(&(false as u8)));
    let bpk_buf = runner.buffer_bytes(&(BYTES_PER_KEY as u32).to_le_bytes());

    let bytes = (TOTAL_FLOATS * 4) as f64;

    let ref_perf = rk.as_ref().and_then(|rk| {
        let st = runner.bench(
            rk,
            &[&keys_buf, &ref_out_buf, &odd_buf, &bpk_buf],
            [NUM_KEYS, 1, 1],
            [1, HALF_SIZE, 1],
            3,
            10,
        );
        to_gbps(&st, bytes)
    });

    let mt_msl = random_msl().ok();
    let mk = mt_msl.as_deref().and_then(|msl| runner.compile(msl, "mt_random_hash").ok());

    // Correctness: compare a small batch against CPU reference.
    let equiv: Option<EquivResult> = mk.as_ref().map(|mk| {
        let check_n: usize = 1024;
        let ref_vals = cpu_xorshift(check_n);
        let n_buf = runner.buffer_u32(check_n as u32);
        let check_out = runner.buffer_zeros(check_n * 4);
        runner.measure(mk, &[&check_out, &n_buf], [check_n.div_ceil(TPG), 1, 1], [TPG, 1, 1], 0, 1);
        // Read as f32 bytes, reinterpret bit patterns as u32 for comparison.
        let raw = runner.read_f32_slice(&check_out, check_n);
        let mt_vals: Vec<u32> = raw.iter().map(|f| f.to_bits()).collect();
        let n_bad = ref_vals.iter().zip(&mt_vals).filter(|(a, b)| a != b).count();
        EquivResult {
            n_checked: check_n,
            max_abs_err: if n_bad == 0 { 0.0 } else { f32::INFINITY },
            cosine_sim: if n_bad == 0 { 1.0 } else { 0.0 },
            passed: n_bad == 0,
        }
    });

    let mt_out_buf = runner.buffer_zeros(TOTAL_FLOATS * 4);
    let n_buf = runner.buffer_u32(TOTAL_FLOATS as u32);
    let mt_perf = mk.as_ref().and_then(|mk| {
        let st = runner.bench(
            mk,
            &[&mt_out_buf, &n_buf],
            [TOTAL_FLOATS.div_ceil(TPG), 1, 1],
            [TPG, 1, 1],
            3,
            10,
        );
        to_gbps(&st, bytes)
    });

    let shape = format!("{}M f32", TOTAL_FLOATS / (1024 * 1024));
    let result = if let Some(mt_perf) = mt_perf {
        BENCH.implemented(shape, ref_perf, mt_perf, equiv.unwrap())
    } else {
        BENCH.nyi(shape, ref_perf)
    };
    vec![result]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt_random_msl_generates() {
        let msl = random_msl().expect("codegen failed");
        assert!(msl.contains("mt_random_hash"));
    }

    #[test]
    fn cpu_xorshift_nonzero() {
        let vals = cpu_xorshift(16);
        assert!(vals.iter().any(|&v| v != 0));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn ref_random_compiles() {
        let Ok(runner) = GpuRunner::new() else { return };
        runner.compile(SRC, REF_NAME).unwrap_or_else(|e| panic!("{REF_NAME} compile error: {e}"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn ref_random_runs() {
        let Ok(runner) = GpuRunner::new() else { return };
        let rk = runner.compile(SRC, REF_NAME).expect("compile");
        let key_data = vec![0xDE_u8, 0xAD, 0xBE, 0xEF, 0x12, 0x34, 0x56, 0x78];
        let keys_buf = runner.buffer_bytes(&key_data);
        let out_buf = runner.buffer_zeros(8 * 4);
        let odd_buf = runner.buffer_bytes(&[0u8]);
        let bpk_buf = runner.buffer_bytes(&8u32.to_le_bytes());
        runner.measure(&rk, &[&keys_buf, &out_buf, &odd_buf, &bpk_buf], [1, 1, 1], [1, 1, 1], 0, 1);
        let result = runner.read_f32_slice(&out_buf, 2);
        assert!(result.iter().any(|&v| v != 0.0), "rbitsc produced all zeros");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mt_random_correct() {
        let Ok(runner) = GpuRunner::new() else { return };
        let msl = random_msl().expect("codegen");
        let mk = runner.compile(&msl, "mt_random_hash").expect("compile");
        let n = 1024usize;
        let ref_vals = cpu_xorshift(n);
        let n_buf = runner.buffer_u32(n as u32);
        let out_buf = runner.buffer_zeros(n * 4);
        runner.measure(&mk, &[&out_buf, &n_buf], [n.div_ceil(TPG), 1, 1], [TPG, 1, 1], 0, 1);
        let raw = runner.read_f32_slice(&out_buf, n);
        let mt_vals: Vec<u32> = raw.iter().map(|f| f.to_bits()).collect();
        for (i, (r, m)) in ref_vals.iter().zip(&mt_vals).enumerate() {
            assert_eq!(r, m, "xorshift mismatch at {i}: ref={r} mt={m}");
        }
    }
}
