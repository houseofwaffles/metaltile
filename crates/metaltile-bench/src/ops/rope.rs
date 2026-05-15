//! RoPE benchmark — DSL kernel matching MLX rope<half, int32_t, N=4>.
//!
//! MLX kernel: rope<T, IdxT, N=4>
//!   Params: (in, out, offset[B], scale, strides[3], out_strides[3],
//!            offset_stride, n_head, base)
//!   For forward=true, traditional=false, hs_transpose=false,
//!   contiguous [L, H, D] layout (strides = [D, H*D, 1]):
//!
//!   Grid: [D/(2*N), L, H/N] × [1, 1, 1]  (D=128, N=4 → [16, 512, 8])
//!   Each thread (px, py, pz):
//!     head_base = pz * N
//!     d_norm = float(px) / float(grid.x)
//!     inv_freq = exp2(-d_norm * base)    // base = log2(10000) in practice
//!     theta = float(py) * inv_freq
//!     cos_t, sin_t = cos(theta), sin(theta)
//!     for i in 0..N:
//!       idx1 = py * H*D + (head_base+i) * D + px
//!       idx2 = idx1 + grid.x                     // non-traditional: + D/(2N)
//!       (x1, x2) → rotate by (cos_t, sin_t) → out[idx1], out[idx2]
//!
//! MetalTile: mt_rope_f16 — same algorithm with constexpr grid/shape params.
//!   KernelMode::Grid3D

use metaltile::kernel;
use metaltile_codegen::msl::MslGenerator;
use metaltile_core::ir::KernelMode;

use crate::{
    ops::{OpBench, OpResult, check_equiv, run_f16_once_as_f32, to_gbps},
    runner::GpuRunner,
};

static SRC: &str = include_str!("../metal/rope.metal");

// B=1, H=32, L=512, D=128, N=4 — standard decode rope shape.
const B: usize = 1;
const H: usize = 32;
const L: usize = 512;
const D: usize = 128;
const N_HEADS_PER_GROUP: usize = 4; // N in MLX template
const BENCH: OpBench = OpBench::new("rope_f16", "GB/s");

// Grid dimensions derived from shape (used when rope is implemented).
#[allow(dead_code)]
const GX: usize = D / (2 * N_HEADS_PER_GROUP); // = 16
#[allow(dead_code)]
const GY: usize = L; // = 512
#[allow(dead_code)]
const GZ: usize = H / N_HEADS_PER_GROUP; // = 8

/// RoPE rotation for f16 tensors, Grid3D dispatch matching MLX rope<half,int32_t,4>.
///
/// Parameters:
///   inp, out  — f16 tensors [L * H * D] (contiguous [L, H, D])
///   h_stride  — D = 128     (stride between heads)
///   seq_stride — H*D = 4096 (stride between sequence positions)
///   grid_x    — D/(2*N) = 16 (used for index2 offset and freq normalization)
///   base      — log2(10000) ≈ 13.2877 (RoPE base in log2 units)
///
/// Dispatch: [GX, GY, GZ] × [1, 1, 1]
#[kernel]
pub fn mt_rope_f16(
    inp: Tensor<f16>,
    out: Tensor<f16>,
    #[constexpr] h_stride: u32,
    #[constexpr] seq_stride: u32,
    #[constexpr] grid_x: u32,
    #[constexpr] base: f32,
) {
    let px = program_id::<0>(); // pair index:  0..GX
    let py = program_id::<1>(); // seq position: 0..L
    let pz = program_id::<2>(); // head group:  0..GZ

    // Compute RoPE frequency for this pair index.
    let px_f = px.cast::<f32>();
    let gx_f = grid_x.cast::<f32>();
    let d_norm = px_f / gx_f;
    let inv_freq = exp2(-(d_norm * base));
    let theta = py.cast::<f32>() * inv_freq;
    let cos_t = cos(theta);
    let sin_t = sin(theta);

    // Process N_HEADS_PER_GROUP=4 heads in this group.
    let head_base = pz * 4;
    for i in range(0, 4, 1) {
        let head = head_base + i;
        let idx1 = py * seq_stride + head * h_stride + px;
        let idx2 = idx1 + grid_x;
        let x1 = load(inp[idx1]).cast::<f32>();
        let x2 = load(inp[idx2]).cast::<f32>();
        let rx1 = x1 * cos_t - x2 * sin_t;
        let rx2 = x1 * sin_t + x2 * cos_t;
        store(out[idx1], rx1.cast::<f16>());
        store(out[idx2], rx2.cast::<f16>());
    }
}

fn rope_msl() -> String {
    let mut k = mt_rope_f16::kernel_ir();
    k.mode = KernelMode::Grid3D;
    MslGenerator::default().generate(&k).unwrap_or_else(|e| {
        eprintln!("[mt_rope_f16]: {e}");
        String::new()
    })
}

pub fn bench_rope(runner: &GpuRunner) -> Vec<OpResult> {
    // rope_float16: (in, out, offset[B], scale, strides[3], out_strides[3], offset_stride, n_head,
    //                dummy, dummy, base)
    // Input shape [B*L, H, D] contiguous: strides = [D, H*D, 1] = [128, 4096, 1]
    // Grid: [D/(2*N), L, B*H/N] with tpg=[1,1,1]  (1 thread handles N=4 freq pairs)
    // forward=true(1), traditional=false(2), hs_transpose=false(3)
    let rk = runner
        .compile_with_bool_constants(SRC, "rope_float16", &[(1, true), (2, false), (3, false)])
        .ok();

    // MT kernel: mt_rope_f16 in Grid3D mode
    let msl = rope_msl();
    let mk = runner.compile(&msl, "mt_rope_f16").ok();

    let n_elems = B * L * H * D;
    let in_f16: Vec<u16> = (0..n_elems).map(|i| fp32_to_f16(i as f32 * 0.001)).collect();
    let inp = runner.buffer_f16(&in_f16);

    // Encode strides as i64[3] = [128, 4096, 1]
    let strides_bytes: Vec<u8> =
        [D as i64, (H * D) as i64, 1i64].iter().flat_map(|v| v.to_le_bytes()).collect();
    let strides_buf = runner.buffer_bytes(&strides_bytes);
    let offset_arr = runner.buffer_i32(0i32); // offset[0] = 0
    let scale_buf = runner.buffer_f32_scalar(1.0f32);
    let offset_stride_buf = runner.buffer_i64(1i64);
    let n_head_buf = runner.buffer_i32(H as i32);
    let dummy = runner.buffer_zeros(4);
    // base parameter: both MLX and MT kernels use exp2(-d * base), so base = log2(10000)
    let base_buf = runner.buffer_f32_scalar((10000f32).log2());

    let bytes = (n_elems * 2 * 2) as f64; // read + write f16

    // Reference kernel dispatch: [GX, GY, GZ] × [1,1,1]
    let ref_out = runner.buffer_zeros(n_elems * 2); // f16
    let ref_perf = rk.as_ref().and_then(|rk| {
        let st = runner.bench(
            rk,
            &[
                &inp,
                &ref_out,
                &offset_arr,
                &scale_buf,
                &strides_buf,
                &strides_buf, // out_strides same as in_strides
                &offset_stride_buf,
                &n_head_buf,
                &dummy,    // slot 8
                &dummy,    // slot 9
                &base_buf, // slot 10
            ],
            [GX, GY, GZ],
            [1, 1, 1],
            3,
            10,
        );
        to_gbps(&st, bytes)
    });

    // MT kernel buffers: (inp, out, h_stride, seq_stride, grid_x, base)
    // h_stride=D=128, seq_stride=H*D=4096, grid_x=GX=16, base=log2(10000)
    let mt_h_stride = runner.buffer_u32(D as u32);
    let mt_seq_stride = runner.buffer_u32((H * D) as u32);
    let mt_grid_x = runner.buffer_u32(GX as u32);
    let mt_base = runner.buffer_f32_scalar((10000f32).log2());

    // Correctness check on a small CL=4 sub-problem, using the MLX reference as ground truth
    let equiv = mk.as_ref().and_then(|mk| {
        rk.as_ref().map(|rk| {
            // Use L_CHECK=4, full H and D
            const L_CHECK: usize = 4;
            let n_check = B * L_CHECK * H * D;
            let check_f16: Vec<u16> = (0..n_check).map(|i| fp32_to_f16(i as f32 * 0.001)).collect();
            let inp_c = runner.buffer_f16(&check_f16);
            let ref_out_c = runner.buffer_zeros(n_check * 2);
            let mt_out_c = runner.buffer_zeros(n_check * 2);
            let ck_h_stride = runner.buffer_u32(D as u32);
            let ck_seq_stride = runner.buffer_u32((H * D) as u32);
            let ck_grid_x = runner.buffer_u32(GX as u32);
            let ck_base = runner.buffer_f32_scalar((10000f32).log2());

            // Run MLX reference
            let ref_vals = run_f16_once_as_f32(
                runner,
                rk,
                &[
                    &inp_c,
                    &ref_out_c,
                    &offset_arr,
                    &scale_buf,
                    &strides_buf,
                    &strides_buf,
                    &offset_stride_buf,
                    &n_head_buf,
                    &dummy,
                    &dummy,
                    &base_buf,
                ],
                &ref_out_c,
                n_check,
                [GX, L_CHECK, GZ],
                [1, 1, 1],
            );

            // Run MT kernel
            let mt_vals = run_f16_once_as_f32(
                runner,
                mk,
                &[&inp_c, &mt_out_c, &ck_h_stride, &ck_seq_stride, &ck_grid_x, &ck_base],
                &mt_out_c,
                n_check,
                [GX, L_CHECK, GZ],
                [1, 1, 1],
            );

            check_equiv(&ref_vals, &mt_vals, 0.01)
        })
    });

    // MT performance on full input
    let mt_out = runner.buffer_zeros(n_elems * 2);
    let mt_perf = mk.as_ref().and_then(|mk| {
        let st = runner.bench(
            mk,
            &[&inp, &mt_out, &mt_h_stride, &mt_seq_stride, &mt_grid_x, &mt_base],
            [GX, GY, GZ],
            [1, 1, 1],
            3,
            10,
        );
        to_gbps(&st, bytes)
    });

    let shape = format!("B{B}H{H}L{L}D{D}");
    if let Some(mt_perf) = mt_perf {
        vec![BENCH.implemented(shape, ref_perf, mt_perf, equiv.expect("mk and rk both Some"))]
    } else {
        vec![BENCH.nyi(shape, ref_perf)]
    }
}

/// Convert f32 to f16 bits (simple approximation for test data).
fn fp32_to_f16(v: f32) -> u16 {
    // Use half crate isn't available, use unsafe transmute via bit manipulation.
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3ff;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7c00
    } else {
        sign | ((exp as u16) << 10) | mant as u16
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_msl_generates() {
        let msl = rope_msl();
        assert!(!msl.trim().is_empty());
        assert!(msl.contains("mt_rope_f16"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rope_kernel_compiles() {
        let Ok(runner) = GpuRunner::new() else {
            return;
        };
        let msl = rope_msl();
        runner
            .compile(&msl, "mt_rope_f16")
            .unwrap_or_else(|e| panic!("mt_rope_f16 compile error: {e}\nMSL:\n{msl}"));
    }
}
