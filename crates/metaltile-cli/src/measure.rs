//! GPU measurement helpers.
//!
//! Moved from `metaltile-bench/src/ops/shared.rs` during Phase 3 refactor.
//! Pure types like `OpResult`, `EquivResult`, `DtypeCtx` live in `metaltile-std::bench_types`.

use metaltile_std::bench_types::{DType, OpResult, elem_bytes};

use crate::{
    runner::{CompiledKernel, GpuBuffer, GpuRunner},
    stats::BenchStats,
};

// ── Dtype ↔ GPU buffer helpers ────────────────────────────────────────────────

fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 31) as u16) << 15;
    let exp = ((x >> 23) & 0xFF) as i32 - 127 + 15;
    let mant32 = x & 0x7F_FFFF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    let mant16 = mant32 >> 13;
    let round_bit = (mant32 >> 12) & 1;
    let sticky = mant32 & 0xFFF;
    let round_up = round_bit == 1 && (sticky != 0 || (mant16 & 1) == 1);
    let mant16 = (mant16 + u32::from(round_up)) as u16;
    if mant16 > 0x3FF {
        sign | (((exp + 1) as u16) << 10)
    } else {
        sign | ((exp as u16) << 10) | mant16
    }
}

fn f32_to_bf16(v: f32) -> u16 {
    let x = v.to_bits();
    let rounded = x.wrapping_add(0x7FFF).wrapping_add((x >> 16) & 1);
    (rounded >> 16) as u16
}

pub fn buffer_typed(runner: &GpuRunner, vals: &[f32], dt: DType) -> GpuBuffer {
    match dt {
        DType::F32 => runner.buffer_f32(vals),
        DType::F16 => runner.buffer_f16(&vals.iter().map(|&v| f32_to_f16(v)).collect::<Vec<_>>()),
        DType::BF16 => runner.buffer_f16(&vals.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>()),
        DType::I32 => runner
            .buffer_bytes(&vals.iter().flat_map(|&v| (v as i32).to_le_bytes()).collect::<Vec<_>>()),
        DType::U32 => runner
            .buffer_bytes(&vals.iter().flat_map(|&v| (v as u32).to_le_bytes()).collect::<Vec<_>>()),
        DType::I8 => runner.buffer_bytes(&vals.iter().map(|&v| v as i8 as u8).collect::<Vec<_>>()),
        DType::U8 => runner.buffer_bytes(&vals.iter().map(|&v| v as u8).collect::<Vec<_>>()),
        DType::Bool => runner.buffer_f32(vals),
        _ => unimplemented!("buffer_typed: unsupported dtype {dt:?}"),
    }
}

pub fn zeros_typed(runner: &GpuRunner, n: usize, dt: DType) -> GpuBuffer {
    runner.buffer_zeros(n * elem_bytes(dt))
}

pub fn read_typed(runner: &GpuRunner, buf: &GpuBuffer, n: usize, dt: DType) -> Vec<f32> {
    match dt {
        DType::F32 => runner.read_f32_slice(buf, n),
        DType::F16 => runner.read_f16_slice(buf, n),
        DType::BF16 => runner.read_bf16_slice(buf, n),
        DType::I32 => {
            let bytes = runner.read_bytes(buf, n * 4);
            bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect()
        },
        DType::U32 => {
            let bytes = runner.read_bytes(buf, n * 4);
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect()
        },
        DType::I8 => {
            let bytes = runner.read_bytes(buf, n);
            bytes.iter().map(|&b| b as i8 as f32).collect()
        },
        DType::U8 => {
            let bytes = runner.read_bytes(buf, n);
            bytes.iter().map(|&b| b as f32).collect()
        },
        _ => runner.read_f32_slice(buf, n),
    }
}

// ── Single-run dispatch ──────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn run_typed_once(
    runner: &GpuRunner,
    kernel: &CompiledKernel,
    buffers: &[&GpuBuffer],
    out: &GpuBuffer,
    n: usize,
    tgs: [usize; 3],
    tpg: [usize; 3],
    dt: DType,
) -> Vec<f32> {
    runner.measure(kernel, buffers, tgs, tpg, 0, 1);
    read_typed(runner, out, n, dt)
}

pub fn run_f16_once_as_f32(
    runner: &GpuRunner,
    kernel: &CompiledKernel,
    buffers: &[&GpuBuffer],
    out: &GpuBuffer,
    n: usize,
    tgs: [usize; 3],
    tpg: [usize; 3],
) -> Vec<f32> {
    runner.measure(kernel, buffers, tgs, tpg, 0, 1);
    runner.read_f16_slice(out, n)
}

// ── Throughput ───────────────────────────────────────────────────────────────

pub fn to_gflops(st: &BenchStats, flops: f64) -> Option<f64> {
    st.is_valid().then(|| flops / (st.mean_us * 1e-6) / 1e9)
}

pub fn to_gbps(st: &BenchStats, bytes: f64) -> Option<f64> {
    // Use median to reject outlier iterations (JIT stalls, OS scheduler spikes).
    st.is_valid().then(|| bytes / (st.median_us * 1e-6) / 1e9)
}

pub fn bench_gbps(
    runner: &GpuRunner,
    kernel: &CompiledKernel,
    buffers: &[&GpuBuffer],
    grid: [usize; 3],
    tpg: [usize; 3],
    bytes: f64,
) -> Option<f64> {
    runner.flush_slc();
    // 5 warmup iterations ensure the working set is fully resident in SLC
    // before any measurements are taken; 20 measurement iterations give a
    // stable distribution from which the median is drawn.
    to_gbps(&runner.bench(kernel, buffers, grid, tpg, 5, 20), bytes)
}

pub fn bench_all_dtypes<F>(runner: &GpuRunner, f: F) -> Vec<OpResult>
where F: Fn(&GpuRunner, DType) -> Vec<OpResult> {
    metaltile_std::bench_types::FLOAT_DTYPES.iter().flat_map(|&dt| f(runner, dt)).collect()
}
