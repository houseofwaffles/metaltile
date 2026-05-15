/// Hardware performance ceiling for Apple Silicon GPU chips.
pub struct ChipProfile {
    pub name: &'static str,
    /// Peak fp16 TFLOPS (2× fp32 on Apple GPUs; both ALU and simdgroup matmul).
    pub peak_tflops_fp16: f64,
    /// Peak fp32 TFLOPS.
    pub peak_tflops_fp32: f64,
    /// Peak memory bandwidth GB/s.
    pub peak_bw_gbps: f64,
}

impl ChipProfile {
    /// Ridge point in FLOPS/byte: above this the kernel is compute-bound.
    pub fn ridge_point_fp16(&self) -> f64 {
        self.peak_tflops_fp16 * 1e12 / (self.peak_bw_gbps * 1e9)
    }

    /// Theoretical peak GFLOPS for a given arithmetic intensity (FLOPS/byte).
    pub fn ceiling_gflops_fp16(&self, ai: f64) -> f64 {
        let compute_bound = self.peak_tflops_fp16 * 1e3; // GFLOPS
        let memory_bound = ai * self.peak_bw_gbps; // GFLOPS
        compute_bound.min(memory_bound)
    }
}

static CHIPS: &[ChipProfile] = &[
    // ── M1 family ──────────────────────────────────────────────────────────
    ChipProfile { name: "M1", peak_tflops_fp16: 5.5, peak_tflops_fp32: 2.6, peak_bw_gbps: 68.0 },
    ChipProfile {
        name: "M1 Pro",
        peak_tflops_fp16: 6.6,
        peak_tflops_fp32: 3.3,
        peak_bw_gbps: 200.0,
    },
    ChipProfile {
        name: "M1 Max",
        peak_tflops_fp16: 10.4,
        peak_tflops_fp32: 5.2,
        peak_bw_gbps: 400.0,
    },
    ChipProfile {
        name: "M1 Ultra",
        peak_tflops_fp16: 20.8,
        peak_tflops_fp32: 10.4,
        peak_bw_gbps: 800.0,
    },
    // ── M2 family ──────────────────────────────────────────────────────────
    ChipProfile { name: "M2", peak_tflops_fp16: 6.8, peak_tflops_fp32: 3.4, peak_bw_gbps: 100.0 },
    ChipProfile {
        name: "M2 Pro",
        peak_tflops_fp16: 9.7,
        peak_tflops_fp32: 4.8,
        peak_bw_gbps: 200.0,
    },
    ChipProfile {
        name: "M2 Max",
        peak_tflops_fp16: 13.6,
        peak_tflops_fp32: 6.8,
        peak_bw_gbps: 400.0,
    },
    ChipProfile {
        name: "M2 Ultra",
        peak_tflops_fp16: 27.2,
        peak_tflops_fp32: 13.6,
        peak_bw_gbps: 800.0,
    },
    // ── M3 family ──────────────────────────────────────────────────────────
    ChipProfile { name: "M3", peak_tflops_fp16: 7.4, peak_tflops_fp32: 3.6, peak_bw_gbps: 102.0 },
    ChipProfile {
        name: "M3 Pro",
        peak_tflops_fp16: 11.5,
        peak_tflops_fp32: 5.8,
        peak_bw_gbps: 150.0,
    },
    ChipProfile {
        name: "M3 Max",
        peak_tflops_fp16: 14.2,
        peak_tflops_fp32: 7.0,
        peak_bw_gbps: 300.0,
    },
    ChipProfile {
        name: "M3 Ultra",
        peak_tflops_fp16: 28.4,
        peak_tflops_fp32: 14.0,
        peak_bw_gbps: 600.0,
    },
    // ── M4 family ──────────────────────────────────────────────────────────
    // fp16 = 2× fp32 on Apple GPUs (two half-precision ops per ALU per cycle).
    // Sources: Apple press release (Oct 2024), Flopper.io, NotebookCheck.
    ChipProfile { name: "M4", peak_tflops_fp16: 8.6, peak_tflops_fp32: 4.3, peak_bw_gbps: 120.0 },
    ChipProfile {
        name: "M4 Pro",
        peak_tflops_fp16: 18.4,
        peak_tflops_fp32: 9.2,
        peak_bw_gbps: 273.0,
    },
    // M4 Max ships in two variants: 32-core GPU (410 GB/s) and 40-core GPU (546 GB/s).
    // Device name from Metal API doesn't include core count, so we default to the
    // top-SKU (40-core). If you're on the 32-core variant, expect ~80% of these peaks.
    ChipProfile {
        name: "M4 Max",
        peak_tflops_fp16: 36.9,
        peak_tflops_fp32: 18.4,
        peak_bw_gbps: 546.0,
    },
];

/// Find the best matching chip for a device name string (longest match wins).
pub fn lookup_chip(device_name: &str) -> Option<&'static ChipProfile> {
    CHIPS.iter().filter(|c| device_name.contains(c.name)).max_by_key(|c| c.name.len())
}
