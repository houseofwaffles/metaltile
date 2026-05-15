//! Debug tool: dump generated MSL for each #[kernel] to stdout or a directory.
//!
//! Usage:
//!   cargo run -p metaltile-bench --bin dump_msl
//!   cargo run -p metaltile-bench --bin dump_msl -- --filter rms
//!   cargo run -p metaltile-bench --bin dump_msl -- --dir /tmp/msl_out
//!   cargo run -p metaltile-bench --bin dump_msl -- rms_norm  (positional = filter)

use metaltile::core::ir::KernelMode;
use metaltile_bench::ops::{
    DType,
    arange::mt_arange,
    arg_reduce::mt_argmax_f32,
    binary::vector_add,
    binary_two::mt_binary_two,
    copy::mt_copy,
    fp_quantized::mt_fp4_quant_dequant,
    gemv::mt_gemv,
    layer_norm::mt_layer_norm,
    logsumexp::mt_logsumexp,
    quantized::mt_qmv_f32,
    random::mt_random_hash,
    reduce::{mt_all_reduce, mt_row_reduce},
    rms_norm::mt_rms_norm,
    rope::mt_rope_f16,
    scaled_dot_product_attention::mt_sdpa,
    scan::mt_scan_f32,
    softmax::mt_softmax,
    ternary::mt_select,
    unary::{mt_abs, mt_exp, mt_gelu, mt_log, mt_relu, mt_rsqrt, mt_silu, mt_sqrt},
};
use metaltile_codegen::{
    TileSchedule,
    msl::{MslConfig, MslGenerator},
};

struct KernelSpec {
    name: &'static str,
    mode: KernelMode,
    msl: String,
}

fn collect_kernels() -> Vec<KernelSpec> {
    let mut out = Vec::new();

    macro_rules! add {
        ($name:expr, $ir:expr, $mode:expr) => {{
            let mut k = $ir;
            k.mode = $mode;
            let msl =
                MslGenerator::default().generate(&k).unwrap_or_else(|e| format!("// ERROR: {e}\n"));
            out.push(KernelSpec { name: $name, mode: $mode, msl });
        }};
    }

    // Reduction kernels
    add!("mt_rms_norm", mt_rms_norm::kernel_ir(), KernelMode::Reduction);
    add!("mt_layer_norm", mt_layer_norm::kernel_ir(), KernelMode::Reduction);
    add!("mt_logsumexp", mt_logsumexp::kernel_ir(), KernelMode::Reduction);
    add!("mt_softmax", mt_softmax::kernel_ir(), KernelMode::Reduction);
    add!("mt_all_reduce", mt_all_reduce::kernel_ir(), KernelMode::Reduction);
    add!("mt_row_reduce", mt_row_reduce::kernel_ir(), KernelMode::Reduction);
    add!("mt_gemv", mt_gemv::kernel_ir(), KernelMode::Reduction);

    // Elementwise kernels
    add!("vector_add", vector_add::kernel_ir(), KernelMode::Elementwise);
    add!("mt_copy", mt_copy::kernel_ir(), KernelMode::Elementwise);
    add!("mt_arange", mt_arange::kernel_ir(), KernelMode::Elementwise);
    add!("mt_select", mt_select::kernel_ir(), KernelMode::Elementwise);
    add!("mt_rope", mt_rope_f16::kernel_ir(), KernelMode::Grid3D);
    add!("mt_exp", mt_exp::kernel_ir(), KernelMode::Elementwise);
    add!("mt_log", mt_log::kernel_ir(), KernelMode::Elementwise);
    add!("mt_sqrt", mt_sqrt::kernel_ir(), KernelMode::Elementwise);
    add!("mt_rsqrt", mt_rsqrt::kernel_ir(), KernelMode::Elementwise);
    add!("mt_abs", mt_abs::kernel_ir(), KernelMode::Elementwise);
    add!("mt_silu", mt_silu::kernel_ir(), KernelMode::Elementwise);
    add!("mt_gelu", mt_gelu::kernel_ir(), KernelMode::Elementwise);
    add!("mt_relu", mt_relu::kernel_ir(), KernelMode::Elementwise);

    // Scan / argmax
    add!("mt_scan_f32", mt_scan_f32::kernel_ir(), KernelMode::Reduction);
    add!("mt_sdpa", mt_sdpa::kernel_ir_for(DType::F32), KernelMode::Reduction);
    add!("mt_argmax_f32", mt_argmax_f32::kernel_ir(), KernelMode::Reduction);

    add!("mt_binary_two", mt_binary_two::kernel_ir(), KernelMode::Elementwise);
    add!("mt_random_hash", mt_random_hash::kernel_ir(), KernelMode::Elementwise);
    add!("mt_fp4_quant_dequant", mt_fp4_quant_dequant::kernel_ir(), KernelMode::Elementwise);

    // Quantized GeMV
    {
        let mut k = mt_qmv_f32::kernel_ir();
        k.mode = KernelMode::Reduction;
        let msl =
            MslGenerator::default().generate(&k).unwrap_or_else(|e| format!("// ERROR: {e}\n"));
        out.push(KernelSpec { name: "mt_qmv_f32", mode: KernelMode::Reduction, msl });
    }

    // Tile2D matmul (scalar path)
    {
        use metaltile_bench::ops::steel::gemm::steel_gemm_fused::mt_matmul;
        let mut k = mt_matmul::kernel_ir();
        k.mode = KernelMode::Tile2D;
        let cfg = MslConfig {
            tile_schedule: TileSchedule::default(),
            use_simd_matrix: true,
            ..MslConfig::default()
        };
        let msl =
            MslGenerator::new(cfg).generate(&k).unwrap_or_else(|e| format!("// ERROR: {e}\n"));
        out.push(KernelSpec { name: "mt_matmul", mode: KernelMode::Tile2D, msl });
    }

    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let filter = flag_val(&args, "--filter")
        .or_else(|| args.get(1).filter(|a| !a.starts_with('-')).cloned());
    let dir = flag_val(&args, "--dir");

    let kernels = collect_kernels();
    let matched: Vec<_> = kernels
        .iter()
        .filter(|k| filter.as_deref().map(|f| k.name.contains(f)).unwrap_or(true))
        .collect();

    if matched.is_empty() {
        eprintln!("No kernels matched filter {:?}", filter);
        eprintln!("Available: {}", kernels.iter().map(|k| k.name).collect::<Vec<_>>().join(", "));
        std::process::exit(1);
    }

    if let Some(ref d) = dir {
        std::fs::create_dir_all(d).expect("failed to create output directory");
        for k in &matched {
            let path = format!("{}/{}.metal", d, k.name);
            std::fs::write(&path, &k.msl).expect("write failed");
            println!("wrote {path}");
        }
    } else {
        for k in &matched {
            let mode_str = match k.mode {
                KernelMode::Elementwise => "Elementwise",
                KernelMode::Reduction => "Reduction",
                KernelMode::Tile2D => "Tile2D",
                KernelMode::Grid3D => "Grid3D",
            };
            println!("// ═══════════════════════════════════════════════════════");
            println!("// kernel: {}  mode: {}", k.name, mode_str);
            println!("// ═══════════════════════════════════════════════════════");
            println!("{}", k.msl);
        }
    }
}

fn flag_val(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}
