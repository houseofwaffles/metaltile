#!/usr/bin/env python3
"""
MetalTile vs MLX reference kernel coverage.

Uses the C preprocessor to expand all Metal macros and extract the exact
[[host_name(...)]] values, then cross-references with what each bench op
actually tries to compile.

Two sections:
  1. Per-op reference status — for each MT op/dtype, what MLX ref is used
     and does it exist in the preprocessed Metal source?
  2. Metal file coverage — what fraction of MLX kernels are benchmarked?
"""

import glob
import os
import re
import subprocess
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METAL_DIR  = os.path.join(SCRIPT_DIR, "src/metal")
OP_DIR     = os.path.join(SCRIPT_DIR, "src/ops")

# MLX dtype name as used in Metal kernel names (mlx_tname in shared.rs)
MLX_DTYPES = {
    "f32":  "float32",
    "f16":  "float16",
    "bf16": "bfloat16",
}

# ── Metal source preprocessing ────────────────────────────────────────────────

_metal_kernel_cache: dict[str, set[str]] = {}

def metal_kernels(metal_file: str) -> set[str]:
    """Return all kernel host_names from a Metal file via C preprocessor."""
    if metal_file in _metal_kernel_cache:
        return _metal_kernel_cache[metal_file]
    try:
        result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "clang", "-E", "-x", "c", metal_file],
            capture_output=True, text=True, timeout=60,
        )
        names: set[str] = set()
        # Template instantiations: [[host_name("a" "b" ...)]]
        for m in re.finditer(r"host_name\(([^)]+)\)", result.stdout):
            parts = re.findall(r'"([^"]+)"', m.group(1))
            names.add("".join(parts))
        # Direct kernel functions: [[kernel]] void name(
        for m in re.finditer(r"\[\[kernel\]\]\s+void\s+(\w+)", result.stdout):
            names.add(m.group(1))
        _metal_kernel_cache[metal_file] = names
        return names
    except Exception as e:
        print(f"  [warn] preprocessing {os.path.basename(metal_file)}: {e}", file=sys.stderr)
        _metal_kernel_cache[metal_file] = set()
        return set()


def all_metal_files() -> list[str]:
    top   = sorted(glob.glob(f"{METAL_DIR}/*.metal"))
    steel = sorted(glob.glob(f"{METAL_DIR}/steel/**/*.metal", recursive=True))
    return top + steel


# ── Bench reference extraction ────────────────────────────────────────────────
#
# We define the reference patterns per op-file using a small declarative spec
# rather than trying to auto-parse Rust code (which would be fragile for all
# the varied patterns used across 15+ op files).
#
# Each entry is:
#   (op_file_rel, mt_kernel, metal_file_rel, ref_spec, dtypes)
#
# ref_spec is one of:
#   ("literal",  "kernel_name")               — single fixed name
#   ("format",   "template_{tn}")             — substitute {tn} per dtype
#   ("unary_v",  "OpName" or None)            — v_{Op}{tn}{tn} or None
#   ("none",     "reason")                    — no reference, with reason
#
# dtypes: list of short names ["f32","f16","bf16"] that this op covers,
#         or None to mean the ref applies without dtype distinction.

BENCH_SPECS = [
    # ── unary (all from UNARY_SPECS in unary.rs) ──────────────────────────────
    # ops with MLX ref: v_{Op}{tn}{tn} via instantiate_unary_float
    *[("unary", f"mt_{op}", "unary.metal", ("unary_v", op_cap), ["f32","f16","bf16"])
      for op, op_cap in [
          ("exp","Exp"), ("log","Log"), ("sqrt","Sqrt"), ("rsqrt","Rsqrt"),
          ("abs","Abs"), ("cos","Cos"), ("sin","Sin"), ("ceil","Ceil"),
          ("floor","Floor"), ("erf","Erf"), ("log2","Log2"), ("sign","Sign"),
          ("round","Round"), ("neg","Negative"), ("square","Square"),
          ("sigmoid","Sigmoid"), ("log1p","Log1p"),
      ]],
    # ops without a standalone MLX unary kernel
    ("unary", "mt_silu",   "unary.metal", ("none", "activation — MLX computes as x·sigmoid(x), no standalone unary kernel"), ["f32","f16","bf16"]),
    ("unary", "mt_gelu",   "unary.metal", ("none", "activation — MLX uses composite poly, no standalone unary kernel"),       ["f32","f16","bf16"]),
    ("unary", "mt_relu",   "unary.metal", ("none", "activation — MLX uses vvn_Maximum with scalar 0, not a unary op"),        ["f32","f16","bf16"]),
    ("unary", "mt_exp2",   "unary.metal", ("none", "exp2 not in instantiate_unary_float; MLX uses exp(x·ln2)"),                ["f32","f16","bf16"]),
    ("unary", "mt_recip",  "unary.metal", ("none", "recip not in unary.metal; MLX uses binary divide kernel"),                ["f32","f16","bf16"]),

    # ── binary ────────────────────────────────────────────────────────────────
    ("binary", "mt_add",       "binary.metal", ("format", "vvn_Add{tn}"),        ["f32","f16","bf16"]),
    ("binary", "mt_mul",       "binary.metal", ("format", "vvn_Multiply{tn}"),   ["f32","f16","bf16"]),
    ("binary", "mt_sub",       "binary.metal", ("format", "vvn_Subtract{tn}"),   ["f32","f16","bf16"]),
    ("binary", "mt_div",       "binary.metal", ("format", "vvn_Divide{tn}"),     ["f32","f16","bf16"]),
    ("binary", "mt_max",       "binary.metal", ("format", "vvn_Maximum{tn}"),    ["f32","f16","bf16"]),
    ("binary", "mt_min",       "binary.metal", ("format", "vvn_Minimum{tn}"),    ["f32","f16","bf16"]),
    ("binary", "mt_pow",       "binary.metal", ("format", "vvn_Power{tn}"),      ["f32","f16","bf16"]),
    ("binary", "mt_logaddexp", "binary.metal", ("format", "vvn_LogAddExp{tn}"),  ["f32","f16","bf16"]),

    # ── binary_two ────────────────────────────────────────────────────────────
    ("binary_two", "mt_binary_two", "binary_two.metal",
     ("none", "no MLX equivalent — MT benchmarks 2-output fused pass that MLX doesn't expose"), ["f32","f16","bf16"]),

    # ── copy ──────────────────────────────────────────────────────────────────
    ("copy", "mt_copy", "copy.metal", ("format", "v_copy{tn}{tn}"), ["f32","f16","bf16"]),

    # ── arange ────────────────────────────────────────────────────────────────
    ("arange", "mt_arange", "arange.metal", ("format", "arange{tn}"), ["f32","f16","bf16"]),

    # ── ternary (select) ─────────────────────────────────────────────────────
    ("ternary", "mt_select", "ternary.metal", ("format", "v_Select{tn}"), ["f32","f16","bf16"]),

    # ── softmax ───────────────────────────────────────────────────────────────
    ("softmax", "mt_softmax", "softmax.metal", ("format", "looped_softmax_{tn}"), ["f32","f16","bf16"]),

    # ── rms_norm ──────────────────────────────────────────────────────────────
    ("rms_norm", "mt_rms_norm", "rms_norm.metal", ("format", "rms{tn}"), ["f32","f16","bf16"]),

    # ── layer_norm ────────────────────────────────────────────────────────────
    ("layer_norm", "mt_layer_norm", "layer_norm.metal", ("format", "layer_norm_looped{tn}"), ["f32","f16","bf16"]),

    # ── logsumexp ─────────────────────────────────────────────────────────────
    ("logsumexp", "mt_logsumexp", "logsumexp.metal", ("format", "looped_logsumexp_{tn}"), ["f32","f16","bf16"]),

    # ── reduce ────────────────────────────────────────────────────────────────
    ("reduce", "mt_all_reduce",     "reduce.metal", ("format", "all_reduce_sum{tn}"),            ["f32","f16","bf16"]),
    ("reduce", "mt_all_reduce_max", "reduce.metal", ("format", "all_reduce_max{tn}"),            ["f32","f16","bf16"]),
    ("reduce", "mt_all_reduce_min", "reduce.metal", ("format", "all_reduce_min{tn}"),            ["f32","f16","bf16"]),
    ("reduce", "mt_row_reduce",     "reduce.metal", ("format", "row_reduce_simple_sum{tn}"),     ["f32","f16","bf16"]),
    ("reduce", "mt_row_reduce_max", "reduce.metal", ("format", "row_reduce_simple_max{tn}"),     ["f32","f16","bf16"]),
    ("reduce", "mt_row_reduce_min", "reduce.metal", ("format", "row_reduce_simple_min{tn}"),     ["f32","f16","bf16"]),

    # ── gemv ──────────────────────────────────────────────────────────────────
    ("gemv", "mt_gemv", "gemv.metal",
     ("format", "gemv_{tn}_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0"), ["f32","f16","bf16"]),

    # ── gemv_masked ───────────────────────────────────────────────────────────
    ("gemv_masked", "mt_gemv_masked", "gemv_masked.metal",
     ("none", "no nomask/nomask variant in instantiate_gemv_base; all MLX variants require explicit mask buffers"), ["f32","f16","bf16"]),

    # ── rope ──────────────────────────────────────────────────────────────────
    ("rope", "mt_rope_f16", "rope.metal", ("literal", "rope_float16"), None),

    # ── scaled_dot_product_attention ─────────────────────────────────────────
    ("scaled_dot_product_attention", "mt_sdpa", "scaled_dot_product_attention.metal",
     ("literal", "sdpa_vector_float_128_128"), ["f32"]),
    ("scaled_dot_product_attention", "mt_sdpa", "scaled_dot_product_attention.metal",
     ("literal", "sdpa_vector_float16_t_128_128"), ["f16"]),

    # ── scan ──────────────────────────────────────────────────────────────────
    ("scan", "mt_scan_f32", "scan.metal",
     ("literal", "contig_scan_inclusive_sum_float32_float32"), None),

    # ── arg_reduce ────────────────────────────────────────────────────────────
    ("arg_reduce", "mt_argmax", "arg_reduce.metal", ("literal", "argmax_float32"), None),

    # ── sort ──────────────────────────────────────────────────────────────────
    ("sort", "mt_sort_f32", "sort.metal",
     ("literal", "c_block_sort_float32_float32_bn256_tn4"), None),

    # ── random ────────────────────────────────────────────────────────────────
    ("random", "mt_random_hash", "random.metal",
     ("literal", "rbitsc"), None),  # direct [[kernel]] void rbitsc in random.metal

    # ── fp_quantized ──────────────────────────────────────────────────────────
    ("fp_quantized", "mt_fp4_quant_dequant", "fp_quantized.metal",
     ("literal", "nvfp4_quantize_dequantize_float_gs_16_b_4"), None),

    # ── quantized ─────────────────────────────────────────────────────────────
    ("quantized", "mt_qmv_f32", "quantized.metal",
     ("literal", "affine_qmv_fast_float16_t_gs_64_b_4_batch_0"), None),

    # ── strided copy (non-contiguous tensor support) ──────────────────────────
    # g2_copy{tn}{tn}: 2D strided copy (copy_g_nd2 template instantiated via instantiate_copy)
    *[("strided", "mt_strided_copy", "copy.metal", ("format", "g2_copy{tn}{tn}"), [dt])
      for dt in ["f32","f16","bf16"]],

    # ── steel/gemm (matmul) ───────────────────────────────────────────────────
    ("steel/gemm/steel_gemm_fused", "mt_matmul",
     "steel/gemm/steel_gemm_fused.metal",
     ("literal", "steel_gemm_fused_nn_float16_float16_bm64_bn64_bk16_wm2_wn2"), None),
]


def resolve_ref_name(spec_type: str, spec_val: str, dtype: str) -> str | None:
    """Expand a ref spec to a concrete kernel name for the given dtype."""
    tn = MLX_DTYPES.get(dtype, dtype)
    if spec_type == "none":
        return None
    if spec_type == "literal":
        return spec_val
    if spec_type == "format":
        return spec_val.replace("{tn}", tn)
    if spec_type == "unary_v":
        # spec_val is the capitalized op name, e.g. "Exp"
        return f"v_{spec_val}{tn}{tn}"
    return None


# ── Output helpers ────────────────────────────────────────────────────────────

ICON = {"ok": "✅", "missing": "❌", "no_ref": "—", "warn": "⚠️"}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Preprocess all Metal files upfront (in parallel would be nicer but not critical)
    print("Preprocessing Metal files…", file=sys.stderr)
    all_mf = all_metal_files()
    mf_map: dict[str, str] = {}   # basename → full path
    for mf in all_mf:
        rel = os.path.relpath(mf, METAL_DIR)
        mf_map[rel] = mf
        metal_kernels(mf)  # warm cache

    # Track which kernels from each Metal file are referenced by the bench
    mf_benched: dict[str, set[str]] = defaultdict(set)

    # ── Section 1: Per-op reference status ───────────────────────────────────
    lines = []
    lines.append("# MetalTile vs MLX Reference Kernel Coverage")
    lines.append("")
    lines.append("## Per-Op Reference Status")
    lines.append("")
    lines.append("Legend: ✅ exists in Metal source  ❌ NOT found  — no reference")
    lines.append("")
    lines.append("| Op | MT Kernel | Dtype | MLX Reference | Status |")
    lines.append("|---|---|---|---|---|")

    prev_op = None
    for spec in BENCH_SPECS:
        op_file, mt_kernel, metal_rel, (spec_type, spec_val), dtype_list = spec
        dtypes = dtype_list if dtype_list else [None]

        for dtype in dtypes:
            ref_name = resolve_ref_name(spec_type, spec_val, dtype or "")

            if spec_type == "none":
                status = ICON["no_ref"]
                ref_display = f"*{spec_val}*"
            else:
                full_mf = mf_map.get(metal_rel)
                if full_mf is None:
                    status = ICON["warn"] + " metal file not found"
                    ref_display = f"`{ref_name}`" if ref_name else "?"
                else:
                    kernels = metal_kernels(full_mf)
                    if ref_name in kernels:
                        status = ICON["ok"]
                        mf_benched[metal_rel].add(ref_name)
                    else:
                        status = ICON["missing"]
                    ref_display = f"`{ref_name}`" if ref_name else "?"

            op_label = op_file if op_file != prev_op else ""
            prev_op = op_file
            dt_label = dtype or ""

            lines.append(f"| {op_label} | `{mt_kernel}` | {dt_label} | {ref_display} | {status} |")

    # ── Section 2: Metal file coverage ───────────────────────────────────────
    lines.append("")
    lines.append("## Metal File Coverage")
    lines.append("")
    lines.append("How many of each Metal file's instantiated kernels are used as references.")
    lines.append("")
    lines.append("| Metal File | Total kernels | Benchmarked | % | Unbenchmarked examples |")
    lines.append("|---|---|---|---|---|")

    grand_total = grand_benched = 0
    for metal_rel in sorted(mf_map):
        full_mf = mf_map[metal_rel]
        all_k = metal_kernels(full_mf)
        if not all_k:
            continue
        benched = mf_benched.get(metal_rel, set())
        unbench = sorted(all_k - benched)
        pct = int(100 * len(benched) / len(all_k)) if all_k else 0
        grand_total   += len(all_k)
        grand_benched += len(benched)

        examples = ", ".join(f"`{k}`" for k in unbench[:3])
        if len(unbench) > 3:
            examples += f", … (+{len(unbench)-3} more)"
        if not unbench:
            examples = "—"

        status_icon = "✅" if pct == 100 else ("⚠️" if pct > 0 else "❌")
        lines.append(f"| `{metal_rel}` | {len(all_k)} | {len(benched)} | {status_icon} {pct}% | {examples} |")

    lines.append("")
    total_pct = int(100 * grand_benched / grand_total) if grand_total else 0
    lines.append(f"**Total**: {grand_benched}/{grand_total} instantiated kernels benchmarked ({total_pct}%)")

    # ── Section 3: Notable gaps ───────────────────────────────────────────────
    lines.append("")
    lines.append("## Notable Gaps (MLX kernels not yet benchmarked)")
    lines.append("")
    lines.append("Selected high-value kernels from each op that have no MT equivalent yet:")
    lines.append("")

    GAP_FILTERS = {
        "unary.metal":   lambda k: "float32float32" in k and k.startswith("v_"),
        "binary.metal":  lambda k: k.startswith("vvn_") and "float32" in k,
        "reduce.metal":  lambda k: "float32" in k and not k.startswith(("all_", "row_")),
        "softmax.metal": lambda k: "float32" in k and "precise" in k,
        "rope.metal":    lambda k: "float32" in k or "rope_single" in k,
        "gemv.metal":    lambda k: "float32" in k and "bm8" in k,
    }

    for metal_rel, filt in sorted(GAP_FILTERS.items()):
        full_mf = mf_map.get(metal_rel)
        if not full_mf:
            continue
        all_k = metal_kernels(full_mf)
        benched = mf_benched.get(metal_rel, set())
        gaps = [k for k in sorted(all_k - benched) if filt(k)][:5]
        if gaps:
            lines.append(f"**{metal_rel}**: " + ", ".join(f"`{k}`" for k in gaps))

    print("\n".join(lines))


if __name__ == "__main__":
    main()
