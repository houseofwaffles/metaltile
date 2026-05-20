//! Structural consistency tests for the metaltile-std kernel registry.
//!
//! Catches two classes of regression without touching the GPU:
//!
//! 1. **Silent missing-module / undefined symbol** — a downstream
//!    consumer hits a Metal link-time `undefined symbol: mt_xyz_smoke`
//!    error because the kernel was referenced from one place but the
//!    source module wasn't compiled in (e.g. forgotten `mod foo;`
//!    declaration). The codegen step here scans every emitted MSL for
//!    `mt_*` symbols and asserts each one is either:
//!    a. The name of another registered kernel (cross-kernel call),
//!    b. A DSL builtin emitted by the codegen preamble, or
//!    c. Defined locally in the same emit (template/inline function).
//!
//! 2. **Empty / unregistered kernel** — every `#[kernel] pub fn name`
//!    in `crates/metaltile-std/src/{mlx,ffai}/**` should have a matching
//!    `inventory::submit!` for a BenchSpec that references its
//!    `kernel_ir_for` function. PR #19 silently emptied a kernel body
//!    via macro refactor; the inverse failure (kernel defined but
//!    never registered) would slip through the build with no GPU
//!    dispatch ever happening.
//!
//! 3. **BenchSpec codegens** — for each `(spec, dtype)` in the
//!    inventory, calling `(spec.kernel_ir)(dt)` and feeding through the
//!    full `MslGenerator` pipeline must succeed and emit a
//!    `kernel void <name>` definition. A spec whose `kernel_ir` panics
//!    or returns an IR that fails type inference / pass pipeline would
//!    surface here.
//!
//! Runs on Ubuntu (host-side codegen, no Metal runtime needed) — this
//! is the only structural-consistency safety net for non-macOS CI.

use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use metaltile_codegen::{MslGenerator, msl::MslConfig};
use metaltile_std::spec::{BenchSpec, effective_mode};

// ── DSL builtin whitelist (mt_* and __mt_* symbols emitted by codegen) ──
//
// Confirmed against `crates/metaltile-codegen/src/msl/preamble.rs` and
// `crates/metaltile-core/src/ir.rs` — these are the only `mt_*` /
// `__mt_*` identifiers the codegen ever emits independently of a
// kernel-registered name. If the codegen grows a new helper, add it
// here; if it grows a new mt_* helper that wasn't intentional, this
// test surfaces the omission.
const DSL_BUILTINS: &[&str] = &[
    "mt_silu",
    "mt_gelu",
    "mt_relu",
    "mt_sigmoid",
    "mt_erf_impl",
    "mt_erfinv_impl",
    "mt_expm1_impl",
    "__mt_simd_product",
];

// ── Helpers ─────────────────────────────────────────────────────────────

/// Walk a directory recursively, returning all `.rs` file paths.
fn walk_rs_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// Extract `mt_X` / `__mt_X` identifiers from a string. Returns each
/// match as a `String`. Excludes substring-of-larger-ident matches by
/// requiring word boundaries on both sides.
fn extract_mt_symbols(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Skip ahead to the next 'm' that could start "mt_" or "__mt_".
        let start = i;
        let c = bytes[i];
        let is_ident_char = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
        if !is_ident_char(c) {
            i += 1;
            continue;
        }
        // Walk to the end of this identifier.
        let mut j = i;
        while j < bytes.len() && is_ident_char(bytes[j]) {
            j += 1;
        }
        // Identifier is bytes[start..j]; check it begins with "mt_" or
        // "__mt_". We require the *whole* identifier match the pattern
        // (no leading alphanumerics), enforced by requiring `start` is
        // either 0 or preceded by a non-ident byte.
        let prev_is_boundary = start == 0 || !is_ident_char(bytes[start - 1]);
        if prev_is_boundary {
            let ident = &src[start..j];
            if (ident.starts_with("mt_") || ident.starts_with("__mt_")) && ident.len() > 3 {
                out.push(ident.to_string());
            }
        }
        i = j;
        if i == start {
            i += 1;
        }
    }
    out
}

/// Locally-defined identifier in the emitted MSL — matches names that
/// appear after `inline` / `template<...>\n... ` / `struct` keywords
/// in the MSL emit (preamble helpers, BF16 struct, simdgroup helpers,
/// kernel-emitted templates, etc.). Build the set lazily per-emit.
fn collect_locally_defined(msl: &str) -> HashSet<String> {
    let mut defs = HashSet::new();
    // Walk lines; look for "inline ... <name>(" patterns or
    // "kernel void <name>(" or "struct <name>".
    for line in msl.lines() {
        let trimmed = line.trim_start();
        // inline T foo(...) / inline float foo(...) / inline void foo(...) ...
        if let Some(rest) = trimmed.strip_prefix("inline") {
            // Expect 2+ tokens: <ret-type> <name>(...
            let after_spaces = rest.trim_start();
            // Find the last identifier before `(`.
            if let Some(paren_at) = after_spaces.find('(') {
                let prefix = &after_spaces[..paren_at];
                if let Some(name) = prefix.split_whitespace().last() {
                    defs.insert(name.trim_start_matches('&').to_string());
                }
            }
        }
        // kernel void foo(
        if let Some(rest) = trimmed.strip_prefix("kernel ")
            && let Some(paren_at) = rest.find('(')
        {
            let prefix = &rest[..paren_at];
            if let Some(name) = prefix.split_whitespace().last() {
                defs.insert(name.to_string());
            }
        }
        // struct foo {
        if let Some(rest) = trimmed.strip_prefix("struct ") {
            let name: String =
                rest.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
            if !name.is_empty() {
                defs.insert(name);
            }
        }
        // Forward decls / `template<typename T>` followed by an inline:
        // skip — the next line's `inline` will catch it.
    }
    defs
}

/// Strip C-style line comments (`// ...`) from a source string. Used so
/// `extract_mt_symbols` doesn't pick up symbol names mentioned in
/// doc-comments / inline comments. Block comments are not handled
/// (rare in the kernel source + a doc-block mention is itself a
/// boundary signal we'd want to surface anyway).
fn strip_line_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    for line in src.lines() {
        // Find first `//` that is not inside a string literal. For our
        // codebase, kernel sources don't put `//` inside strings, so a
        // plain find is safe enough.
        let cut = line.find("//");
        match cut {
            Some(idx) => {
                out.push_str(&line[..idx]);
                out.push('\n');
            },
            None => {
                out.push_str(line);
                out.push('\n');
            },
        }
    }
    out
}

// ── Test 1: every registered BenchSpec codegens cleanly ─────────────────

#[test]
fn every_registered_benchspec_codegens() {
    let mut errors: Vec<String> = Vec::new();
    let mut total = 0_usize;

    for spec in inventory::iter::<BenchSpec>() {
        let mode = effective_mode(spec);
        for &dt in spec.dtypes {
            total += 1;
            let mut kernel = (spec.kernel_ir)(dt);
            kernel.mode = mode;

            let generator = MslGenerator::new(MslConfig::default());
            match generator.generate(&kernel) {
                Ok(msl) => {
                    // Emitted MSL must define the kernel under its
                    // declared name — otherwise the bench/build wiring
                    // would dispatch into a kernel that doesn't exist.
                    // Note: codegen monomorphizes the kernel symbol per
                    // dtype only when the runtime build does so; here we
                    // just check the kernel's `name` field appears as a
                    // `kernel void <name>(` somewhere in the emit.
                    let expected_token = format!("kernel void {}", kernel.name);
                    if !msl.contains(&expected_token) {
                        errors.push(format!(
                            "spec {}/{} kernel_name={} dt={:?}: emitted MSL does not \
                             contain `{expected_token}`",
                            spec.op, spec.subop, spec.kernel_name, dt,
                        ));
                    }
                },
                Err(e) => {
                    errors.push(format!(
                        "spec {}/{} kernel_name={} dt={:?}: codegen failed — {e:?}",
                        spec.op, spec.subop, spec.kernel_name, dt,
                    ));
                },
            }
        }
    }

    assert!(total > 0, "inventory::iter::<BenchSpec>() was empty — link issue?");
    assert!(
        errors.is_empty(),
        "{} of {} (spec, dtype) cells failed codegen:\n  {}",
        errors.len(),
        total,
        errors.join("\n  ")
    );
}

// ── Test 2: no undefined `mt_*` / `__mt_*` symbols in emitted MSL ──────

#[test]
fn no_undefined_mt_symbols_in_emitted_msl() {
    // Build set of known kernel names from the registry.
    let kernel_names: HashSet<String> =
        inventory::iter::<BenchSpec>().map(|s| s.kernel_name.to_string()).collect();
    assert!(!kernel_names.is_empty(), "inventory empty — link issue?");

    let builtins: HashSet<&'static str> = DSL_BUILTINS.iter().copied().collect();

    let mut errors: Vec<String> = Vec::new();

    for spec in inventory::iter::<BenchSpec>() {
        let mode = effective_mode(spec);
        for &dt in spec.dtypes {
            let mut kernel = (spec.kernel_ir)(dt);
            kernel.mode = mode;
            let generator = MslGenerator::new(MslConfig::default());
            let msl = match generator.generate(&kernel) {
                Ok(m) => m,
                Err(_) => continue, // test 1 reports codegen failures
            };

            let local_defs = collect_locally_defined(&msl);
            let symbols = extract_mt_symbols(&msl);
            let mut seen: HashSet<String> = HashSet::new();
            for sym in symbols {
                if !seen.insert(sym.clone()) {
                    continue;
                }
                let known = kernel_names.contains(&sym)
                    || builtins.contains(sym.as_str())
                    || local_defs.contains(&sym)
                    // The emitted kernel always defines its own name —
                    // accept it as locally-defined even if `kernel void`
                    // line scanning misses an edge case.
                    || sym == kernel.name;
                if !known {
                    errors.push(format!(
                        "spec {}/{} kernel_name={} dt={:?}: emitted MSL references \
                         `{sym}` but no registered kernel, DSL builtin, or local \
                         definition matches",
                        spec.op, spec.subop, spec.kernel_name, dt,
                    ));
                }
            }
        }
    }

    assert!(
        errors.is_empty(),
        "{} undefined-symbol references found in emitted MSL:\n  {}",
        errors.len(),
        errors.join("\n  ")
    );
}

// ── Test 3: every `#[kernel] pub fn name` has a matching BenchSpec ─────

#[test]
fn kernel_annotations_have_matching_inventory_submit() {
    // Collect kernel-named registry entries (kernel_name → matched-via-name)
    // PLUS kernel_ir function paths (e.g. `softmax_categorical_sample::kernel_ir_for`)
    // so we can match either form. The macro expands `#[kernel]` to
    // give each function a `kernel_ir_for(dt) -> Kernel` associated
    // function inside a module of the same name; `inventory::submit!`
    // entries typically point to that via `<name>::kernel_ir_for`.
    let mut registered_kernel_names: HashSet<String> = HashSet::new();
    for spec in inventory::iter::<BenchSpec>() {
        registered_kernel_names.insert(spec.kernel_name.to_string());
    }
    assert!(!registered_kernel_names.is_empty(), "inventory empty — link issue?");

    // Find the metaltile-std source tree relative to this test file. The
    // file lives at `crates/metaltile-std/tests/kernel_registry_consistency.rs`;
    // `CARGO_MANIFEST_DIR` points at `crates/metaltile-std`.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let src_root = Path::new(manifest_dir).join("src");
    // Restrict the walk to mlx + ffai (where actual kernels live) to
    // avoid scanning spec.rs / run_spec.rs glue files.
    let scan_dirs = ["mlx", "ffai"];

    // Track (kernel_name, source_file_path) for the second-pass check
    // that looks up `kernel_ir: <name>::kernel_ir_for` in inventory
    // submit blocks.
    let mut annotated: Vec<(String, PathBuf)> = Vec::new();

    for sub in &scan_dirs {
        let root = src_root.join(sub);
        if !root.exists() {
            continue;
        }
        for path in walk_rs_files(&root) {
            let src = match fs::read_to_string(&path) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let cleaned = strip_line_comments(&src);
            // Look for `#[kernel]` immediately followed (possibly with
            // attribute lines / generics) by `pub fn <name>(`. Be
            // permissive about whitespace + intervening attributes.
            let mut idx = 0;
            while let Some(at) = cleaned[idx..].find("#[kernel]") {
                let pos = idx + at;
                // Within the next ~400 bytes we should see `pub fn <name>`.
                let window_end = (pos + 400).min(cleaned.len());
                let window = &cleaned[pos..window_end];
                if let Some(fn_at) = window.find("pub fn ") {
                    let after = &window[fn_at + "pub fn ".len()..];
                    let name: String = after
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    if !name.is_empty() {
                        annotated.push((name, path.clone()));
                    }
                }
                idx = pos + "#[kernel]".len();
            }
        }
    }

    assert!(
        !annotated.is_empty(),
        "no `#[kernel]` annotations found under `src/{{mlx,ffai}}` — walker broken?"
    );

    // Now: for each annotated kernel, it should match a registered
    // kernel either by `kernel_name` directly OR by appearing as
    // `<name>::kernel_ir_for` in the inventory submit text in its
    // source file. The first check is exact; the second is a
    // static string-match fallback for the rare case where the
    // registered `kernel_name` is monomorphized off the original
    // (e.g. `mt_argmax_f32` registers but the source fn is `mt_argmax`).
    let mut errors: Vec<String> = Vec::new();
    for (name, path) in &annotated {
        if registered_kernel_names.contains(name) {
            continue;
        }
        // Fallback: look in the source file for an
        // `inventory::submit!` referencing `<name>::kernel_ir_for`.
        let src = fs::read_to_string(path).unwrap_or_default();
        let needle_a = format!("{name}::kernel_ir_for");
        let needle_b = format!("kernel_ir: {name}::kernel_ir_for");
        if src.contains(&needle_a) || src.contains(&needle_b) {
            continue;
        }
        errors.push(format!(
            "`#[kernel] pub fn {name}` in {} has no matching registered BenchSpec \
             (neither by `kernel_name = \"{name}\"` nor by `{name}::kernel_ir_for` \
             reference) — likely missing `inventory::submit!`",
            path.display(),
        ));
    }

    assert!(
        errors.is_empty(),
        "{} `#[kernel]` annotations are unregistered:\n  {}",
        errors.len(),
        errors.join("\n  ")
    );
}
