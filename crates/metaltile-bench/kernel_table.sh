#!/usr/bin/env bash
# Generate a table of all Metal kernels from reference files,
# cross-referenced with DSL ops.  macOS grep compatible.
set -euo pipefail

cd "$(dirname "$0")"

METAL_DIR="src/metal"
OP_DIR="src/ops"

declare -A IMPL=()

# ── Extract MLX reference kernel names from each DSL op ──
for opf in $(find "$OP_DIR" -name '*.rs' ! -name 'mod.rs' ! -name 'shared.rs'); do
    opname="${opf#$OP_DIR/}"
    opname="${opname%.rs}"
    
    # Get all quoted strings in compile(SRC, "...") calls
    refs=$(grep -h 'compile.*SRC' "$opf" 2>/dev/null | tr '"' '\n' | grep -v '^$' | grep -v '^[[:space:]]*$' | grep -v 'format!\|^SRC$\|^mt_') || true
    
    for ref in $refs; do
        # Skip format! leftovers, variable refs, etc.
        [[ "$ref" == *"format!"* ]] && continue
        [[ "$ref" == *"{"* ]] && continue
        [[ "$ref" == "SRC" ]] && continue
        IMPL["$ref"]="$opname"
    done
done

echo "# Kernel Coverage Report"
echo

# ── Main metal files ──
printf "%-38s %-50s %-42s %s\n" "Metal File" "MLX Template Functions" "Op File" "Status"
printf "%-38s %-50s %-42s %s\n" "----------" "----------------------" "-------" "------"

covered=0
for mf in "$METAL_DIR"/*.metal; do
    mbase="$(basename "$mf")"
    opf="${mbase%.metal}"
    
    [ "$mbase" = "fence.metal" ] && printf "%-38s %-50s %-42s %s\n" "$mbase" "—" "—" "SKIP (empty)" && continue

    # kernel template functions
    kfuncs=$(grep -E '^\[\[kernel\]\] void' "$mf" 2>/dev/null | awk '{print $3}' | paste -s -d ',' - | sed 's/,/, /g')
    [ -z "$kfuncs" ] && kfuncs="—"

    # instantiated kernel names
    instances=$(grep -n 'instantiate_kernel' "$mf" 2>/dev/null | grep -v '#define\|//' | \
        sed -n 's/.*instantiate_kernel([[:space:]]*"\([^"]*\)".*/\1/p' | sort -u)
    
    ninst=$(echo "$instances" | wc -l | tr -d ' ')

    # check DSL
    dsl_refs=""
    for ref in "${!IMPL[@]}"; do
        [ "${IMPL[$ref]}" = "$opf" ] && dsl_refs="$dsl_refs $ref"
    done
    dsl_refs=$(echo "$dsl_refs" | xargs)

    if [ -f "$OP_DIR/${opf}.rs" ] && [ -n "$dsl_refs" ]; then
        if [ "$ninst" -gt 0 ]; then
            n_reported=$(echo "$dsl_refs" | wc -w | tr -d ' ')
            status="PARTIAL ($n_reported/$ninst) $dsl_refs"
        else
            status="IMPL: $dsl_refs"
            covered=$((covered + 1))
        fi
    elif [ "$ninst" -gt 0 ]; then
        first=$(echo "$instances" | head -1)
        status="NYI ($ninst kernels, e.g. $first)"
    else
        status="no instantiations"
    fi

    printf "%-38s %-50s %-42s %s\n" "$mbase" "$kfuncs" "${opf}.rs" "$status"
done

echo
echo "## Steel/"
echo

printf "%-38s %-50s %-42s %s\n" "Metal File" "MLX Template Functions" "Op File" "Status"
printf "%-38s %-50s %-42s %s\n" "----------" "----------------------" "-------" "------"

for mf in $(find "$METAL_DIR"/steel -name '*.metal'); do
    mbase="$(basename "$mf")"
    rel="${mf#$METAL_DIR/steel/}"
    rel_dir="${rel%/*}"
    opf_stub="${mbase%.metal}"
    full_opf="steel/$rel_dir/$opf_stub"

    kfuncs=$(grep -E '^\[\[kernel\]\] void' "$mf" 2>/dev/null | awk '{print $3}' | paste -s -d ',' - | sed 's/,/, /g')
    [ -z "$kfuncs" ] && kfuncs="—"

    instances=$(grep -n 'instantiate_kernel' "$mf" 2>/dev/null | grep -v '#define\|//' | \
        sed -n 's/.*instantiate_kernel([[:space:]]*"\([^"]*\)".*/\1/p' | sort -u)
    
    ninst=$(echo "$instances" | wc -l | tr -d ' ')

    dsl_refs=""
    for ref in "${!IMPL[@]}"; do
        [ "${IMPL[$ref]}" = "$full_opf" ] && dsl_refs="$dsl_refs $ref"
    done
    dsl_refs=$(echo "$dsl_refs" | xargs)

    if [ -f "$OP_DIR/$full_opf.rs" ] && [ -n "$dsl_refs" ]; then
        if [ "$ninst" -gt 0 ]; then
            n_reported=$(echo "$dsl_refs" | wc -w | tr -d ' ')
            status="PARTIAL ($n_reported/$ninst) $dsl_refs"
        else
            status="IMPL: $dsl_refs"
            covered=$((covered + 1))
        fi
    elif [ "$ninst" -gt 0 ]; then
        first=$(echo "$instances" | head -1)
        status="NYI ($ninst kernels, e.g. $first)"
    else
        status="no instantiations"
    fi

    printf "%-38s %-50s %-42s %s\n" "$mbase" "$kfuncs" "$full_opf.rs" "$status"
done

echo
echo "## Full kernel listing per file"
echo

for mf in "$METAL_DIR"/*.metal $(find "$METAL_DIR"/steel -name '*.metal'); do
    mbase="$(basename "$mf")"
    instances=$(grep -n 'instantiate_kernel' "$mf" 2>/dev/null | grep -v '#define\|//' | \
        sed -n 's/.*instantiate_kernel([[:space:]]*"\([^"]*\)".*/\1/p' | sort -u)
    ninst=$(echo "$instances" | wc -l | tr -d ' ')
    if [ "$ninst" -gt 0 ] && [ "$ninst" -lt 100 ]; then
        echo "### $mbase ($ninst kernels)"
        for k in $instances; do
            imarker=""
            [ -n "${IMPL[$k]:-}" ] && imarker="  ← ${IMPL[$k]}"
            echo "  - $k$imarker"
        done
        echo
    elif [ "$ninst" -ge 100 ]; then
        echo "### $mbase ($ninst kernels — too many to list)"
        echo
    fi
done
