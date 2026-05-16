## What

<!-- One-line summary of the change. -->

## Why

<!-- Motivation: bug fix, new op, perf improvement, refactor, etc. -->

## Crates affected

<!-- Check all that apply -->
- [ ] `metaltile-core` (IR types, ops)
- [ ] `metaltile-macros` (proc macros, body parser)
- [ ] `metaltile-codegen` (MSL lowering, passes)
- [ ] `metaltile-interp` (CPU reference interpreter)
- [ ] `metaltile-runtime` (Metal dispatch)
- [ ] `metaltile-std` (kernel stdlib, op files)
- [ ] `metaltile-cli` (`tile` binary)

## Testing

<!-- If bench numbers changed, paste relevant rows from `cargo bench` output. -->
<!-- Format: op | dtype | MT GB/s | MLX GB/s | MT% | correct? -->

## Checklist

- [ ] `cargo clippy --all-targets --all-features -- -D warnings` clean
- [ ] `cargo test --workspace` passes
- [ ] Typos clean (`typos`)
