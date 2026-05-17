## Proposed changes

Please describe the problem or feature this PR addresses. Link any
relevant issue with `#<issue-number>`.

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

- [ ] PR title uses a conventional-commit prefix (see below)
- [ ] `make clippy` passes clean
- [ ] `make test --workspace` passes
- [ ] `make fmt-check` passes
- [ ] `make typos` passes (or `typos` is clean)
- [ ] PR body explains **what** and **why**

## Conventional commit prefix

PR title prefix is used by `auto-label.yml` for release-notes
categorization. Use one of:

`feat: …` `fix: …` `perf: …` `docs: …` `test: …`
`chore: …` `ci: …` `build: …` `refactor: …` `style: …`

Add `!` for breaking changes (`feat!: …`).
