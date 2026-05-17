# Contributing to MetalTile

Thanks for helping make MetalTile better. This doc covers the branching
model, PR process, release flow, and local dev setup.

## Branching model

| Branch | Purpose |
|---|---|
| `main` | Stable releases only. Commits here are tagged (`v0.1.0`, `v0.2.0`, …). |
| `dev`  | Integration branch for the next release. Feature PRs merge here first. |
| `feat/*` `fix/*` `perf/*` | Short-lived topic branches cut from `dev`. |

**Flow:**
1. Cut a topic branch from `dev`.
2. Open a PR targeting `dev`.
3. After review + CI green, squash-merge (or rebase-merge) to `dev`.
4. When `dev` is release-ready, open a PR `dev → main`. Merge with a merge
   commit so the history is preserved.
5. Tag `main` immediately after the merge: `git tag v0.1.0` and push.

## Conventional commits

PR titles must follow [Conventional Commits](https://www.conventionalcommits.org/)
so that `.github/workflows/auto-label.yml` can categorize them for release
notes and `.github/workflows/pr.yml` can validate the format.

```
feat: add softmax vector path for small N
fix(codegen): correct version gate for half2 stores
perf(runtime): cache PSO lookups by function signature
docs: update CLI install instructions
test(interp): add scan correctness test
chore: bump nightly toolchain
ci: add release-notes auto-label workflow
```

Add `!` for breaking changes: `feat!: remove deprecated Tensor::from_raw`.
Include `BREAKING CHANGE` in the PR body for details.

## One-time setup

```bash
git clone git@github.com:0xClandestine/metaltile.git
cd metaltile
./.github/scripts/setup-dev.sh
```

This verifies:
- Rust nightly toolchain (`rust-toolchain.toml`)
- `rustfmt` and `clippy` components
- Optional: `typos-cli` and `cargo-llvm-cov`

## Dev loop

```bash
make build            # debug build
make test             # workspace tests (includes interpreter + GPU if on Mac)
make clippy           # lint
make fmt              # format
make fmt-check        # check format without touching files
make coverage         # html coverage report (needs cargo-llvm-cov)
make bench            # full benchmark suite vs MLX (macOS + Metal)
make clean            # remove target/
```

`make` is preferred over raw `cargo` because it centralises flags and
ensures `--workspace` is always passed.

## PR checklist

Before requesting review:

- [ ] Title uses conventional-commit prefix (`feat:`, `fix:`, `perf:`, …).
- [ ] `make clippy` passes clean (`-D warnings`).
- [ ] `make test` passes.
- [ ] `make fmt-check` passes.
- [ ] `make typos` passes (or `typos` is installed and clean).
- [ ] PR body explains **what** and **why**. Link issues with `#<num>`.
- [ ] If bench numbers changed, paste relevant rows in the PR body.

## Release process (maintainers)

1. Ensure `dev` is green and changelog-worthy PRs are merged.
2. Open PR `dev → main`. Title: `release: v0.x.0`.
3. After merge, tag on `main`:
   ```bash
   git checkout main
   git pull
   git tag v0.1.0
   git push origin v0.1.0
   ```
4. Create the GitHub release:
   ```bash
   gh release create v0.1.0 --generate-notes
   ```
   `.github/release.yml` categorises merged PRs automatically.

## CI

`.github/workflows/check.yml` runs on every push:
- `typos` — spell check
- `clippy` — lint with `-D warnings`
- `cargo test --workspace`

`.github/workflows/pr.yml` validates the PR title format.

`.github/workflows/auto-label.yml` applies release-notes labels based on
the conventional-commit prefix in the PR title.

## What we don't do (yet)

- **MSRV policy.** We track nightly for edition=2024 and unstable
  rustfmt features. An MSRV will be declared once we stabilise on a
  stable compiler.
- **Backport branches.** All fixes land in `dev` and ride the next
  release. If a critical hotfix is needed, we can cut a `v0.x` branch
  retroactively.

## License

By contributing, you agree that your contributions will be licensed
under the Apache License, Version 2.0.
