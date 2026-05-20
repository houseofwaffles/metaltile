# Publishing

How a release moves from `dev` to a tagged build on `main`. Maintainer-facing.

## Branch flow

`dev` is the integration branch; `main` holds stable, tagged releases only. Feature work merges into `dev` (see [Developing → branching](developing.md#branching-model)); a release is a single `dev` → `main` promotion.

## Cutting a release

1. **Confirm `dev` is green.** All [CI checks](testing.md#ci-vs-local) passing, and the changelog-worthy PRs for this release are merged.
2. **Open the release PR.** `dev` → `main`, titled `release: v0.x.0`. Merge it with a **merge commit** (not squash) so the history is preserved.
3. **Tag `main`** immediately after the merge:
   ```bash
   git checkout main
   git pull
   git tag v0.1.0
   git push origin v0.1.0
   ```
4. **Create the GitHub release:**
   ```bash
   gh release create v0.1.0 --generate-notes
   ```
   `.github/release.yml` categorises the merged PRs into release-notes sections automatically, using the conventional-commit prefix on each PR title — which is why [the prefix is enforced](developing.md#conventional-commits).

## What we don't do (yet)

- **No MSRV policy.** The workspace tracks Rust nightly for `edition = 2024` and unstable `rustfmt` features. An MSRV will be declared once the project stabilises on a stable compiler.
- **No backport branches.** All fixes land on `dev` and ride the next release. A critical hotfix can be handled by cutting a `v0.x` branch retroactively if it ever becomes necessary.
