# MetalTile Documentation

Table of contents for the MetalTile docs. The top-level [`README`](../README.md) is the curated landing page; this index lists every page so you can jump straight to a topic. New contributors (and their agents) should read [Getting started](getting-started.md) → [Developing](developing.md) → [Testing](testing.md) before opening a PR.

## Getting started

- [Getting started](getting-started.md) — toolchain, clone, first build, first kernel.

## Local development

- [Developing](developing.md) — repo layout, the `make` dev loop, branching, commits, debugging, and the **kernel-authoring hazards** (⚠️) that cause silent or catastrophic failure. Required reading before writing a kernel.
- [Testing](testing.md) — the test layers, what runs in CI vs locally, how to write a test, coverage targets, and the gaps in the test infrastructure that let bugs through silently.
- [Publishing](publishing.md) — the `dev` → `main` release flow.

## Reference

- [CLI](cli.md) — the `tile` binary: `bench`, `build`, `inspect`, `device`, `snap`, `diff`.

## See also

- Top-level [`README`](../README.md) — project landing page.
- [`CONTRIBUTING`](../CONTRIBUTING.md) — issue / PR process, agentic-contribution disclosure, code of conduct.
