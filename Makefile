# MetalTile — Makefile
#
# Common dev-loop targets. See TOOLCHAIN_PLAN.md for the phased
# build-out and scripts/ for the longer-form scripts.

.DEFAULT_GOAL := help

# ─── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT := $(shell pwd)

# ─── Help ─────────────────────────────────────────────────────────────
.PHONY: help
help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ─── Setup ────────────────────────────────────────────────────────────
.PHONY: setup
setup: ## one-time dev environment setup (toolchains, deps, first build)
	./.github/scripts/setup-dev.sh

# ─── Build ────────────────────────────────────────────────────────────
.PHONY: build
build: ## cargo build (debug)
	cargo build --workspace

.PHONY: build-release
build-release: ## cargo build (release)
	cargo build --workspace --release

# ─── Test ─────────────────────────────────────────────────────────────
.PHONY: test
test: ## cargo test --workspace
	cargo test --workspace

.PHONY: coverage
coverage: ## test coverage report (requires cargo-llvm-cov)
	./scripts/coverage.sh

# ─── Lint / format ────────────────────────────────────────────────────
.PHONY: clippy
clippy: ## run clippy on all targets with -D warnings
	cargo clippy --all-targets --all-features -- -D warnings

.PHONY: fmt
fmt: ## run rustfmt on all crates
	cargo fmt --all

.PHONY: fmt-check
fmt-check: ## check formatting without modifying files
	cargo fmt --all -- --check

.PHONY: typos
typos: ## run typos checker
	typos

# ─── tile CLI ─────────────────────────────────────────────────────────
#
# All targets call into the `tile` CLI via `cargo run --release`,
# which handles incremental rebuilds itself — no explicit
# `build-release` dependency needed. The `-q` quiets cargo's own
# "Compiling … / Finished" lines so the CLI output is unobstructed.
#
# **Two entry points, no overlap**:
#
#   1. `make tile <subcommand> …`
#       — the universal passthrough. Use for any bare CLI call:
#           make tile bench
#           make tile snap
#           make tile diff
#           make tile device
#           make tile inspect aura_encode_int4
#       — args starting with `-` confuse make's option parser, so
#         use `make tile-args ARGS="bench -vv --filter sdpa"` for
#         those. (`-vv` / `--filter …` as bare make args don't work.)
#
#   2. Named wrappers below — pre-baked flag combos that would be
#      awkward to type via the passthrough:
#           make bench-v / bench-vv          (occupancy / GPU timing)
#           make inspect-stats KERNEL=foo    (--stats)
#           make inspect-ir KERNEL=foo       (--ir)
#           make inspect-list                (--all)
#           make time-passes                 (--time-passes)
#           make emit-all OUT=…              (--emit all -o …)
#           make snapshots-{review,accept,pending}  (cargo-insta loops)
#
# The passthrough deliberately does NOT have collision-prone wrappers
# (no `make bench`, no `make snap`, no `make device`). Use the
# passthrough form for those.

ARGS ?=
KERNEL ?=
OUT ?=

# Named wrappers for flag combos that aren't ergonomic via passthrough.
.PHONY: bench-v bench-vv
bench-v: ## tile bench -v — adds occupancy + register-pressure profile
	cargo run --release -q -p metaltile-cli -- bench -v $(ARGS)
bench-vv: ## tile bench -vv — adds GPU timing stats (min µs + bandwidth)
	cargo run --release -q -p metaltile-cli -- bench -vv $(ARGS)

.PHONY: inspect-stats inspect-ir inspect-list
inspect-stats: ## tile inspect KERNEL=<name> --stats — per-pass op-count deltas
	cargo run --release -q -p metaltile-cli -- inspect $(KERNEL) --stats $(ARGS)
inspect-ir: ## tile inspect KERNEL=<name> --ir — raw IR before passes
	cargo run --release -q -p metaltile-cli -- inspect $(KERNEL) --ir $(ARGS)
inspect-list: ## tile inspect --all — list every registered kernel
	cargo run --release -q -p metaltile-cli -- inspect --all $(ARGS)

.PHONY: emit-all time-passes
emit-all: ## tile build --emit all OUT=<dir> — codegen for FFAI consumption
	@if [ -z "$(OUT)" ]; then \
	  echo "Error: set OUT=<dir>, e.g. make emit-all OUT=../FFAI/Sources/MetalTileSwift"; \
	  exit 1; \
	fi
	cargo run --release -q -p metaltile-cli -- build --emit all -o $(OUT) $(ARGS)
time-passes: ## tile build --time-passes — wall-clock per codegen pass
	cargo run --release -q -p metaltile-cli -- build --time-passes $(ARGS)

# ─── insta MSL snapshot loop ──────────────────────────────────────────
.PHONY: snapshots snapshots-review snapshots-accept snapshots-pending
snapshots: ## cargo test (snapshots fail on drift)
	cargo test --workspace
snapshots-review: ## cargo insta review — interactive snapshot accept (interactive!)
	cargo insta review
snapshots-accept: ## cargo insta test --accept — accept ALL pending snapshots
	cargo insta test --accept --workspace
snapshots-pending: ## cargo insta pending-snapshots — list pending without accepting
	cargo insta pending-snapshots

# ─── tile passthrough escape hatch ────────────────────────────────────
#
# Examples:
#   make tile bench
#   make tile snap
#   make tile diff
#   make tile device
#   make tile inspect aura_encode_int4
#
# For args starting with `-` (which make tries to consume), use:
#   make tile-args ARGS="bench -vv --filter sdpa_decode"
#   make tile-args ARGS="inspect aura_encode_int4 --stats --dtype bf16"
#
# The catch-all `%:` rule is gated to only fire when `tile` is the
# first goal — so trailing words like `bench` / `snap` after `make tile`
# become no-op targets (just args to the cargo command), while typos
# elsewhere (e.g. `make typotypo`) still error normally.
.PHONY: tile tile-args
tile: ## tile passthrough: `make tile <subcommand>` (use tile-args for flags)
	@cargo run --release -q -p metaltile-cli -- $(filter-out tile,$(MAKECMDGOALS))
tile-args: ## tile passthrough with flags: `make tile-args ARGS="bench -vv --filter sdpa"`
	@cargo run --release -q -p metaltile-cli -- $(ARGS)

ifeq (tile,$(firstword $(MAKECMDGOALS)))
%:
	@:
endif

# ─── Clean ────────────────────────────────────────────────────────────
.PHONY: clean
clean: ## remove target/ and build artifacts
	cargo clean
