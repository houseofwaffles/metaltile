#!/usr/bin/env bash
# Generate workspace test-coverage report locally and open the HTML view.
#
# Mirrors the ignore-pattern + flags used by the CI Coverage workflow so the
# local numbers match what Codecov sees on the PR.
#
# Usage:
#   ./.github/scripts/coverage.sh           # html report, opens in browser
#   ./.github/scripts/coverage.sh summary   # text summary only (CI-style)
#   ./.github/scripts/coverage.sh lcov      # emit lcov.info for editor integrations

set -euo pipefail

cd "$(dirname "$0")/../.."

if ! command -v cargo-llvm-cov >/dev/null 2>&1; then
  echo "error: cargo-llvm-cov not installed."
  echo "  brew install cargo-llvm-cov"
  echo "  # or: cargo install cargo-llvm-cov"
  exit 1
fi

IGNORE='(crates/metaltile/src/(lib|prelude)\.rs|crates/metaltile-std/src/(ffai|mlx)/.*\.rs|tests/|benches/|build\.rs)'

case "${1:-html}" in
  summary)
    cargo llvm-cov --workspace --summary-only \
      --ignore-filename-regex "$IGNORE"
    ;;
  lcov)
    cargo llvm-cov --workspace --lcov \
      --output-path lcov.info \
      --ignore-filename-regex "$IGNORE"
    echo "wrote lcov.info"
    ;;
  html|*)
    cargo llvm-cov --workspace --html \
      --output-dir target/llvm-cov/html \
      --ignore-filename-regex "$IGNORE"
    echo "report: target/llvm-cov/html/index.html"
    if command -v open >/dev/null 2>&1; then
      open target/llvm-cov/html/index.html
    fi
    ;;
esac
