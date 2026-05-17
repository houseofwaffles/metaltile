#!/bin/bash
# coverage.sh
# Run tests with coverage and print a summary.
# Requires cargo-llvm-cov:  cargo install cargo-llvm-cov

set -e

cd "$(dirname "$0")/.."

if ! command -v cargo-llvm-cov &>/dev/null; then
    echo "cargo-llvm-cov not found. Install with:"
    echo "  cargo install cargo-llvm-cov"
    exit 1
fi

echo "Running tests with coverage enabled..."
cargo llvm-cov --workspace --html --open

echo ""
echo "Coverage report generated. Open target/llvm-cov/html/index.html"
