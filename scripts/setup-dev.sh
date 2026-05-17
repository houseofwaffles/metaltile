#!/bin/bash
# setup-dev.sh
# One-time development environment setup for metaltile.
# Verifies toolchains, resolves dependencies, runs first build.
#
#   ./scripts/setup-dev.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}  ✓${NC} $1"; }
warn() { echo -e "${YELLOW}  ⚠${NC}  $1"; }
fail() { echo -e "${RED}  ✗${NC} $1"; exit 1; }

echo ""
echo "Setting up metaltile development environment..."
echo ""

# ─────────────────────────────────────────────
# Prerequisites
# ─────────────────────────────────────────────
echo "Checking prerequisites..."

if ! command -v rustup &>/dev/null; then
    fail "rustup not found. Install Rust via https://rustup.rs/"
fi
ok "rustup: $(rustup --version 2>&1 | head -1)"

if ! command -v cargo &>/dev/null; then
    fail "Cargo not found. Install Rust via https://rustup.rs/"
fi
ok "Cargo: $(cargo --version)"

# Verify nightly toolchain (metaltile uses edition 2024)
if ! rustup show active-toolchain | grep -q "nightly"; then
    warn "Active toolchain is not nightly. metaltile requires nightly for edition=2024."
    echo ""
    echo "Install with:"
    echo "  rustup toolchain install nightly"
    echo "  rustup component add rustfmt clippy --toolchain nightly"
    echo ""
    fail "nightly toolchain required"
fi
ok "Rust nightly toolchain: active"

if ! rustup component list --toolchain nightly | grep -q "rustfmt.*installed"; then
    warn "rustfmt not installed for nightly"
    rustup component add rustfmt --toolchain nightly || fail "Failed to install rustfmt"
fi
ok "rustfmt (nightly): installed"

if ! rustup component list --toolchain nightly | grep -q "clippy.*installed"; then
    warn "clippy not installed for nightly"
    rustup component add clippy --toolchain nightly || fail "Failed to install clippy"
fi
ok "clippy (nightly): installed"

# Optional: typos checker (used in CI)
if ! command -v typos &>/dev/null; then
    warn "typos-cli not found. Install with: cargo install typos-cli"
else
    ok "typos-cli: $(typos --version 2>&1 | head -1)"
fi

# Optional: cargo-llvm-cov (for coverage reports)
if ! command -v cargo-llvm-cov &>/dev/null; then
    warn "cargo-llvm-cov not found. Install with: cargo install cargo-llvm-cov"
else
    ok "cargo-llvm-cov: installed"
fi

# For Metal runtime tests on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! xcode-select -p &>/dev/null; then
        warn "Xcode Command Line Tools not found. Metal runtime tests require them."
    else
        ok "Xcode CLI tools: $(xcode-select -p)"
    fi
fi

# ─────────────────────────────────────────────
# Resolve packages
# ─────────────────────────────────────────────
echo ""
echo "Resolving Rust package dependencies..."
cd "$PROJECT_ROOT"
cargo fetch
ok "Packages resolved"

# ─────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────
echo ""
echo "Building workspace (debug)..."
cargo build --workspace
ok "Build complete"

# ─────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────
echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Common targets:"
echo "  make test       # run all tests"
echo "  make clippy     # lint with clippy"
echo "  make fmt        # format with rustfmt"
echo "  make fmt-check  # check formatting without modifying"
echo "  make coverage   # test coverage report (requires cargo-llvm-cov)"
echo "  make bench      # run benchmark suite (requires macOS + Metal)"
echo "  make clean      # remove build artifacts"
echo ""
