#!/usr/bin/env bash
# Builds the Rust backbone to WASM and places it in static/wasm/.
set -euo pipefail
cd "$(dirname "$0")/wasm"
source "$HOME/.cargo/env" 2>/dev/null || true
wasm-pack build --release --target web --out-dir ../static/wasm
echo "WASM built into static/wasm/"
