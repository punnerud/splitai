#!/usr/bin/env bash
# Bygger Rust-backbone-en til WASM og legger den i static/wasm/.
set -euo pipefail
cd "$(dirname "$0")/wasm"
source "$HOME/.cargo/env" 2>/dev/null || true
wasm-pack build --release --target web --out-dir ../static/wasm
echo "WASM bygget til static/wasm/"
