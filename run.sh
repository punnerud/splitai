#!/usr/bin/env bash
# Starter Django-utviklingsserveren. Bygg WASM først hvis den mangler.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f static/wasm/splitai_wasm_bg.wasm ]; then
  echo "WASM mangler — bygger …"
  ./build_wasm.sh
fi

./venv/bin/python manage.py migrate --noinput
echo "Åpne http://127.0.0.1:8000/  (åpne i to ulike nettlesere/profiler for å teste deling)"
exec ./venv/bin/python manage.py runserver 127.0.0.1:8000
