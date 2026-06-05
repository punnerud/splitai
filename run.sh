#!/usr/bin/env bash
# Starts the Django development server. Builds WASM first if it is missing.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f static/wasm/splitai_wasm_bg.wasm ]; then
  echo "WASM missing — building …"
  ./build_wasm.sh
fi

./venv/bin/python manage.py migrate --noinput
echo "Open http://127.0.0.1:8000/  (open in two different browsers/profiles to test sharing)"
exec ./venv/bin/python manage.py runserver 127.0.0.1:8000
