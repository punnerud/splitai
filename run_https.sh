#!/usr/bin/env bash
# Starter SplitAI med HTTPS (selvsignert) slik at webkamera virker på telefon
# og andre enheter over LAN-et. Genererer sertifikat med riktig LAN-IP i SAN.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-8443}"
IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 127.0.0.1)"
CERT="certs/cert.pem"
KEY="certs/key.pem"
mkdir -p certs

# Regenerer sertifikat hvis det mangler eller IP-en ikke er dekket.
if [ ! -f "$CERT" ] || ! openssl x509 -in "$CERT" -noout -text 2>/dev/null | grep -q "$IP"; then
  echo "Genererer selvsignert sertifikat for IP:$IP …"
  openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
    -keyout "$KEY" -out "$CERT" \
    -subj "/CN=splitai-local" \
    -addext "subjectAltName=IP:${IP},DNS:localhost,IP:127.0.0.1" >/dev/null 2>&1
fi

if [ ! -f static/wasm/splitai_wasm_bg.wasm ]; then ./build_wasm.sh; fi
./venv/bin/python manage.py migrate --noinput

echo
echo "  HTTPS klart. Åpne på enheter i nettet:"
echo "      https://${IP}:${PORT}/"
echo "  (Godta sertifikat-advarselen én gang per enhet — nødvendig for at"
echo "   webkamera skal virke. På din egen maskin funker også https://localhost:${PORT}/)"
echo
exec ./venv/bin/python manage.py runserver_plus 0.0.0.0:${PORT} \
  --cert-file "$CERT" --key-file "$KEY"
