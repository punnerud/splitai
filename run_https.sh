#!/usr/bin/env bash
# Starts SplitAI with HTTPS (self-signed) so the webcam works on phones and other
# devices over the LAN. Generates a certificate with the correct LAN IP in the SAN.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-8443}"
IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 127.0.0.1)"
CERT="certs/cert.pem"
KEY="certs/key.pem"
mkdir -p certs

# Regenerate the certificate if it is missing or the IP is not covered.
if [ ! -f "$CERT" ] || ! openssl x509 -in "$CERT" -noout -text 2>/dev/null | grep -q "$IP"; then
  echo "Generating self-signed certificate for IP:$IP …"
  openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
    -keyout "$KEY" -out "$CERT" \
    -subj "/CN=splitai-local" \
    -addext "subjectAltName=IP:${IP},DNS:localhost,IP:127.0.0.1" >/dev/null 2>&1
fi

if [ ! -f static/wasm/splitai_wasm_bg.wasm ]; then ./build_wasm.sh; fi
./venv/bin/python manage.py migrate --noinput

echo
echo "  HTTPS ready. Open on devices on the network:"
echo "      https://${IP}:${PORT}/"
echo "  (Accept the certificate warning once per device — required for the"
echo "   webcam to work. On your own machine https://localhost:${PORT}/ also works.)"
echo
exec ./venv/bin/python manage.py runserver_plus 0.0.0.0:${PORT} \
  --cert-file "$CERT" --key-file "$KEY"
