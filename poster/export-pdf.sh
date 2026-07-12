#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CHROME="${CHROME_BIN:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
PDF_DIR="$PROJECT_DIR/output/pdf"
PDF_PATH="$PDF_DIR/oncoflow-project-poster.pdf"
HTML_URL="file://$SCRIPT_DIR/index.html"

if [[ ! -x "$CHROME" ]]; then
  echo "Google Chrome was not found at: $CHROME" >&2
  echo "Set CHROME_BIN to a Chromium-compatible browser executable." >&2
  exit 1
fi

mkdir -p "$PDF_DIR"

"$CHROME" \
  --headless \
  --disable-gpu \
  --no-pdf-header-footer \
  --print-to-pdf="$PDF_PATH" \
  --print-to-pdf-no-header \
  "$HTML_URL"

echo "Exported: $PDF_PATH"
