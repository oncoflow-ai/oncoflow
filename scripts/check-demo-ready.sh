#!/usr/bin/env bash
# Quick sanity check before presenting the OncoFlow demo (backend must be running).
set -euo pipefail

PORT="${1:-8000}"
BASE="${ONCOFLOW_READY_URL:-http://127.0.0.1:${PORT}}"
URL="${BASE%/}/api/v1/ready"

echo "Checking ${URL} ..."
code="$(curl -s -o /tmp/oncoflow-ready.json -w '%{http_code}' "${URL}" || true)"
if [[ "${code}" != "200" ]]; then
  echo "FAIL: HTTP ${code} (expected 200). Is uvicorn running with migrations applied?"
  exit 1
fi

echo "OK (HTTP 200). Payload:"
cat /tmp/oncoflow-ready.json
echo ""
echo ""
echo "Reminder: export ONCOFLOW_JOB_EXECUTION_MODE=threaded before uvicorn for local jobs."
echo "Frontend: http://localhost:5173/auth — see DEMO.md for radiologist/doctor/patient flows."
