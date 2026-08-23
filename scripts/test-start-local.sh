#!/bin/sh

set -eu

REPOSITORY_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TEST_DIR=$(mktemp -d "${TMPDIR:-/tmp}/oncoflow-start-local.XXXXXX")
LAUNCHER_PID=

cleanup() {
  if [ -n "$LAUNCHER_PID" ] && kill -0 "$LAUNCHER_PID" 2>/dev/null; then
    kill -TERM "$LAUNCHER_PID" 2>/dev/null || true
  fi

  if [ -n "$LAUNCHER_PID" ]; then
    wait "$LAUNCHER_PID" 2>/dev/null || true
  fi

  rm -rf "$TEST_DIR"
}

trap cleanup EXIT INT TERM

fail() {
  printf '%s\n' "FAIL: $*" >&2
  exit 1
}

wait_for_file() {
  file=$1
  attempts=0

  while [ ! -s "$file" ] && [ "$attempts" -lt 50 ]; do
    sleep 0.1
    attempts=$((attempts + 1))
  done

  [ -s "$file" ] || fail "Timed out waiting for $file"
}

wait_for_exit() {
  pid=$1
  attempts=0

  while kill -0 "$pid" 2>/dev/null && [ "$attempts" -lt 50 ]; do
    sleep 0.1
    attempts=$((attempts + 1))
  done

  if kill -0 "$pid" 2>/dev/null; then
    fail "Timed out waiting for process $pid to exit"
  fi
}

mkdir -p "$TEST_DIR/venv/bin" "$TEST_DIR/node_modules" "$TEST_DIR/bin"

cat >"$TEST_DIR/venv/bin/uvicorn" <<'EOF'
#!/bin/sh
printf 'uvicorn cwd=%s args=%s pythonpath=%s\n' "$PWD" "$*" "${PYTHONPATH:-}" >>"$ONCOFLOW_TEST_LOG"
printf '%s\n' "$$" >"$ONCOFLOW_TEST_UVICORN_PID"
trap 'exit 0' INT TERM
printf '%s\n' 'uvicorn ready' >&2
while :; do sleep 1; done
EOF
chmod +x "$TEST_DIR/venv/bin/uvicorn"

cat >"$TEST_DIR/bin/npm" <<'EOF'
#!/bin/sh
printf 'npm cwd=%s args=%s\n' "$PWD" "$*" >>"$ONCOFLOW_TEST_LOG"
printf '%s\n' "$$" >"$ONCOFLOW_TEST_NPM_PID"
trap 'exit 0' INT TERM
printf '%s\n' 'vite ready' >&2
while :; do sleep 1; done
EOF
chmod +x "$TEST_DIR/bin/npm"

export ONCOFLOW_TEST_LOG="$TEST_DIR/commands.log"
export ONCOFLOW_TEST_UVICORN_PID="$TEST_DIR/uvicorn.pid"
export ONCOFLOW_TEST_NPM_PID="$TEST_DIR/npm.pid"

(
  cd "$TEST_DIR"
  PATH="$TEST_DIR/bin:$PATH" \
    ONCOFLOW_VENV_PATH="$TEST_DIR/venv" \
    ONCOFLOW_FRONTEND_NODE_MODULES="$TEST_DIR/node_modules" \
    ONCOFLOW_BACKEND_PORT=18000 \
    ONCOFLOW_FRONTEND_PORT=15173 \
    "$REPOSITORY_ROOT/scripts/start-local.sh" >"$TEST_DIR/launcher.log" 2>&1
) &
LAUNCHER_PID=$!

wait_for_file "$ONCOFLOW_TEST_UVICORN_PID"
wait_for_file "$ONCOFLOW_TEST_NPM_PID"

grep -F 'app.main:app --reload --host 127.0.0.1 --port 18000' "$ONCOFLOW_TEST_LOG" >/dev/null || \
  fail 'Uvicorn was not started with the expected import path and loopback binding'
grep -F 'npm cwd=' "$ONCOFLOW_TEST_LOG" >/dev/null || fail 'npm was not invoked'
grep -F 'args=run dev -- --host 127.0.0.1 --port 15173' "$ONCOFLOW_TEST_LOG" >/dev/null || \
  fail 'Vite was not started through npm run dev with the expected loopback binding'

UVICORN_PID=$(cat "$ONCOFLOW_TEST_UVICORN_PID")
NPM_PID=$(cat "$ONCOFLOW_TEST_NPM_PID")

kill -TERM "$LAUNCHER_PID"
wait_for_exit "$LAUNCHER_PID"
wait "$LAUNCHER_PID" || fail 'Launcher returned a nonzero status after TERM'
LAUNCHER_PID=

if kill -0 "$UVICORN_PID" 2>/dev/null; then
  fail 'Uvicorn stub remained alive after launcher shutdown'
fi

if kill -0 "$NPM_PID" 2>/dev/null; then
  fail 'Vite stub remained alive after launcher shutdown'
fi

printf '%s\n' 'PASS: local launcher starts both services and reaps them on shutdown'
