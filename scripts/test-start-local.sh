#!/bin/sh

set -eu

REPOSITORY_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TEST_DIR=$(mktemp -d "${TMPDIR:-/tmp}/oncoflow-start-local.XXXXXX")
LAUNCHER_PID=
RUNTIME_DIR="$TEST_DIR/runtime"

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
printf 'uvicorn cwd=%s args=%s pythonpath=%s database_url=%s seed_demo_data=%s\n' \
  "$PWD" "$*" "${PYTHONPATH:-}" "${ONCOFLOW_DATABASE_URL:-}" "${ONCOFLOW_SEED_DEMO_DATA:-}" >>"$ONCOFLOW_TEST_LOG"
printf 'uvicorn_env storage_root=%s job_execution_mode=%s demo_job_delay_seconds=%s\n' \
  "${ONCOFLOW_STORAGE_ROOT:-}" "${ONCOFLOW_JOB_EXECUTION_MODE:-}" "${ONCOFLOW_DEMO_JOB_DELAY_SECONDS:-}" >>"$ONCOFLOW_TEST_LOG"
printf '%s\n' "$$" >"$ONCOFLOW_TEST_UVICORN_PID"
trap 'exit 0' INT TERM
printf '%s\n' 'uvicorn ready' >&2
while :; do sleep 1; done
EOF
chmod +x "$TEST_DIR/venv/bin/uvicorn"

cat >"$TEST_DIR/venv/bin/python" <<'EOF'
#!/bin/sh
case "$*" in
  *'import nibabel, numpy'*) exit "${ONCOFLOW_TEST_NIFTI_IMPORT_STATUS:-0}" ;;
  *) exit 0 ;;
esac
EOF
chmod +x "$TEST_DIR/venv/bin/python"

cat >"$TEST_DIR/bin/npm" <<'EOF'
#!/bin/sh
printf 'npm cwd=%s args=%s\n' "$PWD" "$*" >>"$ONCOFLOW_TEST_LOG"
printf '%s\n' "$$" >"$ONCOFLOW_TEST_NPM_PID"
trap 'exit 0' INT TERM
printf '%s\n' 'vite ready' >&2
while :; do sleep 1; done
EOF
chmod +x "$TEST_DIR/bin/npm"

cat >"$TEST_DIR/bin/ps" <<'EOF'
#!/bin/sh
case "$*" in
  *'lstart='*) printf '%s\n' 'Sun Aug 23 16:00:00 2026' ;;
  *'command='*) printf '%s\n' "$ONCOFLOW_TEST_SCRIPT_PATH" ;;
  *) exit 1 ;;
esac
EOF
chmod +x "$TEST_DIR/bin/ps"

export ONCOFLOW_TEST_LOG="$TEST_DIR/commands.log"
export ONCOFLOW_TEST_UVICORN_PID="$TEST_DIR/uvicorn.pid"
export ONCOFLOW_TEST_NPM_PID="$TEST_DIR/npm.pid"
export ONCOFLOW_TEST_SCRIPT_PATH="$REPOSITORY_ROOT/scripts/start-local.sh"

cat >"$TEST_DIR/launcher.env" <<'EOF'
# Isolated non-secret backend settings for launcher propagation coverage.
ONCOFLOW_JOB_EXECUTION_MODE=threaded
ONCOFLOW_DEMO_JOB_DELAY_SECONDS=10
OFLOW_BACKEND=local
EOF

STATE_KEY=$(printf '%s' "$REPOSITORY_ROOT" | cksum | awk '{print $1 "-" $2}')
STATE_RECORD="$RUNTIME_DIR/oncoflow-launcher-$STATE_KEY.state"

cd "$TEST_DIR"
MISSING_NIFTI_LOG="$TEST_DIR/missing-nifti.log"
env -i \
  PATH="$TEST_DIR/bin:/usr/bin:/bin" \
  ONCOFLOW_VENV_PATH="$TEST_DIR/venv" \
  ONCOFLOW_FRONTEND_NODE_MODULES="$TEST_DIR/node_modules" \
  ONCOFLOW_RUNTIME_DIR="$RUNTIME_DIR" \
  ONCOFLOW_ENV_FILE="$TEST_DIR/launcher.env" \
  ONCOFLOW_TEST_NIFTI_IMPORT_STATUS=1 \
  ONCOFLOW_TEST_LOG="$ONCOFLOW_TEST_LOG" \
  ONCOFLOW_TEST_UVICORN_PID="$ONCOFLOW_TEST_UVICORN_PID" \
  ONCOFLOW_TEST_NPM_PID="$ONCOFLOW_TEST_NPM_PID" \
  ONCOFLOW_TEST_SCRIPT_PATH="$ONCOFLOW_TEST_SCRIPT_PATH" \
  "$REPOSITORY_ROOT/scripts/start-local.sh" >"$MISSING_NIFTI_LOG" 2>&1 &
MISSING_NIFTI_PID=$!

sleep 1
if kill -0 "$MISSING_NIFTI_PID" 2>/dev/null; then
  kill -TERM "$MISSING_NIFTI_PID" 2>/dev/null || true
  wait "$MISSING_NIFTI_PID" 2>/dev/null || true
  fail 'Launcher did not reject missing NIfTI dependencies before starting services'
fi
wait "$MISSING_NIFTI_PID" 2>/dev/null || true
grep -F 'NIfTI demo dependencies are not installed.' "$MISSING_NIFTI_LOG" >/dev/null || \
  fail 'Launcher did not explain how to install missing NIfTI dependencies'

env -i \
  PATH="$TEST_DIR/bin:/usr/bin:/bin" \
  ONCOFLOW_VENV_PATH="$TEST_DIR/venv" \
  ONCOFLOW_FRONTEND_NODE_MODULES="$TEST_DIR/node_modules" \
  ONCOFLOW_BACKEND_PORT=18000 \
  ONCOFLOW_FRONTEND_PORT=15173 \
  ONCOFLOW_RUNTIME_DIR="$RUNTIME_DIR" \
  ONCOFLOW_ENV_FILE="$TEST_DIR/launcher.env" \
  ONCOFLOW_TEST_LOG="$ONCOFLOW_TEST_LOG" \
  ONCOFLOW_TEST_UVICORN_PID="$ONCOFLOW_TEST_UVICORN_PID" \
  ONCOFLOW_TEST_NPM_PID="$ONCOFLOW_TEST_NPM_PID" \
  ONCOFLOW_TEST_SCRIPT_PATH="$ONCOFLOW_TEST_SCRIPT_PATH" \
  "$REPOSITORY_ROOT/scripts/start-local.sh" >"$TEST_DIR/launcher.log" 2>&1 &
LAUNCHER_PID=$!

wait_for_file "$ONCOFLOW_TEST_UVICORN_PID"
wait_for_file "$ONCOFLOW_TEST_NPM_PID"

grep -F 'app.main:app --reload --host 127.0.0.1 --port 18000' "$ONCOFLOW_TEST_LOG" >/dev/null || \
  fail 'Uvicorn was not started with the expected import path and loopback binding'
grep -F "database_url=sqlite+pysqlite:///$REPOSITORY_ROOT/var/oncoflow/dev.sqlite3 seed_demo_data=true" "$ONCOFLOW_TEST_LOG" >/dev/null || \
  fail 'Uvicorn was not given the self-contained SQLite demo configuration'
grep -F "uvicorn_env storage_root=$REPOSITORY_ROOT/var/oncoflow job_execution_mode=threaded demo_job_delay_seconds=10" "$ONCOFLOW_TEST_LOG" >/dev/null || \
  fail 'Uvicorn did not receive the dotenv worker configuration'
grep -F 'npm cwd=' "$ONCOFLOW_TEST_LOG" >/dev/null || fail 'npm was not invoked'
grep -F 'args=run dev -- --host 127.0.0.1 --port 15173' "$ONCOFLOW_TEST_LOG" >/dev/null || \
  fail 'Vite was not started through npm run dev with the expected loopback binding'

UVICORN_PID=$(cat "$ONCOFLOW_TEST_UVICORN_PID")
NPM_PID=$(cat "$ONCOFLOW_TEST_NPM_PID")

[ -s "$STATE_RECORD" ] || fail 'Launcher did not create its runtime record'

PATH="$TEST_DIR/bin:$PATH" \
  ONCOFLOW_RUNTIME_DIR="$RUNTIME_DIR" \
  "$REPOSITORY_ROOT/scripts/start-local.sh" --stop >"$TEST_DIR/stop.log" 2>&1

wait_for_exit "$LAUNCHER_PID"
wait "$LAUNCHER_PID" || fail 'Launcher returned a nonzero status after --stop'
LAUNCHER_PID=

if kill -0 "$UVICORN_PID" 2>/dev/null; then
  fail 'Uvicorn stub remained alive after launcher shutdown'
fi

if kill -0 "$NPM_PID" 2>/dev/null; then
  fail 'Vite stub remained alive after launcher shutdown'
fi

[ ! -e "$STATE_RECORD" ] || fail 'Launcher runtime record remained after shutdown'

PATH="$TEST_DIR/bin:$PATH" \
  ONCOFLOW_RUNTIME_DIR="$RUNTIME_DIR" \
  "$REPOSITORY_ROOT/scripts/start-local.sh" --stop >"$TEST_DIR/stop-again.log" 2>&1

grep -F 'No tracked OncoFlow launcher is running.' "$TEST_DIR/stop-again.log" >/dev/null || \
  fail 'Repeated --stop did not report that no tracked launcher is running'

OVERRIDE_LOG="$TEST_DIR/override-commands.log"
OVERRIDE_UVICORN_PID="$TEST_DIR/override-uvicorn.pid"
OVERRIDE_NPM_PID="$TEST_DIR/override-npm.pid"

env -i \
  PATH="$TEST_DIR/bin:/usr/bin:/bin" \
  ONCOFLOW_VENV_PATH="$TEST_DIR/venv" \
  ONCOFLOW_FRONTEND_NODE_MODULES="$TEST_DIR/node_modules" \
  ONCOFLOW_BACKEND_PORT=18000 \
  ONCOFLOW_FRONTEND_PORT=15173 \
  ONCOFLOW_RUNTIME_DIR="$RUNTIME_DIR" \
  ONCOFLOW_ENV_FILE="$TEST_DIR/launcher.env" \
  ONCOFLOW_JOB_EXECUTION_MODE=deferred \
  ONCOFLOW_TEST_LOG="$OVERRIDE_LOG" \
  ONCOFLOW_TEST_UVICORN_PID="$OVERRIDE_UVICORN_PID" \
  ONCOFLOW_TEST_NPM_PID="$OVERRIDE_NPM_PID" \
  ONCOFLOW_TEST_SCRIPT_PATH="$ONCOFLOW_TEST_SCRIPT_PATH" \
  "$REPOSITORY_ROOT/scripts/start-local.sh" >"$TEST_DIR/override-launcher.log" 2>&1 &
LAUNCHER_PID=$!

wait_for_file "$OVERRIDE_UVICORN_PID"
wait_for_file "$OVERRIDE_NPM_PID"

grep -F 'uvicorn_env storage_root=' "$OVERRIDE_LOG" >/dev/null || \
  fail 'Override lifecycle did not start Uvicorn'
grep -F 'job_execution_mode=deferred demo_job_delay_seconds=10' "$OVERRIDE_LOG" >/dev/null || \
  fail 'Explicit shell job mode did not override the dotenv value'

OVERRIDE_UVICORN_PROCESS=$(cat "$OVERRIDE_UVICORN_PID")
OVERRIDE_NPM_PROCESS=$(cat "$OVERRIDE_NPM_PID")

PATH="$TEST_DIR/bin:$PATH" \
  ONCOFLOW_RUNTIME_DIR="$RUNTIME_DIR" \
  "$REPOSITORY_ROOT/scripts/start-local.sh" --stop >"$TEST_DIR/override-stop.log" 2>&1

wait_for_exit "$LAUNCHER_PID"
wait "$LAUNCHER_PID" || fail 'Override launcher returned a nonzero status after --stop'
LAUNCHER_PID=

if kill -0 "$OVERRIDE_UVICORN_PROCESS" 2>/dev/null; then
  fail 'Override Uvicorn stub remained alive after launcher shutdown'
fi

if kill -0 "$OVERRIDE_NPM_PROCESS" 2>/dev/null; then
  fail 'Override Vite stub remained alive after launcher shutdown'
fi

[ ! -e "$STATE_RECORD" ] || fail 'Override launcher runtime record remained after shutdown'

printf '%s\n' 'PASS: local launcher loads safe backend dotenv settings, honors shell overrides, and reaps both service lifecycles through --stop'
