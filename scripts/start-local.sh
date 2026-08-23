#!/bin/sh

set -eu

REPOSITORY_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
VENV_PATH=${ONCOFLOW_VENV_PATH:-"$REPOSITORY_ROOT/.venv"}
FRONTEND_NODE_MODULES=${ONCOFLOW_FRONTEND_NODE_MODULES:-"$REPOSITORY_ROOT/frontend/node_modules"}
BACKEND_PORT=${ONCOFLOW_BACKEND_PORT:-8000}
FRONTEND_PORT=${ONCOFLOW_FRONTEND_PORT:-5173}
UVICORN="$VENV_PATH/bin/uvicorn"
BACKEND_PID=
FRONTEND_PID=
EVENT_DIR=

fail() {
  printf '%s\n' "$*" >&2
  exit 1
}

validate_port() {
  port=$1
  label=$2

  case "$port" in
    '' | *[!0-9]*) fail "$label must be an integer between 1 and 65535." ;;
  esac

  if [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
    fail "$label must be an integer between 1 and 65535."
  fi
}

terminate_child() {
  pid=$1

  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
}

wait_for_child() {
  pid=$1

  if [ -n "$pid" ]; then
    set +e
    wait "$pid"
    set -e
  fi
}

cleanup() {
  status=$1
  trap - INT TERM EXIT

  terminate_child "$BACKEND_PID"
  terminate_child "$FRONTEND_PID"
  wait_for_child "$BACKEND_PID"
  wait_for_child "$FRONTEND_PID"

  if [ -n "$EVENT_DIR" ]; then
    exec 3>&-
    rm -rf "$EVENT_DIR"
  fi

  exit "$status"
}

on_signal() {
  cleanup 0
}

validate_port "$BACKEND_PORT" 'ONCOFLOW_BACKEND_PORT'
validate_port "$FRONTEND_PORT" 'ONCOFLOW_FRONTEND_PORT'

if [ ! -x "$UVICORN" ]; then
  fail "Backend dependencies are not installed. Run: python3 -m venv .venv && .venv/bin/python -m pip install -e \"backend[dev,ml]\" && .venv/bin/python -m pip install -r ml/inference/requirements.txt"
fi

if [ ! -d "$FRONTEND_NODE_MODULES" ]; then
  fail 'Frontend dependencies are not installed. Run: (cd frontend && npm ci)'
fi

if ! command -v npm >/dev/null 2>&1; then
  fail 'npm is not available. Install Node.js, then run: (cd frontend && npm ci)'
fi

umask 077
EVENT_DIR=$(mktemp -d "${TMPDIR:-/tmp}/oncoflow-start-local.XXXXXX")
EVENT_FIFO="$EVENT_DIR/service-exit"
mkfifo "$EVENT_FIFO"
exec 3<> "$EVENT_FIFO"

run_backend() {
  child_pid=

  stop_backend() {
    terminate_child "$child_pid"
    wait_for_child "$child_pid"
  }

  trap 'stop_backend; exit 0' INT TERM

  (
    cd "$REPOSITORY_ROOT/backend"
    PYTHONPATH=".:..${PYTHONPATH:+:$PYTHONPATH}" exec "$UVICORN" app.main:app --reload --host 127.0.0.1 --port "$BACKEND_PORT"
  ) &
  child_pid=$!

  set +e
  wait "$child_pid"
  child_status=$?
  set -e

  trap - INT TERM
  printf 'backend %s\n' "$child_status" >"$EVENT_FIFO"
  exit "$child_status"
}

run_frontend() {
  child_pid=

  stop_frontend() {
    terminate_child "$child_pid"
    wait_for_child "$child_pid"
  }

  trap 'stop_frontend; exit 0' INT TERM

  (
    cd "$REPOSITORY_ROOT/frontend"
    exec npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT"
  ) &
  child_pid=$!

  set +e
  wait "$child_pid"
  child_status=$?
  set -e

  trap - INT TERM
  printf 'frontend %s\n' "$child_status" >"$EVENT_FIFO"
  exit "$child_status"
}

trap on_signal INT TERM
trap 'cleanup "$?"' EXIT

run_backend &
BACKEND_PID=$!
run_frontend &
FRONTEND_PID=$!

printf 'Backend:  http://localhost:%s/api/v1/ready\n' "$BACKEND_PORT"
printf 'Frontend: http://localhost:%s\n' "$FRONTEND_PORT"
printf '%s\n' 'Press Ctrl-C to stop both services.'

if IFS=' ' read -r stopped_service stopped_status <&3; then
  printf '%s\n' "$stopped_service exited unexpectedly with status $stopped_status; stopping the remaining service." >&2
else
  printf '%s\n' 'A local service exited without reporting its status; stopping the remaining service.' >&2
fi

exit 1
