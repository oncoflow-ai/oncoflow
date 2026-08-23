#!/bin/sh

set -eu

REPOSITORY_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SCRIPT_PATH="$REPOSITORY_ROOT/scripts/start-local.sh"

# Keep the running launcher's command line anchored to this repository so a
# later --stop invocation can validate it before sending a signal.
if [ "$0" != "$SCRIPT_PATH" ]; then
  exec "$SCRIPT_PATH" "$@"
fi

VENV_PATH=${ONCOFLOW_VENV_PATH:-"$REPOSITORY_ROOT/.venv"}
FRONTEND_NODE_MODULES=${ONCOFLOW_FRONTEND_NODE_MODULES:-"$REPOSITORY_ROOT/frontend/node_modules"}
BACKEND_PORT=${ONCOFLOW_BACKEND_PORT:-8000}
FRONTEND_PORT=${ONCOFLOW_FRONTEND_PORT:-5173}
RUNTIME_DIR=${ONCOFLOW_RUNTIME_DIR:-"${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}/oncoflow-$(id -u)}"}
RUNTIME_KEY=$(printf '%s' "$REPOSITORY_ROOT" | cksum | awk '{print $1 "-" $2}')
RUNTIME_STATE_FILE="$RUNTIME_DIR/oncoflow-launcher-$RUNTIME_KEY.state"
LOCAL_STATE_DIR="$REPOSITORY_ROOT/var/oncoflow"
LOCAL_DATABASE_URL=${ONCOFLOW_DATABASE_URL:-"sqlite+pysqlite:///$LOCAL_STATE_DIR/dev.sqlite3"}
LOCAL_STORAGE_ROOT=${ONCOFLOW_STORAGE_ROOT:-"$REPOSITORY_ROOT/var/oncoflow"}
LOCAL_SEED_DEMO_DATA=${ONCOFLOW_SEED_DEMO_DATA:-true}
ENV_FILE=${ONCOFLOW_ENV_FILE:-"$REPOSITORY_ROOT/.env"}
UVICORN="$VENV_PATH/bin/uvicorn"
BACKEND_PYTHON="$VENV_PATH/bin/python"
BACKEND_PID=
FRONTEND_PID=
EVENT_DIR=
LAUNCHER_START_TIME=
RUNTIME_RECORD_ROOT=
RUNTIME_RECORD_PID=
RUNTIME_RECORD_START=

fail() {
  printf '%s\n' "$*" >&2
  exit 1
}

load_project_env() {
  if [ ! -f "$ENV_FILE" ]; then
    if [ -n "${ONCOFLOW_ENV_FILE+x}" ]; then
      fail "Environment file does not exist: $ENV_FILE"
    fi
    return 0
  fi

  [ -r "$ENV_FILE" ] || fail "Environment file is not readable: $ENV_FILE"

  dotenv_line_number=0
  while IFS= read -r dotenv_line || [ -n "$dotenv_line" ]; do
    dotenv_line_number=$((dotenv_line_number + 1))
    dotenv_record=$(printf '%s\n' "$dotenv_line" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')

    case "$dotenv_record" in
      '' | \#*) continue ;;
      *=*) ;;
      *) fail "Invalid dotenv record at $ENV_FILE:$dotenv_line_number" ;;
    esac

    dotenv_name=$(printf '%s\n' "${dotenv_record%%=*}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
    dotenv_value=$(printf '%s\n' "${dotenv_record#*=}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')

    case "$dotenv_name" in
      '' | [!A-Za-z_]* | *[!A-Za-z0-9_]*)
        fail "Invalid dotenv record at $ENV_FILE:$dotenv_line_number"
        ;;
    esac

    case "$dotenv_value" in
      \"*\")
        dotenv_value=${dotenv_value#\"}
        dotenv_value=${dotenv_value%\"}
        ;;
      \'*\')
        dotenv_value=${dotenv_value#\'}
        dotenv_value=${dotenv_value%\'}
        ;;
      \"* | \'* | *\" | *\')
        fail "Invalid dotenv record at $ENV_FILE:$dotenv_line_number"
        ;;
    esac

    if printenv "$dotenv_name" >/dev/null 2>&1; then
      continue
    fi

    export "$dotenv_name=$dotenv_value"
  done <"$ENV_FILE"
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

process_start_time() {
  ps -o lstart= -p "$1" 2>/dev/null | awk '{$1=$1; print}'
}

process_command() {
  ps -o command= -p "$1" 2>/dev/null | sed 's/^[[:space:]]*//; s/[[:space:]]*$//'
}

load_runtime_record() {
  RUNTIME_RECORD_ROOT=
  RUNTIME_RECORD_PID=
  RUNTIME_RECORD_START=

  [ -f "$RUNTIME_STATE_FILE" ] || return 1

  record_line_count=$(wc -l <"$RUNTIME_STATE_FILE" | tr -d '[:space:]')
  [ "$record_line_count" = 3 ] || return 1

  record_root_line=$(sed -n '1p' "$RUNTIME_STATE_FILE")
  record_pid_line=$(sed -n '2p' "$RUNTIME_STATE_FILE")
  record_start_line=$(sed -n '3p' "$RUNTIME_STATE_FILE")

  case "$record_root_line" in
    repository_root=*) RUNTIME_RECORD_ROOT=${record_root_line#repository_root=} ;;
    *) return 1 ;;
  esac

  case "$record_pid_line" in
    launcher_pid=*) RUNTIME_RECORD_PID=${record_pid_line#launcher_pid=} ;;
    *) return 1 ;;
  esac

  case "$record_start_line" in
    launcher_start=*) RUNTIME_RECORD_START=${record_start_line#launcher_start=} ;;
    *) return 1 ;;
  esac

  case "$RUNTIME_RECORD_PID" in
    '' | *[!0-9]*) return 1 ;;
  esac

  [ -n "$RUNTIME_RECORD_ROOT" ] && [ -n "$RUNTIME_RECORD_START" ]
}

runtime_record_matches_launcher() {
  load_runtime_record || return 1
  [ "$RUNTIME_RECORD_ROOT" = "$REPOSITORY_ROOT" ] || return 1
  kill -0 "$RUNTIME_RECORD_PID" 2>/dev/null || return 1

  current_start_time=$(process_start_time "$RUNTIME_RECORD_PID")
  [ -n "$current_start_time" ] && [ "$current_start_time" = "$RUNTIME_RECORD_START" ] || return 1

  current_command=$(process_command "$RUNTIME_RECORD_PID")
  case "$current_command" in
    *"$SCRIPT_PATH"*) return 0 ;;
    *) return 1 ;;
  esac
}

runtime_record_belongs_to_current_launcher() {
  load_runtime_record || return 1
  [ "$RUNTIME_RECORD_ROOT" = "$REPOSITORY_ROOT" ] && \
    [ "$RUNTIME_RECORD_PID" = "$$" ] && \
    [ "$RUNTIME_RECORD_START" = "$LAUNCHER_START_TIME" ]
}

remove_own_runtime_record() {
  if runtime_record_belongs_to_current_launcher; then
    rm -f "$RUNTIME_STATE_FILE"
  fi
}

write_runtime_record() {
  (umask 077 && mkdir -p "$RUNTIME_DIR") || fail "Unable to create launcher runtime directory: $RUNTIME_DIR"
  chmod 700 "$RUNTIME_DIR" || fail "Unable to secure launcher runtime directory: $RUNTIME_DIR"

  runtime_temp_file="$RUNTIME_STATE_FILE.$$.tmp"
  (
    umask 077
    {
      printf 'repository_root=%s\n' "$REPOSITORY_ROOT"
      printf 'launcher_pid=%s\n' "$$"
      printf 'launcher_start=%s\n' "$LAUNCHER_START_TIME"
    } >"$runtime_temp_file"
  ) || fail "Unable to write launcher runtime record: $RUNTIME_STATE_FILE"

  mv -f "$runtime_temp_file" "$RUNTIME_STATE_FILE" || fail "Unable to publish launcher runtime record: $RUNTIME_STATE_FILE"
}

stop_tracked_launcher() {
  if ! runtime_record_matches_launcher; then
    rm -f "$RUNTIME_STATE_FILE" 2>/dev/null || true
    printf '%s\n' 'No tracked OncoFlow launcher is running.'
    return 0
  fi

  target_pid=$RUNTIME_RECORD_PID
  printf 'Stopping tracked OncoFlow launcher (PID %s).\n' "$target_pid"
  if ! kill -TERM "$target_pid" 2>/dev/null; then
    printf '%s\n' 'Tracked OncoFlow launcher could not be signaled.' >&2
    return 1
  fi

  attempts=0
  while kill -0 "$target_pid" 2>/dev/null && [ "$attempts" -lt 50 ]; do
    sleep 0.1
    attempts=$((attempts + 1))
  done

  if kill -0 "$target_pid" 2>/dev/null; then
    printf 'Tracked OncoFlow launcher (PID %s) did not exit after TERM.\n' "$target_pid" >&2
    return 1
  fi

  rm -f "$RUNTIME_STATE_FILE" 2>/dev/null || true
  printf '%s\n' 'Tracked OncoFlow launcher stopped.'
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

  remove_own_runtime_record

  if [ -n "$EVENT_DIR" ]; then
    exec 3>&-
    rm -rf "$EVENT_DIR"
  fi

  exit "$status"
}

on_signal() {
  cleanup 0
}

case "${1-}" in
  --stop)
    [ "$#" -eq 1 ] || fail 'Usage: ./scripts/start-local.sh [--stop]'
    stop_tracked_launcher
    exit 0
    ;;
  '') ;;
  *) fail 'Usage: ./scripts/start-local.sh [--stop]' ;;
esac

validate_port "$BACKEND_PORT" 'ONCOFLOW_BACKEND_PORT'
validate_port "$FRONTEND_PORT" 'ONCOFLOW_FRONTEND_PORT'
load_project_env

if [ ! -x "$UVICORN" ]; then
  fail "Backend dependencies are not installed. Run: python3 -m venv .venv && .venv/bin/python -m pip install -e \"backend[dev,ml]\" && .venv/bin/python -m pip install -r ml/inference/requirements.txt"
fi

if [ ! -x "$BACKEND_PYTHON" ] || ! "$BACKEND_PYTHON" -c 'import nibabel, numpy' >/dev/null 2>&1; then
  fail "NIfTI demo dependencies are not installed. Run: \"$BACKEND_PYTHON\" -m pip install -e \"$REPOSITORY_ROOT/backend[dev,ml]\""
fi

if [ ! -d "$FRONTEND_NODE_MODULES" ]; then
  fail 'Frontend dependencies are not installed. Run: (cd frontend && npm ci)'
fi

if ! command -v npm >/dev/null 2>&1; then
  fail 'npm is not available. Install Node.js, then run: (cd frontend && npm ci)'
fi

LAUNCHER_START_TIME=$(process_start_time "$$")
[ -n "$LAUNCHER_START_TIME" ] || fail 'Unable to determine launcher process start time.'

umask 077
mkdir -p "$LOCAL_STATE_DIR" || fail "Unable to create local state directory: $LOCAL_STATE_DIR"
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
    ONCOFLOW_DATABASE_URL="$LOCAL_DATABASE_URL" \
      ONCOFLOW_STORAGE_ROOT="$LOCAL_STORAGE_ROOT" \
      ONCOFLOW_SEED_DEMO_DATA="$LOCAL_SEED_DEMO_DATA" \
      PYTHONPATH=".:..${PYTHONPATH:+:$PYTHONPATH}" \
      exec "$UVICORN" app.main:app --reload --host 127.0.0.1 --port "$BACKEND_PORT"
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
write_runtime_record

printf 'Backend:  http://localhost:%s/api/v1/ready\n' "$BACKEND_PORT"
printf 'Frontend: http://localhost:%s\n' "$FRONTEND_PORT"
printf '%s\n' 'Press Ctrl-C to stop both services.'

if IFS=' ' read -r stopped_service stopped_status <&3; then
  printf '%s\n' "$stopped_service exited unexpectedly with status $stopped_status; stopping the remaining service." >&2
else
  printf '%s\n' 'A local service exited without reporting its status; stopping the remaining service.' >&2
fi

exit 1
