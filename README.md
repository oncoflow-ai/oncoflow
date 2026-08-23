# oncoflow

## Local development

Install local dependencies once from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e "backend[dev,ml]"
.venv/bin/python -m pip install -r ml/inference/requirements.txt
(cd frontend && npm ci)
```

Afterward, start the backend and frontend together from the repository root:

```bash
./scripts/start-local.sh
```

The standard launcher is self-contained: it uses
`var/oncoflow/dev.sqlite3` and seeds the local demo accounts, so PostgreSQL is
not required for the normal demo. Set `ONCOFLOW_DATABASE_URL` or
`ONCOFLOW_SEED_DEMO_DATA` before launching to override either default.

It also reads the repository-root `.env` for backend settings, including the
threaded Class Demo job configuration, so no separate export is needed for
local MRI uploads. Explicit environment values supplied when launching always
take precedence over `.env`. Frontend API configuration remains in
`frontend/.env.local`, where Vite loads it normally.

The backend readiness endpoint is `http://localhost:8000/api/v1/ready`, and
Vite is available at `http://localhost:5173`. Press Ctrl-C to stop both
development servers. To stop a launcher that is running in another terminal,
use:

```bash
./scripts/start-local.sh --stop
```

This command stops only a launcher it can validate as tracked by this
repository; it never selects processes by port.

For local overrides, set `ONCOFLOW_VENV_PATH`, `ONCOFLOW_BACKEND_PORT`, or
`ONCOFLOW_FRONTEND_PORT` before running the launcher. `ONCOFLOW_RUNTIME_DIR`
is an advanced local/test override for the launcher's private runtime record.

## Local demo login

Run the frontend locally, then sign in with the demo admin account:

```text
Email: admin@oncoflow.local
Password: admin123
```

The admin user opens `/admin/users`, where you can add users and assign patient access.
