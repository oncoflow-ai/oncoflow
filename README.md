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
