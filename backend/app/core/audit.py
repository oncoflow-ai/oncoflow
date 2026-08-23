import logging
from typing import Any
from contextvars import ContextVar

# We use a dedicated logger for auditing
audit_logger = logging.getLogger("oncoflow.audit")

# Context variable to hold the current user's ID
current_actor: ContextVar[str] = ContextVar("current_actor", default="system")


def log_audit_event(
    action: str,
    resource_id: str,
    actor: str | None = None,
    details: dict[str, Any] | None = None,
    db: Any | None = None,
) -> None:
    """
    Emits a structured JSON audit log and persists to the audit_logs database table.
    If 'actor' is not explicitly provided, it is retrieved from the context variable.
    """
    if actor:
        final_actor = actor
    else:
        final_actor = current_actor.get()

    event = {
        "action": action,
        "resource_id": resource_id,
        "actor": final_actor,
    }
    if details:
        event["details"] = details

    audit_logger.info("Audit Event", extra=event)

    if db is not None:
        from app.infra.db.models import AuditLog

        audit_entry = AuditLog(
            actor_id=str(final_actor),
            action=action,
            resource_id=resource_id,
            details=details or {},
        )
        db.add(audit_entry)
        db.flush()
        return

    try:
        from app.infra.db.models import AuditLog
        from app.infra.db.session import create_session_factory

        audit_entry = AuditLog(
            actor_id=str(final_actor),
            action=action,
            resource_id=resource_id,
            details=details or {},
        )
        session_factory = create_session_factory()
        with session_factory() as session:
            session.add(audit_entry)
            session.commit()
    except Exception as exc:
        audit_logger.debug("Failed to persist audit log to database: %s", exc)
