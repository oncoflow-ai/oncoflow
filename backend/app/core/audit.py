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
    details: dict[str, Any] | None = None
) -> None:
    """
    Emits a structured JSON audit log.
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
