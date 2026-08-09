import logging
from typing import Any

# We use a dedicated logger for auditing
audit_logger = logging.getLogger("oncoflow.audit")

def log_audit_event(
    action: str,
    resource_id: str,
    actor: str = "system",
    details: dict[str, Any] | None = None
) -> None:
    """
    Emits a structured JSON audit log.
    The actor defaults to 'system' but should be the authenticated user's ID
    once Goal 4 (Access Control & Authentication) is fully implemented.
    """
    event = {
        "action": action,
        "resource_id": resource_id,
        "actor": actor,
    }
    if details:
        event["details"] = details
        
    audit_logger.info("Audit Event", extra=event)
