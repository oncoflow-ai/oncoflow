from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_session
from app.api.schemas.audit import AuditLogResponse
from app.core.audit import log_audit_event
from app.infra.db.models import AuditLog, User

router = APIRouter(prefix="/audit-logs", tags=["audit"])


@router.get("", response_model=list[AuditLogResponse])
def list_audit_logs(
    action: str | None = Query(None, description="Filter by action name"),
    actor_id: str | None = Query(None, description="Filter by actor ID"),
    resource_id: str | None = Query(None, description="Filter by resource ID"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of logs to return"),
    offset: int = Query(0, ge=0, description="Number of logs to skip"),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> list[AuditLogResponse]:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can view audit logs",
        )

    q = session.query(AuditLog)

    if action:
        q = q.filter(AuditLog.action == action)
    if actor_id:
        q = q.filter(AuditLog.actor_id == actor_id)
    if resource_id:
        q = q.filter(AuditLog.resource_id == resource_id)

    logs = q.order_by(desc(AuditLog.timestamp)).offset(offset).limit(limit).all()

    log_audit_event(
        action="QUERY_AUDIT_LOGS",
        resource_id="audit_logs",
        actor=str(current_user.public_id),
        details={"result_count": len(logs), "offset": offset, "limit": limit},
    )

    return [
        AuditLogResponse(
            id=log.id,
            actor_id=log.actor_id,
            action=log.action,
            resource_id=log.resource_id,
            details=log.details,
            timestamp=log.timestamp,
        )
        for log in logs
    ]
