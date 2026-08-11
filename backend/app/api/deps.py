from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from sqlalchemy.orm import Session
from uuid import UUID
from typing import Generator

from app.core.config import get_settings
from app.infra.db.session import create_session_factory
from app.infra.db.models import User
from app.core.audit import current_actor

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

def get_session() -> Generator[Session, None, None]:
    factory = create_session_factory()
    with factory() as session:
        yield session

def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session)
) -> User:
    settings = get_settings()
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        user_id_str: str | None = payload.get("sub")
        if user_id_str is None:
            raise credentials_exception
    except jwt.InvalidTokenError:
        raise credentials_exception
        
    try:
        user_uuid = UUID(user_id_str)
    except ValueError:
        raise credentials_exception
        
    user = session.query(User).filter(User.public_id == user_uuid).first()
    if user is None:
        raise credentials_exception
        
    # Inject into context var for audit logging
    current_actor.set(str(user.public_id))
    
    return user
