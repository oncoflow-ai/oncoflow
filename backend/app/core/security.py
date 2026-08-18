from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
import jwt
from typing import Any
from app.core.config import get_settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(subject: str | Any, expires_delta: timedelta | None = None) -> str:
    settings = get_settings()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        # Default to 8 hours
        expire = datetime.now(timezone.utc) + timedelta(minutes=480)
        
    if isinstance(subject, dict):
        to_encode = subject.copy()
        to_encode["exp"] = expire
        if "sub" in to_encode:
            to_encode["sub"] = str(to_encode["sub"])
    else:
        to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt

