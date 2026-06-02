"""Authentication and security middleware for UAIS-V API."""

import hmac
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

# Configuration
SECRET_KEY = os.getenv("UAIS_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
API_KEYS = frozenset(key.strip() for key in os.getenv("UAIS_API_KEYS", "").split(",") if key.strip())

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


class Token(BaseModel):
    """OAuth2 access token response."""

    access_token: str
    token_type: str


class TokenData(BaseModel):
    """JWT token payload data."""

    username: Optional[str] = None
    scopes: list[str] = Field(default_factory=list)


class User(BaseModel):
    """User model."""

    username: str
    email: Optional[str] = None
    disabled: bool = False
    scopes: list[str] = Field(default_factory=list)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    secret_key = _require_secret_key()
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """Verify and decode a JWT token."""
    secret_key = _require_secret_key()
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, secret_key, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_scopes = payload.get("scopes", [])
        return TokenData(username=username, scopes=token_scopes)
    except JWTError as err:
        raise credentials_exception from err


def _require_secret_key() -> str:
    if SECRET_KEY:
        return SECRET_KEY
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="JWT authentication is not configured",
    )


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> bool:
    """Verify API key from header."""
    if not API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key authentication is not configured",
        )

    # Constant-time comparison against every configured key (audit H2): avoids a
    # timing oracle from short-circuit/hash-based membership on the secret.
    if api_key is None or not any(hmac.compare_digest(api_key, k) for k in API_KEYS):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or missing API key")
    return True


async def verify_bearer_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> TokenData:
    """Verify JWT bearer token."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return verify_token(credentials.credentials)


async def get_current_user(token_data: TokenData = Depends(verify_bearer_token)) -> User:
    """Get current user from token."""
    # In production, fetch user from database
    # For now, create user from token data
    return User(username=token_data.username, scopes=token_data.scopes)


async def require_scope(required_scope: str):
    """Dependency to require specific scope."""

    async def scope_checker(user: User = Depends(get_current_user)) -> User:
        if required_scope not in user.scopes and "admin" not in user.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required scope: {required_scope}",
            )
        return user

    return scope_checker


async def authenticate(api_key: Optional[str] = Security(api_key_header)) -> bool:
    """Require a configured API key for protected routes."""
    return await verify_api_key(api_key)
