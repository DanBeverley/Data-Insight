"""FastAPI dependencies for authentication"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from .utils import verify_token
from ..database.models import User
from ..database.connection import get_database_manager

security = HTTPBearer()


def get_db():
    """Get database session"""
    try:
        db_manager = get_database_manager()
        if not db_manager or not db_manager.SessionLocal:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database not available")
        db = db_manager.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Database connection error: {str(e)}"
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token

    Args:
        credentials: HTTP Authorization header with Bearer token
        db: Database session

    Returns:
        User object if authentication succeeds

    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token = credentials.credentials
    user_id = verify_token(token)

    if user_id is None:
        raise credentials_exception

    try:
        user = db.query(User).filter(User.user_id == user_id).first()
    except Exception:
        user = None

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")

    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user (additional check)"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Get current user if token provided, otherwise None.
    Useful for endpoints that work with or without authentication.
    """
    if not credentials:
        return None

    token = credentials.credentials
    user_id = verify_token(token)

    if user_id is None:
        return None

    try:
        user = db.query(User).filter(User.user_id == user_id).first()
        if user and user.is_active:
            return user
    except Exception:
        pass

    return None
