"""Authentication router for user registration, login, and OAuth"""

import uuid
import os
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, Field
from ..auth.utils import verify_password, get_password_hash, create_access_token
from ..auth.dependencies import get_db, get_current_user, get_optional_user
from ..auth.oauth import get_oauth, is_google_oauth_configured
from ..database.models import User

router = APIRouter(prefix="/api/auth", tags=["authentication"])

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8000")


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    full_name: Optional[str] = None
    allow_email_notifications: bool = False


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    avatar_url: Optional[str]
    allow_email_notifications: bool
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class AuthStatus(BaseModel):
    is_authenticated: bool
    is_guest: bool
    user: Optional[UserResponse] = None


def user_to_dict(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "avatar_url": user.avatar_url,
        "allow_email_notifications": user.allow_email_notifications,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(user_data.password)

    new_user = User(
        id=user_id,
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        allow_email_notifications=user_data.allow_email_notifications,
        is_active=True,
        created_at=datetime.utcnow(),
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    access_token = create_access_token(data={"sub": new_user.id})

    return {"access_token": access_token, "token_type": "bearer", "user": user_to_dict(new_user)}


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == credentials.email).first()

    if not user or not user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive")

    user.last_login = datetime.utcnow()
    db.commit()

    access_token = create_access_token(data={"sub": user.id})

    return {"access_token": access_token, "token_type": "bearer", "user": user_to_dict(user)}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    return current_user


@router.get("/status")
async def get_auth_status(user: Optional[User] = Depends(get_optional_user)):
    if user:
        return {"is_authenticated": True, "is_guest": False, "user": user_to_dict(user)}
    return {"is_authenticated": False, "is_guest": True, "user": None}


@router.post("/logout")
async def logout():
    return {"success": True, "message": "Logged out successfully"}


@router.get("/google")
async def google_login(request: Request):
    if not is_google_oauth_configured():
        raise HTTPException(status_code=503, detail="Google OAuth not configured")

    oauth = get_oauth()
    redirect_uri = f"{FRONTEND_URL}/api/auth/google/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    if not is_google_oauth_configured():
        raise HTTPException(status_code=503, detail="Google OAuth not configured")

    oauth = get_oauth()

    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        return RedirectResponse(url=f"{FRONTEND_URL}/login?error=oauth_failed")

    user_info = token.get("userinfo")
    if not user_info:
        return RedirectResponse(url=f"{FRONTEND_URL}/login?error=no_user_info")

    google_id = user_info.get("sub")
    email = user_info.get("email")
    name = user_info.get("name")
    picture = user_info.get("picture")

    user = db.query(User).filter(User.google_oauth_id == google_id).first()

    if not user:
        user = db.query(User).filter(User.email == email).first()
        if user:
            user.google_oauth_id = google_id
            user.avatar_url = picture
            if name and not user.full_name:
                user.full_name = name
        else:
            user = User(
                id=str(uuid.uuid4()),
                email=email,
                full_name=name,
                avatar_url=picture,
                google_oauth_id=google_id,
                allow_email_notifications=True,
                is_active=True,
                created_at=datetime.utcnow(),
            )
            db.add(user)

    user.last_login = datetime.utcnow()
    db.commit()
    db.refresh(user)

    access_token = create_access_token(data={"sub": user.id})

    return RedirectResponse(url=f"{FRONTEND_URL}/?token={access_token}")


@router.get("/google/available")
async def google_oauth_available():
    return {"available": is_google_oauth_configured()}
