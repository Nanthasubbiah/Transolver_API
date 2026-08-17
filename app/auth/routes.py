"""
Auth endpoints: register, login (get/regenerate API key).
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from .models import User, APIKey

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class APIKeyResponse(BaseModel):
    api_key: str
    message: str


@router.post("/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(400, "Username already exists")
    user = User(
        username=req.username,
        password_hash=User.hash_password(req.password),
    )
    db.add(user)
    db.commit()

    # Auto-generate first API key on registration
    raw_key = APIKey.generate_key()
    api_key = APIKey(
        key_hash=APIKey.hash_key(raw_key),
        key_prefix=raw_key[:12],
        user_id=user.id,
    )
    db.add(api_key)
    db.commit()

    return {
        "message": f"User '{req.username}' created.",
        "api_key": raw_key,
        "note": "Save this key — it won't be shown again. If lost, POST /auth/reset-key to get a new one."
    }


@router.post("/reset-key", response_model=APIKeyResponse)
def reset_api_key(req: LoginRequest, db: Session = Depends(get_db)):
    """
    Forgot your API key? Login here to deactivate the old one and get a new one.
    """
    user = db.query(User).filter(User.username == req.username).first()
    if not user or not user.verify_password(req.password):
        raise HTTPException(401, "Invalid credentials")

    # Deactivate all old keys
    db.query(APIKey).filter(APIKey.user_id == user.id).update({"is_active": False})

    # Generate new key
    raw_key = APIKey.generate_key()
    api_key = APIKey(
        key_hash=APIKey.hash_key(raw_key),
        key_prefix=raw_key[:12],
        user_id=user.id,
    )
    db.add(api_key)
    db.commit()

    return APIKeyResponse(
        api_key=raw_key,
        message="Old key deactivated. Save this new key — it won't be shown again.",
    )
