"""
API key validation dependency.
"""
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from ..db import get_db
from .models import APIKey, User

api_key_header = APIKeyHeader(name="X-API-Key")


def get_current_user(
    api_key: str = Security(api_key_header),
    db: Session = Depends(get_db),
) -> User:
    key_hash = APIKey.hash_key(api_key)
    db_key = (
        db.query(APIKey)
        .filter(APIKey.key_hash == key_hash, APIKey.is_active == True)
        .first()
    )
    if not db_key:
        raise HTTPException(403, "Invalid or inactive API key")
    return db_key.user
