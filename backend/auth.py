import os
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from passlib.context import CryptContext

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET = os.getenv("JWT_SECRET", "dev-only-change-me")
ALG = "HS256"

def hash_password(value: str) -> str:
    return pwd.hash(value)

def verify_password(value: str, hashed: str) -> bool:
    return pwd.verify(value, hashed)

def create_token(user_id: str) -> str:
    payload = {"sub": user_id, "exp": datetime.now(timezone.utc) + timedelta(days=7)}
    return jwt.encode(payload, SECRET, algorithm=ALG)

def decode_token(token: str):
    try:
        return jwt.decode(token, SECRET, algorithms=[ALG]).get("sub")
    except JWTError:
        return None
