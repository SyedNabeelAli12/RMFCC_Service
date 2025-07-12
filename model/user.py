from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta


# Pydantic models
class UserIn(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    email: EmailStr
    created_at: datetime


class UserInDB(UserOut):
    hashed_password: str


class Token(BaseModel):
    access_token: str
    token_type: str

