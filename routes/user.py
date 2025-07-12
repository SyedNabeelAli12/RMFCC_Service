from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
import motor.motor_asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from model.user import UserIn, UserOut, UserInDB, Token

router = APIRouter()

# Load env variables
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 7))
MONGO_URI = os.getenv("MONGO_URI")

print("MONGODB",MONGO_URI)

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client["piracy_app"]
users_collection = db.get_collection("users")

# Updated: Use sha256_crypt instead of bcrypt
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Password utils
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# Database helpers
async def get_user(email: str) -> Optional[UserInDB]:
    user = await users_collection.find_one({"email": email})
    if user:
        return UserInDB(
            email=user["email"],
            created_at=user["created_at"],
            hashed_password=user["hashed_password"],
        )

async def authenticate_user(email: str, password: str):
    user = await get_user(email)
    print(email)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Routes
@router.post("/signup", response_model=UserOut)
async def signup(user_in: UserIn):
    if await users_collection.find_one({"email": user_in.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user_in.password)
    user_doc = {
        "email": user_in.email,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow(),
    }
    print(user_doc)
    await users_collection.insert_one(user_doc)
    return UserOut(email=user_in.email, created_at=user_doc["created_at"])

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    token = create_access_token(data={"sub": user.email}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer"}

@router.get("/users/me", response_model=UserOut)
async def read_users_me(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials"
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = await get_user(email=email)
    if user is None:
        raise credentials_exception
    return UserOut(email=user.email, created_at=user.created_at)
