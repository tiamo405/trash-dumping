from datetime import datetime, timedelta
import os, sys

from bson import ObjectId
import jwt
root = os.getcwd()
sys.path.append(root)
from ..utils.Respones import *
from storage.mongo import MongoDBManager
from storage.s3_minio import S3Minio
from fastapi import APIRouter, Depends, HTTPException, Form
from pydantic import BaseModel

login_router = APIRouter(tags=["login"])
mongo_manager = MongoDBManager()
s3 = S3Minio()

SECRET_KEY = "your_secret_key"  # Thay thế bằng một khóa bí mật mạnh mẽ
ALGORITHM = "HS256"  # Thuật toán mã hóa
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Thời gian hết hạn token (ví dụ: 60 phút)
class User(BaseModel):
    username: str
    password: str

# Hàm tạo token JWT
def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Giả sử có một hàm kiểm tra tài khoản người dùng
def authenticate_user(username: str, password: str):
    user = mongo_manager.collection_account.find_one({"username": username, "password": password})
    return user if user else None

# Router đăng nhập
@login_router.post("/login")
async def login(userRequest : User):
    print(userRequest.username)
    user = authenticate_user(userRequest.username, userRequest.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    # get role
    role_id = user['role_id']
    role = mongo_manager.collection_role.find_one({"_id": ObjectId(role_id)})
    user["role"] = role["role_name"]
    # Tạo token cho user
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["username"]}, expires_delta=access_token_expires)
    user["_id"] = str(user["_id"])
    user.pop("password")
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Đơn vị giây,
        "user": user
    }

# get profile
@login_router.get("/profile")
async def get_profile(username: str):
    user = mongo_manager.collection_account.find_one({"username": username})
    if user:
        user["role"] = mongo_manager.collection_role.find_one({"_id": ObjectId(user["role_id"])})["role_name"]
        user["_id"] = str(user["_id"])
        # remove pass
        user.pop("password")
        return Response(500, entities={"use": user})
    else:
        return Response(502, error_resp=502)
    
