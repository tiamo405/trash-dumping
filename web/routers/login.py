import os, sys

from bson import ObjectId
root = os.getcwd()
sys.path.append(root)
from ..utils.Respones import *
from storage.mongo import MongoDBManager
from storage.s3_minio import S3Minio
from fastapi import APIRouter, Depends, HTTPException, Form
from pydantic import BaseModel
from jose import JWTError, jwt 

login_router = APIRouter(tags=["login"])
mongo_manager = MongoDBManager()
s3 = S3Minio()

SECRET_KEY = "your_secret_key"  # Thay thế bằng một khóa bí mật mạnh mẽ
ALGORITHM = "HS256"  # Thuật toán mã hóa
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Thời gian hết hạn token (ví dụ: 60 phút)
class User(BaseModel):
    username: str
    password: str

@login_router.post("/login")
async def login(userReponse : User):
    user = mongo_manager.collection_account.find_one({"username": userReponse.username, "password": userReponse.password})
    if user:
        user["role"] = mongo_manager.collection_role.find_one({"_id": ObjectId(user["role_id"])})["role_name"]
        return Response(500, entities={"username": user["username"], "role": user["role"]})
    else:
        return Response(501, error_resp=401)

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
    
