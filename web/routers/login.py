import os, sys

from bson import ObjectId
root = os.getcwd()
sys.path.append(root)
from ..utils.Respones import *
from storage.mongo import MongoDBManager
from storage.s3_minio import S3Minio
from fastapi import APIRouter, Depends, HTTPException, Form

login_router = APIRouter(tags=["login"])
mongo_manager = MongoDBManager()
s3 = S3Minio()

@login_router.post("/login")
async def login(username: str, password: str):
    user = mongo_manager.collection_account.find_one({"username": username, "password": password})
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
        return Response(500, entities={"use": user})
    else:
        return Response(502, error_resp=502)
    
