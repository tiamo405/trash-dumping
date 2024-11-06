import uvicorn
import os

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import timedelta
from storage.mongo import MongoDBManager
from storage.s3_minio import S3Minio
from web.routers.auth import create_access_token, verify_password, get_password_hash, decode_access_token

from web.routers.camera import camera_router
from web.routers.history import view_router
from web.routers.login import login_router

mongo_manager = MongoDBManager()
s3 = S3Minio()
app = FastAPI()

app.include_router(camera_router)
app.include_router(view_router)
# app.include_router(login_router)

allow_origins = ["*"]
app.add_middleware(CORSMiddleware, 
                   allow_origins=allow_origins, 
                   allow_credentials=True, 
                   allow_methods=["*"], 
                   allow_headers=["*"])
# OAuth2 schema
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# # hello world
@app.get("/")
async def root():
    return {"message": "Hello World"}

# Hàm xác thực người dùng
def authenticate_user(username: str, password: str):
    print(username, password)
    user = mongo_manager.collection_account.find_one({"username": username, "password": password})
    if user:
        return user
    return False

# API đăng nhập
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), token: str = Depends(oauth2_scheme)):
    # Giải mã token cũ nếu có
    old_token_payload = decode_access_token(token)
    
    # Nếu token cũ vẫn còn hạn, không cần tạo token mới
    if old_token_payload and old_token_payload["sub"] == form_data.username:
        return {"access_token": token, "token_type": "bearer"}
    
    # Nếu không có token hoặc token đã hết hạn, tạo token mới
    user = authenticate_user( form_data.username, form_data.password)
    if not user:
        print("Incorrect username or password")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(days=2)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


# API yêu cầu xác thực bằng token
@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"username": payload["sub"]}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5005, reload=True)

# python app.py or unicorn app:app --reload --host 0.0.0.0 --port 5005