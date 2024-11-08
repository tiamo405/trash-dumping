import uvicorn

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from storage.mongo import MongoDBManager
from storage.s3_minio import S3Minio

from web.routers.camera import camera_router
from web.routers.history import view_router
from web.routers.login import login_router
SECRET_KEY = "your_secret_key"  # Thay thế bằng một khóa bí mật mạnh mẽ
ALGORITHM = "HS256"  # Thuật toán mã hóa
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Thời gian hết hạn token (ví dụ: 60 phút)
mongo_manager = MongoDBManager()
s3 = S3Minio()
app = FastAPI()


allow_origins = ["*"]
app.add_middleware(CORSMiddleware, 
                   allow_origins=allow_origins, 
                   allow_credentials=True, 
                   allow_methods=["*"], 
                   allow_headers=["*"])

# # hello world
@app.get("/")
async def root():
    return {"message": "Hello World"}

    
app.include_router(login_router)
app.include_router(camera_router)
app.include_router(view_router)
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5005, reload=True)

# python app.py or unicorn app:app --reload --host 0.0.0.0 --port 5005