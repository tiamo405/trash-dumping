import uvicorn
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from web.routers.camera import camera_router
from web.routers.history import view_router

app = FastAPI()


app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(camera_router)
app.include_router(view_router)

# # hello world
@app.get("/")
async def root():
    return {"message": "Hello World"}



        
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5005, reload=True)

# python app.py or unicorn app:app --reload --host 0.0.0.0 --port 5005