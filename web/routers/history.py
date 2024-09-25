import os, sys
root = os.getcwd()
sys.path.append(root)
from ..utils.Respones import *
from storage.mongo import MongoDBManager
from fastapi import APIRouter, Depends, HTTPException, Form

view_router = APIRouter(prefix="/prod/api/v1/view", tags=["history"])
mongo_manager = MongoDBManager()

@view_router.post("/get_history")
async def get_history(rtsp_url: str = Form(...), dateStart: str = Form(...), dateEnd: str = Form(...), page: int = Form(...), limit: int = Form(...)):
    camera = mongo_manager.get_camera_by_rtsp(rtsp_url)
    if not camera:
        return Response(402, error_resp=402)
    else:
        camera_id = camera["_id"]
        
        return Response(200, entities={"rtsp_url": rtsp_url})

