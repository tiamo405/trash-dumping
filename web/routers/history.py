import os, sys
root = os.getcwd()
sys.path.append(root)
from ..utils.Respones import *
from storage.mongo import MongoDBManager
from storage.s3_minio import S3Minio
from fastapi import APIRouter, Depends, HTTPException, Form

view_router = APIRouter(prefix="/prod/api/v1/view", tags=["history"])
mongo_manager = MongoDBManager()
s3 = S3Minio()

@view_router.get("/get_history")
async def get_history(rtsp_url: str , date: int  , page: int = 1, limit: int = 10):
    camera = mongo_manager.get_camera_by_rtsp(rtsp_url)
    if not camera:
        return Response(402, error_resp=402)
    else:
        camera_id = str(camera["_id"])
        violation_images = mongo_manager.get_all_violation_images(camera_id, date)
        # phan trang
        start_index = (page - 1) * limit
        end_index = start_index + limit
        paginated_data = violation_images[start_index:end_index]
        for violation in paginated_data:
            violation["_id"] = str(violation["_id"])
            violation["url_image"] = s3.geturl(violation["_id"] + ".jpg")
            violation["location"] = camera["location"]
        return Response(200, entities={"rtsp_url": rtsp_url, "violations": paginated_data})

