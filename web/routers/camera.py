from datetime import datetime
import cv2
import os, sys
root = os.getcwd()
sys.path.append(root)
from ..utils.Respones import *
from storage.mongo import MongoDBManager
from fastapi import APIRouter, Depends, HTTPException, Form


camera_router = APIRouter(prefix="/prod/api/v1/camera", tags=["camera"])

mongo_manager = MongoDBManager()

def check_camera(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        return False
    return True

@camera_router.post("/create")
async def create_cam(rtsp_url: str = Form(...), location: str = Form(...)):
    camera = mongo_manager.get_camera_by_rtsp(rtsp_url)
    if camera: # camera already
        return Response(101, error_resp=101)
    else: # camera not exists
        # check camera
        if not check_camera(rtsp_url):
            return Response(102, error_resp=102)
        else:
            new_camera = {
                "rtsp_cam": rtsp_url,
                "isOpen": True,
                "date_added": datetime.utcnow().timestamp(),
                "location": location
            }
            camera_id = mongo_manager.insert_camera(new_camera)
            return Response(200, entities={"rtsp_url": rtsp_url, "camera_id": camera_id})

@camera_router.post("/remove")
async def remove_cam(rtsp_url: str = Form(...)):
    camera = mongo_manager.get_camera_by_rtsp(rtsp_url)
    if not camera: # camera not exists
        return Response(201, error_resp=201)
    else: # camera exists
        camera_id = camera["_id"]
        mongo_manager.delete_camera_by_id(camera_id)
        return Response(200, msg="Success remove camera", entities={"rtsp_url": rtsp_url})

@camera_router.get("/list")
async def list_cam(page: int = 1, limit: int = 5):
    cameras = mongo_manager.get_all_cameras()
    for cam in cameras:
        cam["_id"] = str(cam["_id"])
    start_index = (page - 1) * limit
    end_index = start_index + limit
    paginated_data = cameras[start_index:end_index]
    total_pages = (len(cameras) + limit - 1) // limit
    return Response(0, entities={"cameras": paginated_data, "total_pages": total_pages})