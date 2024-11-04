from datetime import datetime
import cv2
import os, sys
root = os.getcwd()
sys.path.append(root)
from bson import ObjectId
from ..utils.Respones import *
from storage.mongo import MongoDBManager
from storage.s3_minio import S3Minio
from fastapi import APIRouter, Depends, HTTPException, Form


camera_router = APIRouter(prefix="/prod/api/v1/camera", tags=["camera"])

mongo_manager = MongoDBManager()
s3 = S3Minio()

def check_camera(rtsp_url):
    return True
    # cap = cv2.VideoCapture(rtsp_url)
    # if not cap.isOpened():
    #     cap.release()
    #     return False
    # cap.release()
    # return True

@camera_router.post("/create")
async def create_cam(rtsp_url: str, location: str):
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
            # run container name camid
            return Response(200, entities={"rtsp_url": rtsp_url, "camera_id": camera_id})

@camera_router.delete("/remove")
async def remove_cam(cam_id : str):
    camera = mongo_manager.collection_camera.find_one({"_id": ObjectId(cam_id)})
    if not camera: # camera not exists
        return Response(201, error_resp=201)
    else: # camera exists
        camera_id = camera["_id"]
        mongo_manager.delete_camera_by_id(camera_id)
        # remove container name camid
        return Response(200, msg="Success remove camera")

@camera_router.get("/list")
async def list_cam(page: int = 1, limit: int = 5):
    cameras = mongo_manager.get_all_cameras()
    for cam in cameras:
        cam["_id"] = str(cam["_id"])
        cam["origin_image"] = s3.geturl(cam["origin_image"])
        cam["add_by"] = mongo_manager.collection_account.find_one({"_id": ObjectId(cam["add_by_customer_id"])})["username"]
    start_index = (page - 1) * limit
    end_index = start_index + limit
    paginated_data = cameras[start_index:end_index]
    total_pages = (len(cameras) + limit - 1) // limit
    return Response(0, entities={"cameras": paginated_data, "total_pages": total_pages})

@camera_router.post("/start")
async def start_cam(rtsp_url: str = Form(...)):
    camera = mongo_manager.get_camera_by_rtsp(rtsp_url)
    if not camera:
        return Response(102, error_resp=102)
    else:
        if camera['is_activate']:
            return Response(200, msg="Camera is already activated")
        else:
            camera['is_activate'] = True
            ok = mongo_manager.update_camera(camera)
            if ok:
                return Response(200, msg="Camera is activated")
            else:
                return Response(500, error_resp=500)
            
# get all lacation camera
@camera_router.get("/get_all_location")
async def get_location():
    cams = list(mongo_manager.collection_camera.find())
    locations = []
    for cam in cams:
        location = cam["location"]  
        if location not in locations:
            locations.append(location)
    return Response(200, entities={"locations": locations})