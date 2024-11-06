from datetime import datetime
import uuid
import cv2
import os, sys
root = os.getcwd()
sys.path.append(root)
import subprocess
from bson import ObjectId
from ..utils.Respones import *
from storage.mongo import MongoDBManager
from storage.s3_minio import S3Minio
from fastapi import APIRouter, Depends, HTTPException, Form
from pydantic import BaseModel

camera_router = APIRouter(prefix="/prod/api/v1/camera", tags=["camera"])

mongo_manager = MongoDBManager()
s3 = S3Minio()

class Camera(BaseModel):
    rtsp_url: str
    location: str
    add_by_customer_id: str = "671b0c1be9383de32aefd299"

def check_camera(rtsp_url):
    # return True
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        cap.release()
        return False
    # get frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return False
    # save to miniostorage
    file_name = uuid.uuid4().hex + ".jpg"
    file_path = os.path.join("temp_file", file_name)
    cv2.imwrite(file_path, frame)
    s3.upload_file(file_path, file_name)
    os.remove(file_path)
    cap.release()
    return file_name

@camera_router.get("/list")
async def list_cam(page: int = 1, limit: int = 5):
    cameras = mongo_manager.get_all_cameras()
    for cam in cameras:
        cam["_id"] = str(cam["_id"])
        if cam["origin_image"]:
            cam["origin_image"] = s3.geturl(cam["origin_image"])
        try:
            cam["add_by"] = mongo_manager.collection_account.find_one({"_id": ObjectId(cam["add_by_customer_id"])})["username"]
        except:
            cam["add_by"] = "Unknown"
    start_index = (page - 1) * limit
    end_index = start_index + limit
    paginated_data = cameras[start_index:end_index]
    total_pages = (len(cameras) + limit - 1) // limit
    return Response(0, entities={"cameras": paginated_data, "total_pages": total_pages})

@camera_router.post("/create")
# async def create_cam(rtsp_url: str, location: str, add_by_customer_id: str = "671b0c1be9383de32aefd299"):
async def create_cam(cameraRequest: Camera):
# async def create_cam(camera: Camera):
    camera = mongo_manager.get_camera_by_rtsp(cameraRequest.rtsp_url)
    if camera: # camera already
        # return Response(101, error_resp=101)
        raise HTTPException(
                status_code=400,
                detail={"error_code": "FS.0101", "message": "Camra already exists."}
            )
    else: # camera not exists
        # check camera
        id_image = check_camera(cameraRequest.rtsp_url)
        if not id_image:
            raise HTTPException(
                status_code=400,
                detail={"error_code": "FS.0102", "message": "Camera doesn't recognize."}
            )
        else:
            new_camera = {
                "rtsp_cam": cameraRequest.rtsp_url,
                "is_activate": True,
                "date_added": datetime.utcnow().timestamp(),
                "location": cameraRequest.location,
                "add_by_customer_id": cameraRequest.add_by_customer_id,
                "origin_image": id_image,
            }
            camera_id = mongo_manager.insert_camera(new_camera)
            name_container_ai = f"ai_{camera_id}"
            name_comtainer_recording = f"recording_{camera_id}"
            # run container name camid
            command_ai = [
                "docker", "run", "--name", name_container_ai, "--net=host", "-dit", "--privileged",
                "-e", f"VIDEO_PATH={cameraRequest.rtsp_url}", 
                "-v", "/home/server/namtp/code/trash-dumping:/workspace",
                "littering:latest"
            ]
            command_recording = [
                "docker", "run", "--name", name_comtainer_recording, "--net=host", "-dit", "--privileged",
                "-e", f"VIDEO_PATH={cameraRequest.rtsp_url}", 
                "-v", "/home/server/namtp/code/trash-dumping:/workspace",
                "recording:latest"
            ]
            result_record = subprocess.run(command_recording, capture_output=True, text=True)
            if result_record.returncode != 0:
                # remove 
                mongo_manager.collection_camera.delete_one({"_id": ObjectId(camera_id)})
                raise HTTPException(status_code=500, detail={"status": "can't run recording container"})
            result_ai = subprocess.run(command_ai, capture_output=True, text=True)
            if result_ai.returncode != 0:
                # remove
                mongo_manager.collection_camera.delete_one({"_id": ObjectId(camera_id)})
                raise HTTPException(status_code=500, detail={"status": "can't run ai container"})
            return Response(200, entities={"rtsp_url": cameraRequest.rtsp_url, "camera_id": camera_id})

@camera_router.delete("/remove")
async def remove_cam(cam_id : str):
    try:
        camera = mongo_manager.collection_camera.find_one({"_id": ObjectId(cam_id)})
        if not camera: # camera not exists
            return Response(201, error_resp=201)
        else: # camera exists
            camera_id = camera["_id"]
            mongo_manager.delete_camera_by_id(camera_id)
            # remove container name camid
            return Response(200, msg="Success remove camera")
    except:
        return Response(1, error_resp=1)


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