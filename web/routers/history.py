import os, sys

from bson import ObjectId
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
        totalPage = len(violation_images) // limit + 1
        for violation in paginated_data:
            violation["_id"] = str(violation["_id"])
            violation["url_image"] = s3.geturl(violation["_id"] + ".jpg")
            violation["location"] = camera["location"]
        return Response(200, entities={"rtsp_url": rtsp_url, "violations": paginated_data, "page": page, "limit": limit, "totalPage": totalPage})

@view_router.get("/get_history_video_violation")
async def get_history_video(id_image_violation: str , page: int = 1, limit: int = 10):
    image_data = mongo_manager.collection_violation_image.find_one({"_id": ObjectId(id_image_violation)})
    camera_id = image_data["camera_id"]
    date_violation = image_data["violation_date"]
    detect_timestamp = image_data["detect_timestamp"]
    # get all video in date camera
    all_videos = list(mongo_manager.collection_video.find({"camera_id": camera_id, "date_time": date_violation}))
    # phan trang
    start_index = (page - 1) * limit
    end_index = start_index + limit
    paginated_data = all_videos[start_index:end_index]
    
    video_violations = []
    for video in paginated_data:
        start_time = video["start_time"]
        end_time = video["end_time"]
        if detect_timestamp >= start_time and detect_timestamp <= end_time:
            video["_id"] = str(video["_id"])
            video["url_video"] = s3.geturl(video["video_path"])
            video_violations.append(video)
    totalPage = len(video_violations) // limit + 1
    return Response(200, entities={"videos": video_violations, "page": page, "limit": limit, "totalPage": totalPage})

# get all violation has is_violation = True
@view_router.get("/get_all_violation")
async def get_all_violation(page: int = 1, limit: int = 10):
    violation_images = list(mongo_manager.collection_violation_image.find({"is_violation": "True"}))
    # phan trang
    start_index = (page - 1) * limit
    end_index = start_index + limit
    paginated_data = violation_images[start_index:end_index]
    totalPage = len(violation_images) // limit + 1
    for violation in paginated_data:
        violation["_id"] = str(violation["_id"])
        violation["url_image"] = s3.geturl(violation["_id"] + ".jpg")
        camera = mongo_manager.get_camera_by_id(violation["camera_id"])
        violation["location"] = camera["location"]
    return Response(200, entities={"violations": paginated_data, "page": page, "limit": limit, "totalPage": totalPage})

# set is_violation = True or False
@view_router.post("/set_violation")
async def set_violation(id_image: str , is_violation: str):
    violation = mongo_manager.collection_violation_image.find_one({"_id": ObjectId(id_image)})
    violation["is_violation"] = is_violation
    mongo_manager.collection_violation_image.update_one({"_id": violation["_id"]}, {"$set": violation})
    return Response(200, entities={"id_image": id_image, "is_violation": "True"})

# delete violation_image
@view_router.post("/delete_violation")
async def delete_violation(id_image: str):
    violation = mongo_manager.collection_violation_image.find_one({"_id": ObjectId(id_image)})
    s3.delete_file(id_image + ".jpg")
    mongo_manager.collection_violation_image.delete_one({"_id": violation["_id"]})
    return Response(200, entities={"id_image": id_image})