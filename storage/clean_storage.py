# xoa cac video ma k có ai pham loi khi bat dau 1 ngay moi
import os, sys
root = os.getcwd()
sys.path.append(root)
import datetime
import time
from utils import time_utils
from bson import ObjectId
from storage.mongo import MongoDBManager
from storage.s3_minio import S3Minio
mongo_manager = MongoDBManager()
s3 = S3Minio()

def check_video_has_violation(video, violation_images):
    for violation in violation_images:
        detect_timestamp = violation["detect_timestamp"]
        start_time_video = video["start_time"]
        end_time_video = video["end_time"]
        if detect_timestamp >= start_time_video and detect_timestamp <= end_time_video:
            return True
    return False

def delete_videos_without_violations(date_delete=None):
    # kiem tra xem co phai la ngay moi k hoac neu truyen tham so la muon xoa ngay nao
    now = datetime.datetime.now()
    # xoa cac video ma k có ai pham loi
    all_cameras = mongo_manager.get_all_cameras()
    for camera in all_cameras:
        camera_id = str(camera["_id"])
        # xet ngay hom truoc hoac ngay truyen vao
        yesterday  = date_delete if date_delete else (now - datetime.timedelta(days=1)).timestamp()
        # get all loi trong ngay hom truoc
        violation_images = list(mongo_manager.collection_violation_image.find({"camera_id": camera_id, "violation_date": yesterday}))
        # get all video trong ngay hom truoc
        all_videos = list(mongo_manager.collection_video.find({"camera_id": camera_id, "date_time": yesterday}))
        for video in all_videos:
            if not check_video_has_violation(video, violation_images):
                # xoa video
                video_id = str(video["_id"])
                video_path = video["video_path"]
                mongo_manager.collection_video.delete_one({"_id": ObjectId(video_id)})
                s3.delete_file(video_path)
                print("delete video: ", video_path)

def main(date_delete=None):
    if date_delete:
        delete_videos_without_violations(date_delete)
        exit()
    while True:
        now = datetime.datetime.now()
        # Kiểm tra nếu 1h sáng den 2h59p thì thực hiện việc xóa
        if now.hour == 1 or now.hour == 2:
            delete_videos_without_violations()
            print("Hoàn thành việc xóa video không có lỗi vi phạm vào lúc 1h sáng")
            time.sleep(60 * 60 * 24)  # Đợi 24 giờ trước khi kiểm tra lại

        # Chờ thêm 1 phút để tiết kiệm tài nguyên thay vì kiểm tra liên tục
        time.sleep(60)
if __name__ == "__main__":
    # xoa video ngay hom truoc
    date_delete = 1729962000
    main()