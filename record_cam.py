import threading
import cv2
import numpy as np
import time
import os
from datetime import datetime
from utils import time_utils
# import config
from config import VIDEO_PATH, DEBUG
# import logs
from logs import setup_logger
# import storage
import storage.s3_minio as s3
import storage.mongo as mongo

MongoDBManager = mongo.MongoDBManager()
S3Minio = s3.S3Minio()


def open_rtsp_camera(rtsp_url, logger_cam, retry_interval=5, max_retries=5, stop_cam = False):
    retries = 0

    while (retries < max_retries):
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

        if cap.isOpened():
            print(f"Camera {rtsp_url} is opened.")
            logger_cam.info(f"Camera {rtsp_url} is opened.")
            return cap
        else:
            print(f"Error: Could not open camera {rtsp_url}. Retrying in {retry_interval} seconds...")
            logger_cam.error(f"Error: Could not open camera {rtsp_url}. Retrying in {retry_interval} seconds...")
            if stop_cam:
                retries += 1
            cap.release()
            time.sleep(retry_interval)

    print(f"Error: Could not open camera {rtsp_url} after multiple attempts. Exiting.")
    logger_cam.error(f"Error: Could not open camera {rtsp_url} after multiple attempts. Exiting.")
    # exit(1)
    return None


def main(camData, logger_cam, record_duration=300, overlap_time=30):
    
    rtsp_url = camData['rtsp_cam']
    # Kết nối tới RTSP stream
    cap = open_rtsp_camera(rtsp_url, logger_cam, stop_cam = False)
    fps = int(cap.get(cv2.CAP_PROP_FPS))


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Tính toán số frame cần ghi cho mỗi video và số frame cho đoạn trùng lặp
    total_frames_to_record = int(record_duration * fps)
    overlap_frames_count = int(overlap_time * fps)

    overlap_frames = []  # Lưu trữ các frame trùng lặp
    start_time_tmp = None
    while True:
        # Tạo đối tượng VideoWriter để ghi video
        start_time = start_time_tmp if start_time_tmp is not None else time_utils.get_current_timestamp() # Thời gian bắt đầu ghi video
        print(f"Start recording video at {start_time}")
        out_filename = str(camData['_id']) + "_" + str(start_time) + '.mp4'
        out = cv2.VideoWriter(out_filename, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        # Nếu có các frame trùng lặp từ video trước, ghi chúng vào video hiện tại
        if overlap_frames:
            for frame in overlap_frames:
                out.write(frame)

        overlap_frames = []  # Reset danh sách frame trùng lặp cho lần tiếp theo

        frame_count = 0
        while frame_count < (total_frames_to_record - overlap_frames_count):
            ret, frame = cap.read()
            if not ret:
                print(f"Camera {rtsp_url} failed to grab frame. Retrying...")
                logger_cam.error(f"Camera {rtsp_url} failed to grab frame. Retrying...")
                cap.release()
                cap = open_rtsp_camera(rtsp_url, logger_cam=logger_cam)
                continue

            # Ghi frame vào video hiện tại
            out.write(frame)
            frame_count += 1

            # Lưu frame vào danh sách để sử dụng cho đoạn trùng lặp overlap_time giây
            overlap_frames.append(frame)
            if len(overlap_frames) > overlap_frames_count:  # Giữ lại overlap_time giây cuối 
                overlap_frames.pop(0)

            
        # Ghi tiếp 10 giây trùng lặp cho video hiện tại
        start_time_tmp = time_utils.get_current_timestamp() # Cập nhật thời gian bắt đầu cho video tiếp theo
        print(f"Start recording overlap frames at {start_time_tmp}")
        overlap_frame_index = 0
        while overlap_frame_index < overlap_frames_count:
            ret, frame = cap.read()
            if not ret:
                print(f"Camera {rtsp_url} failed to grab frame. Retrying...")
                logger_cam.error(f"Camera {rtsp_url} failed to grab frame. Retrying...")
                cap.release()
                cap = open_rtsp_camera(rtsp_url, logger_cam=logger_cam)
                continue
            out.write(frame)
            overlap_frames.append(frame)
            if len(overlap_frames) > overlap_frames_count:
                overlap_frames.pop(0)
            overlap_frame_index += 1

        out.release()
        
        # save to MongoDB
        video_data = {
            "camera_id": str(camData['_id']),
            "video_path": out_filename,
            "start_time": start_time,
            "end_time": time_utils.get_current_timestamp(),
            "date_time": time_utils.get_date_timestamp()
        }
        print(video_data)
        MongoDBManager.insert_video(video_data)
        
        # Upload to S3 in a separate thread
        s3_thread = threading.Thread(target=S3Minio.upload_file, args=(out_filename, out_filename, True))
        s3_thread.start()
        
        print(f"Video {out_filename} đã lưu xong.")
        logger_cam.info(f"Video {out_filename} đã lưu xong.")

if __name__ == "__main__":
    # Địa chỉ RTSP của camera
    camData = MongoDBManager.get_camera_by_rtsp(VIDEO_PATH)
    if camData is None:
        print(f"Không tìm thấy camera với RTSP URL: {VIDEO_PATH}")
        exit(1)
    camid = str(camData['_id'])
    logger_cam = setup_logger(f"record_cam_{camid}.log")
    main(camData = camData, logger_cam = logger_cam)