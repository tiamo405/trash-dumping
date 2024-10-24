import cv2
import numpy as np
import time
import os
from datetime import datetime
# import config
from config import VIDEO_PATH, DEBUG
# import storage
import storage.s3_minio as s3
import storage.mongo as mongo

MongoDBManager = mongo.MongoDBManager()
S3Minio = s3.S3Minio()




# Địa chỉ RTSP của camera
rtsp_url = 'test.mp4'

# Thời gian ghi video (tính theo giây)
record_duration = 1 * 60  # 1 phút
overlap_time = 10  # 30 giây trùng

# Kết nối tới RTSP stream
cap = cv2.VideoCapture(rtsp_url)
fps = cap.get(cv2.CAP_PROP_FPS)

# Kiểm tra kết nối
if not cap.isOpened():
    print("Không thể kết nối tới camera RTSP.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

def get_output_filename():
    """Tạo tên file video dựa trên thời gian hiện tại"""
    return datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"

overlap_frames = []  # Lưu trữ các frame trùng lặp

while True:
    # Tạo đối tượng VideoWriter để ghi video
    out_filename = get_output_filename()
    out = cv2.VideoWriter(out_filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Nếu có các frame trùng lặp từ video trước, ghi chúng vào video hiện tại
    if overlap_frames:
        for frame in overlap_frames:
            out.write(frame)

    overlap_frames = []  # Reset danh sách frame trùng lặp cho lần tiếp theo

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể nhận frame từ camera.")
            break

        # Ghi frame vào video hiện tại
        out.write(frame)

        # Lưu frame vào danh sách để sử dụng cho đoạn trùng lặp 30 giây
        overlap_frames.append(frame)
        if len(overlap_frames) > fps * overlap_time:  # Giữ lại 30 giây cuối (nếu 20 fps thì sẽ giữ lại 600 frame)
            overlap_frames.pop(0)

        # Kiểm tra nếu đã ghi đủ thời gian (4 phút 30 giây)
        if time.time() - start_time >= (record_duration - overlap_time):
            break

    # Ghi tiếp 30 giây cho video hiện tại
    overlap_start_time = time.time()
    while time.time() - overlap_start_time < overlap_time:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        overlap_frames.append(frame)
        if len(overlap_frames) > 20 * overlap_time:
            overlap_frames.pop(0)

    out.release()
    print(f"Video {out_filename} đã lưu xong.")

