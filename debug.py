# from datetime import date

# print(str(date.today()))

# import cv2

# # Đường dẫn tới video gốc
# input_video_path = 'trash/videos/video_split/video1_00006.avi'

# # Đường dẫn đến video mới sau khi lưu
# output_video_path = 'video1_00006.mp4'

# # Tạo đối tượng VideoCapture để đọc video gốc
# video_capture = cv2.VideoCapture(input_video_path)

# # Lấy các thông số kỹ thuật của video gốc
# fps = video_capture.get(cv2.CAP_PROP_FPS)
# frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Tạo đối tượng VideoWriter để ghi video mới
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Định dạng video mới là MP4
# video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# # Đọc từng khung hình của video gốc, xử lý (nếu cần), và ghi vào video mới
# while True:
#     ret, frame = video_capture.read()
#     if not ret:
#         break  # Hết video, thoát khỏi vòng lặp

#     # Xử lý khung hình ở đây nếu muốn

#     # Ghi khung hình vào video mới
#     video_writer.write(frame)

# # Giải phóng các đối tượng
# video_capture.release()
# video_writer.release()
# cv2.destroyAllWindows()

import os
import cv2
import numpy as np
from logs import setup_logger

logg = setup_logger('test', debug=True)
logg.debug('debug')

# change is_violation to True
# from storage.mongo import MongoDBManager
# mongo_manager = MongoDBManager()
# images = list(mongo_manager.collection_violation_image.find())
# for image in images:
#     image["is_violation"] = False
#     mongo_manager.collection_violation_image.update_one({"_id": image["_id"]}, {"$set": image})

# from storage.s3_minio import S3Minio
# s3 = S3Minio()
# url = s3.geturl('ffmpeg_reencoded.mp4')
# print(url)

# import cv2

# cap = cv2.VideoCapture('671b0ca3b236731f0056e606_1730346216.avi')
# fps = cap.get(cv2.CAP_PROP_FPS)
# # ghi lai video
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('output.avi', fourcc, fps, (int(cap.get(3)), int(cap.get(4)))
# )
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     out.write(frame)
# cap.release()
# out.release()

# import cv2
# import time

# def record_video_without_audio(rtsp_url, output_file, duration=600):
#     # Kết nối tới camera
#     cap = cv2.VideoCapture(rtsp_url)

#     # Kiểm tra kết nối camera
#     if not cap.isOpened():
#         print("Không thể kết nối tới camera.")
#         return

#     # Lấy thông tin về FPS và kích thước khung hình
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     print(f"FPS: {fps}")
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Cấu hình ghi video (không ghi âm thanh)
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#     out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

#     start_time = time.time()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Không thể lấy khung hình từ camera.")
#             break

#         # Ghi frame vào file video
#         out.write(frame)

#         # Kiểm tra thời gian ghi
#         if time.time() - start_time > duration:
#             print("Ghi video hoàn tất.")
#             break

#     # Giải phóng tài nguyên
#     cap.release()
#     out.release()
#     print(f"Video đã được lưu tại: {output_file}")

# # Ví dụ sử dụng
# rtsp_url = "rtsp://cxview:gs252525@113.161.58.13:554/Streaming/Channels/701"
# output_file = "output.avi"
# record_video_without_audio(rtsp_url, output_file, duration=10)  # Ghi 10 phút

# tai tat ca file mp4 tren minio roi encode ffmpeg va luu lai
# import os
# import ffmpeg
# from storage.s3_minio import S3Minio
# import cv2
# import numpy as np
# from storage.mongo import MongoDBManager

# s3 = S3Minio()
# mongo_manager = MongoDBManager()
# videos = list(mongo_manager.collection_video.find())
# for video in videos:
#     video_path = video["video_path"]
#     url = s3.geturl(video_path)
#     if url:
#         # download video
#         s3.download_file(video_path, video_path)
#         # encode video
#         reencoded_filename = video_path.replace('.mp4', '_reencoded.mp4')
#         try:
#             ffmpeg.input(video_path).output(
#                 reencoded_filename, 
#                 vcodec='libx264', 
#                 preset='fast', 
#                 movflags='+faststart', 
#                 an=None
#             ).run(quiet = True)
#             print(f"Tái mã hóa video thành công: {reencoded_filename}")
#             # os.remove(video_path)
#             # # update video to mongo and s3
#             # video["video_path"] = reencoded_filename
#             # mongo_manager.collection_video.update_one({"_id": video["_id"]}, {"$set": video})
#             # s3.upload_file(file_path=reencoded_filename, object_name=reencoded_filename, is_remove=True)
#         except Exception as e:
#             print(f"Lỗi khi tái mã hóa video: {e}")
#             continue

# from storage.mongo import MongoDBManager
# from storage.s3_minio import S3Minio
# from bson import ObjectId
# mongo_manager = MongoDBManager()
# s3 = S3Minio()

# def get_history_video(id_image_violation: str , page: int = 1, limit: int = 10):
#     image_data = mongo_manager.collection_violation_image.find_one({"_id": ObjectId(id_image_violation)})
#     camera_id = image_data["camera_id"]
#     date_violation = image_data["violation_date"]
#     detect_timestamp = image_data["detect_timestamp"]
#     # get all video in date camera
#     all_videos = list(mongo_manager.collection_video.find({"camera_id": camera_id, "date_time": date_violation}))
#     # phan trang
#     # start_index = (page - 1) * limit
#     # end_index = start_index + limit
#     # paginated_data = all_videos[start_index:end_index]
    
#     video_violations = []
#     for video in all_videos:
#         start_time = video["start_time"]
#         end_time = video["end_time"]
#         if str(video['_id']) == "6724939eb080eb3bccfe1eb4":
#             print(video)
#         if detect_timestamp >= start_time and detect_timestamp <= end_time:
#             video["_id"] = str(video["_id"])
#             video["url_video"] = s3.geturl(video["video_path"])
#             video_violations.append(video)
#     totalPage = len(video_violations) // limit + 1
#     print(video_violations)
# get_history_video("6724925600a13dc48251d2f4")
# image = mongo_manager.collection_violation_image.find_one({"_id": ObjectId("6724925600a13dc48251d2f4")})
# print(image)
# video = mongo_manager.collection_video.find_one({"_id": ObjectId("6724939eb080eb3bccfe1eb4")})
# print(video)

# import ffmpeg

# video = "/workspace/671b0ca3b236731f0056e606_1730777028.avi"
# encode_path = video.replace('.avi', '_reencoded.mp4')
# try:
#     ffmpeg.input(video).output(
#         encode_path, 
#         vcodec='libx264', 
#         preset='fast', 
#         movflags='+faststart', 
#         an=None
#     ).run(quiet = True)
#     print(f"Tái mã hóa video thành công: {encode_path}")
# except Exception as e:
#     print(f"Lỗi khi tái mã hóa video: {e}")

import uuid
print(str(uuid.uuid4().hex))

# label = '/home/server/namtp/code/trash-dumping/trash/predata/labels/None/VID_20230310_163540-00001/00040.txt'
# pathimg = label.replace('.txt', '.jpg').replace('labels', 'rgb-images')

# boxs = np.loadtxt(label)
# print(boxs)
# img = cv2.imread(pathimg)
# left, top, right, bottom = int(boxs[1]), int(boxs[1]), int(boxs[3]), int(boxs[4])
# cv2.rectangle(img, (left, 0), (right, bottom), (0, 255, 0), 2)
# cv2.imwrite('d.jpg', img)
# for i in range(1, len(boxs)):
#     if boxs[i] < 0 :
#         boxs[i] = 0
# #save boxs to label cach 1 dau cach
# boxs[1:] = np.round(boxs[1:])
# print(boxs)
# with open('text.txt', 'w') as f:
#     # Lưu phần tử đầu tiên (không có phần thập phân)
#     f.write(f"{boxs[0]:.0f}")
    
#     # Lưu các phần tử còn lại với một chữ số thập phân
#     for value in boxs[1:]:
#         f.write(f" {value:.1f}")

# labels = ['Normal', 'Littering']
# raw_data = '/home/server/namtp/code/trash-dumping/trash/'
# for label in labels:
#     fvideos = os.listdir(os.path.join(raw_data, 'labels', label))
#     for fvideo in fvideos:
#         print(fvideo)
#         txts_file = os.listdir(os.path.join(raw_data, 'labels', label, fvideo))
#         for txt_file in txts_file:
#             bbox = np.loadtxt(os.path.join(raw_data, 'labels', label, fvideo, txt_file))
#             print(bbox)


#-----------------------------------------------------------------------------------------
# i have 1 camera rtsp, i want to record video from this camera 30s and save to local
# import cv2
# import time
# import datetime

# def save_rtsp_video(rtsp_url, output_dir, segment_duration=30):
#     # Kết nối camera RTSP
#     cap = cv2.VideoCapture(rtsp_url)
    
#     if not cap.isOpened():
#         print("Không thể kết nối với camera.")
#         return
    
#     # Lấy thông tin về FPS và kích thước frame
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec để lưu video
    
#     start_time = time.time()
#     segment_count = 0
    
#     while True:
#         segment_start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_file = f"{output_dir}/video_{segment_start_time}.avi"
        
#         out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
        
#         # Ghi video trong thời gian segment_duration giây
#         while time.time() - start_time < segment_duration:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Không đọc được frame từ camera.")
#                 break
#             out.write(frame)
        
#         # Đóng file video hiện tại
#         out.release()
#         print(f"Đã lưu: {output_file}")
        
#         # Reset thời gian bắt đầu cho đoạn video tiếp theo
#         start_time = time.time()
#         segment_count += 1

#     cap.release()

# # Cấu hình
# rtsp_url = "rtsp://cxview:gs252525@113.161.58.13:554/Streaming/Channels/701"
# output_dir = "videos_recorded"
# save_rtsp_video(rtsp_url, output_dir)


            
#---------------------
# rename file video in folder
# folder_path = '/home/server/namtp/code/trash-dumping/trash/video/New1703/'
# videos = os.listdir(folder_path)
# for i, video in enumerate(videos):
#     if video.endswith('.mp4'):
#         # new name = stt
#         new_name = f'New1703_{i}.mp4'
#         os.rename(os.path.join(folder_path, video), os.path.join(folder_path, new_name))
#         print(f'Rename {video} to {new_name}')

pathlabel = 'results/ucf_detections/yowo_v2_medium/detections_1/Littering_VID_20230310_162034-00001_00010.txt'
pathimg = 'trash/rgb-images/Littering/VID_20230310_162034-00001/00010.jpg'
img = cv2.imread(pathimg)

with open(pathlabel, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        labelclass, conf, x1, y1, x2, y2 = line[0], float(line[1]), int(line[2]), int(line[3]), int(line[4]), int(line[5])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, labelclass + '_' +str(conf), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imwrite('d.jpg', img)