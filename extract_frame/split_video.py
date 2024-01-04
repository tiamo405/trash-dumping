import cv2
import numpy as np
import os
import glob
import argparse

def split_video(path_video, duration_per_video = 20):# Độ dài (số giây) của video con
    cap = cv2.VideoCapture(path_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 10
    name_video = os.path.basename(path_video).split('.')[0]
    

    # Lấy tổng số khung hình trong video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tổng số khung hình cho mỗi video con (với độ dài duration_per_video giây)
    frames_per_video = int(duration_per_video * fps)
    
    # Biến đếm số video con đã tạo
    video_count = 1
    # Biến đếm số khung hình đã ghi vào video con hiện tại
    frames_written = 0

    # Biến lưu trữ tên file video con hiện tại
    output_folder = os.path.join("trash", "videos", "video_split")
    os.makedirs(output_folder, exist_ok= True)
    output_video = os.path.join(output_folder, '{}_{}.avi'.format(name_video, str(video_count).zfill(5)))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video, fourcc, 
            fps, (width, height))

    while(cap.isOpened()) :
        ret, frame = cap.read()
        if frame is None:
            break
        frames_written += 1
        out.write(frame)
        if frames_written == frames_per_video:
            
            # Đóng video con hiện tại và chuyển sang video con mới
            out.release()
            video_count += 1
            output_video = os.path.join(output_folder, '{}_{}.avi'.format(name_video, str(video_count).zfill(5)))
            out = cv2.VideoWriter(output_video, fourcc, 
                    fps, (width, height))
            frames_written = 0
        # Giải phóng đối tượng VideoCapture và VideoWriter

    cap.release()
    out.release()

    print("Tách video thành các video nhỏ thành công.")

def convert(path, new_path) :
    video_capture = cv2.VideoCapture(path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Định dạng video mới là MP4
    video_writer = cv2.VideoWriter(new_path, fourcc, fps, (frame_width, frame_height))
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # Hết video, thoát khỏi vòng lặp

        # Xử lý khung hình ở đây nếu muốn

        # Ghi khung hình vào video mới
        video_writer.write(frame)

    # Giải phóng các đối tượng
    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()

def convert_avi2mp4(opt):
    dir_folder = opt.dir_folder
    dir_folder = os.path.join(dir_folder, "videos", "video_split")
    file_list = list(glob.glob(f"{dir_folder}/*"))
    for file in file_list :
        new_file = str(file).replace('.avi', '.mp4') 
        print(new_file)
        convert(file, new_file)
        os.remove(file)
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_folder", type= str, default= "trash")
    parser.add_argument('--path_video', type=str, default= 'trash/video.mp4')
    parser.add_argument('--duration_per_video', type= int, default= 20)
    opt = parser.parse_args()
    return opt
if __name__ == "__main__":

    opt = get_opt()
    print('\n'.join(map(str,(str(opt).split('(')[1].split(',')))))
    # split_video(opt.path_video, opt.duration_per_video)
    convert_avi2mp4(opt)
