import cv2
import os
import json
import numpy as np
import copy
import sys
import random

from pathlib import Path
from deep_sort.tracker import Tracker
root = os.getcwd()
pwd = os.path.dirname(os.path.realpath("sort"))
sys.path.insert(0, root)
# from sort.sort import Sort 

# from preprocessing.util import point_object, search_id, draw, createjson
from util import write_txt

def crop_video_to_image(path_video, folder_save, label, model_detect_person, model_open_pose) :
    detection_threshold = 0.8

    name_video = (path_video.split('/')[-1].split('.')[0]).zfill(5)
    print(f'name video : {name_video}')


    path_save_image = os.path.join(folder_save, 'rgb-images', label)
    path_save_txt = os.path.join(folder_save, 'labels', label)

    path_save_video = os.path.join(folder_save, 'video_detect', label)

    os.makedirs(path_save_video, exist_ok= True)
    video_out_path = os.path.join(path_save_video, name_video+'.avi')
    cap = cv2.VideoCapture(path_video)
    tracker = Tracker()
    frame_width = 640
    frame_height = 480
    cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MJPG'), cap.get(cv2.CAP_PROP_FPS),
                          (frame_width, frame_height))
    font = cv2.FONT_HERSHEY_DUPLEX
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(20)]
    results = None
    while(cap.isOpened()) :
        ret, frame = cap.read()
        if frame is None:
            break
        frame = cv2.resize(frame, (frame_width, frame_height))
        results = model_detect_person(frame)
        # person_locations = point_object(results)
        for result in results:
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                if score > detection_threshold and class_id == 0:
                    detections.append([x1, y1, x2, y2, score])
            try:
                tracker.update(frame, detections)
            except :
                break

            for track in tracker.tracks: # tung ng trong 1 frame
                bbox = track.bbox
                left, top, right, bottom = bbox
                # left, top, right, bottom = int(left / rate), int(top/rate), min(int(right*rate), frame_width - 5), min(int(bottom * rate), frame_height - 5)
                track_id = track.track_id
            
                # image = frame[top:bottom, left: right]
                if bottom - top < 20 or right - left < 20 :
                    continue

                path_save_img_id = os.path.join(path_save_image, name_video + '-'+ (str(track_id).zfill(5)))
                path_save_labels_id = os.path.join(path_save_txt, name_video + '-'+ (str(track_id).zfill(5)))

                os.makedirs(path_save_img_id, exist_ok= True)

                os.makedirs(path_save_labels_id, exist_ok= True)
                

                id_frame = len(os.listdir(path_save_img_id))
                cv2.imwrite(os.path.join(path_save_img_id, str(id_frame).zfill(5) + '.jpg'), frame)
                # print(f'save image: name video: {name_video}, id:{track_id}, id_frame:{id_frame}')
                if label == 'trashDumping' :
                    write_txt(noidung= '1 {} {} {} {}'.format(left, top, right, bottom),\
                            path= os.path.join(path_save_labels_id, str(id_frame).zfill(5) + '.txt'))
                else :
                    write_txt(noidung= '2 {} {} {} {}'.format(left, top, right, bottom),\
                            path= os.path.join(path_save_labels_id, str(id_frame).zfill(5) + '.txt'))
                
                # ve de kiem tra detect cua video
                cv2.putText(frame, 'frame: '+ str(id_frame), (30, 30), font, 1.0, (0,255,0), 1)
                cv2.putText(frame, str(track_id), (int(left) - 6, int(top) + 30), font, 1.0, (colors[track_id % len(colors)]), 1)
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (colors[track_id % len(colors)]), 3)

        # cv2.imshow("Image", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break  
        cap_out.write(frame)
    cap.release()
    cap_out.release()
