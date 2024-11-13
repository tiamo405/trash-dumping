import cv2
import os
import torch
import sys
import numpy as np
import pandas as pd
import argparse
root = os.getcwd()
pwd = os.path.dirname(os.path.realpath("src"))
sys.path.insert(0, root)
from cropvideo import crop_video_to_image

from ultralytics import YOLO
# from deep_sort.tracker import Tracker
from util import find_file
class Preprocessing :
    def __init__(self, opt):
        self.model_detect_person = YOLO("yolov5n.pt")

        self.model_open_pose = None
        self.dir_data = opt.dir_data
        self.dir_save = opt.dir_save

        self.label = opt.label
    def crop_video(self):
        
        folder_save = self.dir_save
        folder_videos = os.path.join(self.dir_data)

        os.makedirs(folder_save, exist_ok= True)
        os.makedirs(os.path.join(folder_save, 'rgb-images', self.label), exist_ok= True)
        os.makedirs(os.path.join(folder_save, 'labels', self.label), exist_ok= True)

        name_videos = sorted(os.listdir(folder_videos))

        for name_video in name_videos :
            # if find_file(nameFile= name_video, nameFolder= os.path.join(folder_save, 'rgb-images', self.label)) :
            #     continue
            print('path video: ',os.path.join(folder_videos, name_video))
            
            crop_video_to_image(path_video         = os.path.join(folder_videos, name_video),\
                            folder_save           = folder_save, \
                            label               = self.label,
                            model_detect_person = self.model_detect_person,\
                            model_open_pose     = self.model_open_pose)

    def crop_one_video(self, path_video = ""):

        crop_video_to_image(path_video         = path_video,\
                            folder_save           = self.dir_save, \
                            label               = self.label,
                            model_detect_person = self.model_detect_person,\
                            model_open_pose     = self.model_open_pose)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', type=str, default= 'trash/video/Webcam')
    parser.add_argument('--label', type=str, default='None')
    parser.add_argument('--weight_yolo', type= str, default='checkpoints/yolov5nu.pt')
    parser.add_argument('--dir_save', type= str, default='trash/predata')
    opt = parser.parse_args()
    return opt
if __name__ == "__main__":

    opt = get_opt()
    print('\n'.join(map(str,(str(opt).split('(')[1].split(',')))))
    prepro = Preprocessing(opt)
    prepro.crop_video()
    # prepro.crop_one_video(path_video="trash/videos/video_split/video1_00087.mp4")

    