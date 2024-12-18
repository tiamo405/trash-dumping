import numpy as np
import os
import cv2
import json
def point_object(res, label = [0]) :
    person_locations = []
    for i in range(len(res.xyxy[0])):
        if int(res.xyxy[0][i][5]) in label:
            l, t, r, b=int(res.xyxy[0][i][0]), int(res.xyxy[0][i][1]),\
            int(res.xyxy[0][i][2]), int(res.xyxy[0][i][3])
            if r - l > 50 and b - t > 50 :
                person_locations.append((l,t,r,b))
    return person_locations

def search_id(path) :
    fnames = os.listdir(path)
    dem = 0
    for fname in fnames :
        if '.jpg' in fname or '.png' in fname :
            dem += 1
    return dem + 1

def draw(image, pose, left = 0, top= 0) :
    BODY_25_LINES = [
    [17, 15, 0, 1, 8, 9, 10, 11, 22, 23],  # Right eye down to right leg
    [11, 24],  # Right heel
    [0, 16, 18],  # Left eye
    [4, 3, 2, 1, 5, 6, 7],  # Arms
    [8, 12, 13, 14, 19, 20],  # Left leg
    [14, 21]  # Left heel
]
    points = np.array(pose[0])[:, 0:2]
    # for (x,y) in (points) :
        
    # print(points)
    for lines in BODY_25_LINES :
        for i in range(len(lines)-1):
            (xpt1, ypt1) = points[lines[i]]
            (xpt2, ypt2) = points[lines[i+1]]
            if (xpt1, ypt1) == (0, 0) or (xpt2, ypt2) == (0, 0):
                continue
            cv2.line(image, (int(xpt1)+left , int(ypt1)+top), \
                     (int(xpt2)+ left, int(ypt2)+ top), color= (255,0,0), thickness= 1)
    return image

def read_json(path):
    with open(path, 'r') as f :
        pose = json.load(f)
        pose_data = pose['pose_keypoints_2d']
        pose_data = np.array(pose_data)
    return pose_data
    
def createjson(poses, path_json, width, height):
    tmp = list(np.array(poses[0]).reshape(75))
    arr = []
    for i in range(len(tmp)) :
        if i % 3 == 0 :
            arr.append(tmp[i] / width)
        elif i%3 ==1:
            arr.append(tmp[i]/ height)
    data_dict={
        # "pose_keypoints_2d":list(np.array(poses[0]).reshape(75)),
        "pose_keypoints_2d":arr, # 25 toa do (x,y)
        'width' : width,
        'height': height
    }
    data_string = json.dumps(data_dict)
    myjsonfile = open(path_json, "w")
    myjsonfile.write(data_string)
    myjsonfile.close()

def find_file(nameFile, nameFolder) :
    folders = os.listdir(nameFolder)
    for folder in folders :
        if nameFile.split('.')[0] in folder.split('-') :
            return True
    return False

def write_txt(noidung, path, remove = False):
    if os.path.exists(path) and remove == True:
        os.remove(path)
    with open(path, 'a') as f:
        f.write("%s\n" % noidung)
        f.close()
if __name__ =="__main__":
    if find_file(nameFile='VID_20230310_161738.mp4', nameFolder='data/train/datasets/labels/Littering') :
        print(1)
    else : print(0)