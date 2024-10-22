import numpy as np
import os
import glob
import cv2

rgb_images = '/home/server/namtp/code/trash-dumping/trash/ucf24/rgb-images/Diving/v_Diving_g01_c01/'
labels = rgb_images.replace('rgb-images', 'labels')
save = labels.replace('labels', 'test-data')
os.makedirs(save, exist_ok=True)

for file in os.listdir(labels):
    # read file txt
    path = os.path.join(labels, file)
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            x1, y1, x2, y2 = int(float(line[1])), int(float(line[2])), int(float(line[3])), int(float(line[4]))
            img = cv2.imread(os.path.join(rgb_images, file.replace('txt', 'jpg')))
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(save, file.replace('txt', 'jpg')), img)
            
