import cv2
import os
import pandas as pd
import numpy as np

def demsolan(a1, a2):
    a1 = np.array(a1)
    a2 = np.array(a2)
    # Sử dụng hàm np.isin() để kiểm tra số lần xuất hiện
    occurrences = np.sum(np.isin(a1, a2))
    print("Số lần xuất hiện của các phần tử trong a2 trong a1:", occurrences)
    print("ti le: ", occurrences/len(a2))
    print("=================================")
    

thucte = {
    "VID_20230317_161005": [55,85],
    "VID_20230317_161015":	[95	,110],
    "VID_20230317_161024"	:[145	,167] ,
    "VID_20230317_161037"	:[238	,248] ,
    "VID_20230317_161048"	:[235,	275],
    "VID_20230317_161110"	:[120	,142],
    "VID_20230317_161130"	:[180	,237],
    "VID_20230317_161145"	:[0	,0],
    "VID_20230317_161159"	:[328	,373],
    "VID_20230317_161222"	:[164	,187],
    "VID_20230317_161232"	:[120,	141]
}
path = "test_model/outputs/model"
print(thucte["VID_20230317_161005"])
for item in thucte.items() :

    namevideo, start_tt, end_tt = item[0], item[1][0], item[1][1]
    print(namevideo)
    dir_label = os.path.join(path, namevideo,"trash")
    label_model = sorted(os.listdir(dir_label))
    if len(label_model) == 0:
        print("Số lần xuất hiện của các phần tử trong a2 trong a1:", 0)
        print("ti le: ", 0)
        continue
    start_model = int(label_model[0].split('.')[0])
    end_model = int(label_model[-1].split('.')[0])

    a1 = []
    for i in range(start_tt, end_tt+1):
        a1.append(i)
    a2=[]
    for num in range(start_model, end_model+1) :
        a2.append(num)
    demsolan(a1, a2)
    
    