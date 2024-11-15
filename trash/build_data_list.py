import argparse
import os
import numpy as np
import glob

def build_split_list(raw_path, mode):
    raw_path = os.path.join(raw_path, '{}list01.txt'.format(mode))
    print('{} analysis begin'.format(raw_path))
    with open(raw_path, 'r') as fin:
        lines = fin.readlines()
    fin.close()

    with open('{}list.txt'.format(mode), 'w') as fout:
        for i, line in enumerate(lines):
            line = line.strip()  # 'class_name/video_name'
            label_dir = 'labels' + '/' + line  # 'data/ucf24/labels/class_name/video_name'
            if not os.path.isdir(label_dir):
                continue
            txt_list = os.listdir(label_dir)
            txt_list.sort()
            for txt_item in txt_list:
                filename = label_dir + '/' + txt_item
                fout.write(filename + '\n')
            if i % 200 == 0:
                print('{} videos parsed'.format(i))
    fout.close()
    print('{} analysis done'.format(raw_path))

def create_txt(raw_data, label, arr_videos, file_txt) :

    with open(file_txt, 'a') as fout:
        for video in arr_videos :
            file_list = os.listdir(os.path.join(raw_data, 'labels', label, video))
            file_list.sort()
            for file in file_list :
                if int(file.split('.')[0]) >= 8 and int(file.split('.')[0]) < 300:
                    path = os.path.join('labels', label, video, file)
                    fout.write(path + '\n')
    fout.close()

def build_split_data(raw_data) :
    # labels = os.listdir(os.path.join(raw_data, 'labels')) # lay cac label
    labels = ["Normal", "Littering"]
    train_file_txt = '{}/{}list.txt'.format(raw_data ,'train')
    test_file_txt = '{}/{}list.txt'.format(raw_data ,'test')
    if os.path.exists(train_file_txt):
        os.remove(train_file_txt)
    if os.path.exists(test_file_txt):
        os.remove(test_file_txt)

    for label in labels :

        fvideos = os.listdir(os.path.join(raw_data, 'labels', label))
        N = len(fvideos)
        n_train = int(N * 0.9)

        # np.random.shuffle(fvideos)

        train_video, test_video = fvideos[:n_train], fvideos[n_train:]
        create_txt(raw_data= raw_data, label= label, arr_videos= train_video, file_txt= train_file_txt)
        create_txt(raw_data= raw_data, label= label, arr_videos= test_video, file_txt= test_file_txt)
    print('{} analysis split data done'.format(raw_data))


# def remove_images(raw_data):
#     labels = os.listdir(os.path.join(raw_data, 'labels'))
#     print(labels)
#     for label in labels :
#         print('Start remove label: ', label)
#         dir_images = os.path.join(raw_data, 'rgb-images', label)
#         dir_labels = os.path.join(raw_data, 'labels', label)
        
#         folder_images = sorted(list(glob.glob(f"{dir_images}/*")))
#         folder_labels = sorted(list(glob.glob(f"{dir_labels}/*")))
#         # print(folder_images)
#         for folder_label, folder_image in zip(folder_labels, folder_images):
#             print('start remove folder: ', folder_image)
#             file_txts = list(glob.glob(f"{folder_label}/*"))
#             file_images = list(glob.glob(f"{folder_image}/*"))
#             for file in file_images :
#                 path_file = file.replace('rgb-images','labels')
#                 path_file = path_file.replace('.jpg','.txt')
#                 if path_file not in file_txts :
#                     os.remove(file)

# chang label 0-> 1 for littering and 0 -> 2 for normal
def change_label(raw_data):
    labels = os.listdir(os.path.join(raw_data, 'labels'))
    for label in labels :
        print('Start change label: ', label)
        dir_labels = os.path.join(raw_data, 'labels', label)
        folder_labels = sorted(list(glob.glob(f"{dir_labels}/*")))
        for folder_label in folder_labels:
            print('start change folder: ', folder_label)
            file_txts = list(glob.glob(f"{folder_label}/*"))
            for file in file_txts :
                boxs = np.loadtxt(file)
                if label == 'Normal':
                    boxs[0] = 2
                else:
                    boxs[0] = 1
                for i in range(1, len(boxs)):
                    if boxs[i] < 0 :
                        boxs[i] = 0
                boxs = np.round(boxs)
                with open(file, 'w') as f:
                    f.write(f"{boxs[0]:.0f}")
                    for value in boxs[1:]:
                        f.write(f" {value:.1f}")

    print('{} analysis done'.format(raw_data))    

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument('--raw_data', type=str, default='./trash')
    parser.add_argument('--out_list_path', type=str, default='./')
    parser.add_argument('--shuffle', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    build_split_data(args.raw_data)
    # remove_images(args.raw_data)
    # change_label(args.raw_data)