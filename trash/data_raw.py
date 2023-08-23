import cv2
import os
import glob
import shutil

def edit_txt(path, label) :
    dict_label = {"Walking": "2",
                  "trashDumping": "1"}
    files = list(glob.glob(f"{path}/*.txt"))
    for file_path in files :
        with open(file_path, 'r') as file:
            lines = file.read()     
            label_number = dict_label[label]  # Lấy số tương ứng từ bảng
            lines = lines.replace(lines[0], label_number)
            with open(file_path, 'w') as file:
                file.writelines(lines)

def remove_folder(folder_1, folder_2):
    subfolder_name = os.path.basename(folder_1)
    subfolder_path = os.path.join(folder_2, subfolder_name)
    # Sao chép thư mục nguồn sang thư mục đích
    shutil.copytree(folder_1, subfolder_path)


dir_raw = "trash/data_raw"
dir_clear = "trash"
labels = ["trashDumping"]
name_folder_raw = "video_split"

def clear_data():
    for label in labels:
        folder_move_label = os.path.join(dir_clear, "labels", label)
        folder_move_image = os.path.join(dir_clear, "rgb-images", label)
        path_folders = list(glob.glob(f"{os.path.join(dir_raw, label)}/*"))
        
        for folder in path_folders :
            name_folder = os.path.basename(folder)
            rgb_folder = os.path.join(dir_clear, "rgb-images", 
                                      name_folder_raw, name_folder)
            print("name folder: ", name_folder)
            
            edit_txt(path= folder, label= label)
            remove_folder(folder_1= folder, folder_2= folder_move_label)
            remove_folder(folder_1= rgb_folder, folder_2= folder_move_image)



if __name__ == "__main__":

    # edit_txt(path= "video1_00031-00005", label="trashDumping")
    clear_data()
    # remove_folder("linhtinh", "video_test")
