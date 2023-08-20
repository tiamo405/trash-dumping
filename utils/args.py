import os
import argparse
import json

def save_args(args, path_save):
    args_dict = vars(args)

    # Lưu thông tin tham số vào tệp JSON
    with open(os.path.join(path_save,'args.json'), 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

def read_args(path):
    with open(path, 'r') as json_file:
        loaded_args = json.load(json_file)

    # Tạo đối tượng argparse.Namespace từ thông tin đã đọc
    namespace = argparse.Namespace(**loaded_args)
    return namespace