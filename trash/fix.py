import os
import argparse

def add0tofile(pathfolder):
    for video in os.listdir(pathfolder) :
        for fname in os.listdir(os.path.join(pathfolder, video)) :
            old_path = os.path.join(pathfolder, video, fname)
            new_path = '/'.join(old_path.split('/')[:-1]) + '/0' + old_path.split('/')[-1]
            os.rename(old_path, new_path)

def rename(path) :
    for i , fname in enumerate(sorted(os.listdir(path))) :
        new_name ='Walking_'+ str(i).zfill(4)+'.'+ fname.split('.')[-1]
        os.rename(os.path.join(path, fname), os.path.join(path, new_name ))

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument('--raw_path', type=str, default='./trash')
    parser.add_argument('--out_list_path', type=str, default='./')
    parser.add_argument('--shuffle', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # build_split_data(args.raw_path)
    # rename("trash/videos/Walking")
    add0tofile('trash/rgb-images/Walking')