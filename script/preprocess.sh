# docker run --name trashDumping-namtp --runtime nvidia -dit -v /mnt/nvme0n1/phuongnam/Trash-Dumping:/workspace nvcr.io/nvidia/pytorch:21.03-py3

# tach frame tu video 
# python extract_frame/video2frame.py \
#     --dir_data trash \
#     --label video_split \
#     --weight_yolo yolov5n.pt \
#     --dir_save trash



# chia data thanh train test
python trash/build_data_list.py \
    --raw_data trash \

# split video
# python trash/split_video.py \
#     --path_video trash/videos/video1.mp4 \
#     --duration_per_video 20 \
#     --dir_folder trash \
    