# docker run --name namtp_test --runtime nvidia -dit -v /mnt/nvme0n1/phuongnam/Trash-Dumping:/workspace nvcr.io/nvidia/pytorch:20.03-py3

# tach frame tu video 
# python extract_frame/video2frame.py \
#     --dir_data trash \
#     --label Walking \
#     --weight_yolo yolov5n.pt

# chia data thanh train test
python trash/build_split.py \
    --raw_path ./trash \