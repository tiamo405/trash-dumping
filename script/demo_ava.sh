# python test_video_ava.py \
#     --cuda \
#     -d ava_v2.2 \
#     -v yowo_v2_nano \
#     --weight /mnt/nvme0n1/phuongnam/Trash-Dumping/checkpoints/ava/yowo_v2_nano_ava.pth \
#     --video test_model/video_test/walking_in_city.mp4

# run demo
CUDA_VISIBLE_DEVICES=1 python test_video_ava.py --cuda -d ava_v2.2 -v yowo_v2_nano -size 224 --weight /mnt/nvme0n1/phuongnam/Trash-Dumping/checkpoints/ava/yowo_v2_nano_ava.pth --video test_model/video_test/walking_in_city.mp4