# python test_video_ava.py \
#     --cuda \
#     -d ava_v2.2 \
#     -v yowo_v2_nano \
#     --weight /mnt/nvme0n1/phuongnam/Trash-Dumping/checkpoints/ava/yowo_v2_nano_ava.pth \
#     --video test_model/video_test/walking_in_city.mp4

# run demo
CUDA_VISIBLE_DEVICES=1 python test_video_ava.py --cuda -d ava_v2.2 -v yowo_v2_nano -size 224 --weight checkpoints/ava/yowo_v2_nano_ava.pth -K 16 --video trash/New_video/VID_20230310_161738.mp4

python test_video_ava.py --cuda -d ava_v2.2 -v yowo_v2_nano -size 224 --weight checkpoints/ava/yowo_v2_nano_ava_k32.pth -K 32 --video trash/New1703/VID_20230317_160459.mp4

python test_video_ava.py -d ava_v2.2 -v yowo_v2_large -size 224 --weight checkpoints/ava/yowo_v2_large_ava_k32.pth -K 32 --video test_model/video_test/2.mp4

python test_video_ava.py -d ava_v2.2 -v yowo_v2_large -size 224 -K 32 