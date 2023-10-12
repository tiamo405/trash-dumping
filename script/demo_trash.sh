checkpoints=checkpoints/trash/yowo_v2_medium/2023-10-01/yowo_v2_medium_epoch_50.pth
video_test=test_model/video_test/xarac3.mp4
# python demo.py --cuda \
#                 -v yowo_v2_medium \
#                 --num_classes 2 \
#                 -size 224 \
#                 --weight $checkpoints \
#                 --video $video_test \
#                 --vis_thresh 0.7 \
#                 -d trash \
#                 # --gif \

                
CUDA_VISIBLE_DEVICES=0 python demo.py -v yowo_v2_medium \
                --cuda \
                --dataset trash \
                --len_clip 8 \
                --img_size 224 \
                --weight $checkpoints \
                --vis_thresh 0.8 \
                --save_path test_model/outputs \
                --video $video_test \

