# python demo.py --cuda \
#                 -v yowo_v2_medium \
#                 --num_classes 2 \
#                 -size 224 \
#                 --weight checkpoints/trash/yowo_v2_medium/yowo_v2_medium_epoch_50.pth \
#                 --video trash/videos/trashDumping/VID_20230310_161738.avi \
#                 --vis_thresh 0.7 \
#                 -d trash \
#                 # --gif \

                
CUDA_VISIBLE_DEVICES=0 python demo.py -v yowo_v2_medium \
                --dataset trash \
                --len_clip 16 \
                -size 224 \
                --weight checkpoints/trash/yowo_v2_tiny/2023-08-20/yowo_v2_tiny_epoch_6.pth \
                --vis_thresh 0.9 \
                --dataset trash \
                --save_path outputs \
                --video video_test/walking_test.mp4 \
                # -video trash/videos/Walking/Walking_0014.mp4 \--gif \                

