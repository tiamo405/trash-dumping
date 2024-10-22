checkpoints=checkpoints/trash/yowo_v2_medium/yowo_v2_medium_epoch_50.pth
video_test=test_model/video_test/2.mp4
# python demo.py --cuda \
#                 -v yowo_v2_medium \
#                 --num_classes 2 \
#                 -size 224 \
#                 --weight $checkpoints \
#                 --video $video_test \
#                 --vis_thresh 0.7 \
#                 -d trash \
#                 # --gif \

                
python run_model_custom.py -v yowo_v2_medium \
                --dataset trash \
                --len_clip 8 \
                --img_size 224 \
                --weight $checkpoints \
                --vis_thresh 0.8 \
                --save_path test_model/outputs/danhgia \
                --video $video_test \

