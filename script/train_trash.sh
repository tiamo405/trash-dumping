# python train.py \
#         --dataset trash \
#         --version yowo_v2_medium \
#         --root . \
#         --num_workers 2 \
#         --eval_epoch 5 \
#         --max_epoch 50 \
#         --lr_epoch 2 3 4 5 \
#         -lr 0.0001 \
#         -ldr 0.5 \
#         --batch_size 4 \
#         -accu 16 \
#         --len_clip 8 \
#         --cuda


python train.py \
        -d trash \
        -v yowo_v2_tiny \
        --root . \
        --num_workers 2 \
        --eval_epoch 5 \
        --max_epoch 50 \
        --lr_epoch 2 3 4 5 \
        -lr 0.0001 \
        -ldr 0.5 \
        --batch_size 2 \
        -accu 16 \
        -K 4
