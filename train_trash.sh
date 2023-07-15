CUDA_VISIBLE_DEVICES=0 python train.py \
        --dataset trash \
        --version yowo_v2_medium \
        --root . \
        --num_classes 2 \
        --num_workers 2 \
        --eval_epoch 5 \
        --max_epoch 50 \
        --lr_epoch 2 3 4 5 \
        -lr 0.0001 \
        -ldr 0.5 \
        --batch_size 8 \
        -accu 16 \
        -K 16 \
        # --cuda


# CUDA_VISIBLE_DEVICES=0 python train.py \
#         --cuda \
#         -d trash \
#         -v yowo_v2_tiny \
#         --root . \
#         --num_classes 2 \
#         --num_workers 2 \
#         --eval_epoch 5 \
#         --max_epoch 50 \
#         --lr_epoch 2 3 4 5 \
#         -lr 0.0001 \
#         -ldr 0.5 \
#         --batch_size 2 \
#         -accu 16 \
#         -K 4
