python -u train_dino_nowds.py   --arch "resnext50_32x4d" \
        --patch_size 14 \
        --num_workers 8 \
        --lr 0.0001 \
        --min_lr 0.0001 \
        --optimizer adamw\
        --output_dir . \
        --data_path /l/users/abrham.gebreselasie/datasets/SAYCam/imfldr_train_5fps/  \
        --save_prefix "dino_s_info_drop_subset" \
        --batch_size 32