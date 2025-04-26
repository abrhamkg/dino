#! /bin/bash

#SBATCH --export=ALL
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=dino_s_info_drop_subset
#SBATCH --output=slurm_logs/dino_s_info_drop_subset.out
#SBATCH --error=slurm_logs/dino_s_info_drop_subset.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kgabrham@gmail.com
#SBATCH --partition=cscc-gpu-p
#SBATCH -q cscc-gpu-qos

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