#!/bin/bash
#
# ---------- Slurm directives ----------
#SBATCH --export=ALL
#SBATCH --cpus-per-task=48          # adjust if you want a different CPU-to-GPU ratio
#SBATCH --gres=gpu:4                # **4 GPUs on this node**
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # one Slurm task; torchrun handles the GPU procs
#SBATCH --job-name=dino_s_info_drop_subset_4g
#SBATCH --output=slurm_logs/dino_s_info_drop_subset_4g.out
#SBATCH --error=slurm_logs/dino_s_info_drop_subset_4g.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kgabrham@gmail.com
#SBATCH --partition=cscc-gpu-p
#SBATCH -q cscc-gpu-qos
# --------------------------------------

# (Optional) environment setup
# module load cuda/12.4
# conda activate myenv

python train_dino_nowds_lightning.py \
    --arch resnext50_32x4d \
    --patch_size 14 \
    --num_workers 32 \
    --lr 0.0001 \
    --min_lr 0.0001 \
    --optimizer adamw \
    --output_dir . \
    --data_path /l/users/abrham.gebreselasie/datasets/expt_saycam/imfldr_train_5fps/ \
    --save_prefix dino_s_info_drop_subset \
    --batch_size 256 \
    --accelerator gpu \
    --devices 4 \
    --precision 32-true
