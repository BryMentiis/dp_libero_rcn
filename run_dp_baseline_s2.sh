#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
lerobot-train \
  --policy.type=diffusion \
  --policy.push_to_hub=false \
  --dataset.repo_id=nvidia/LIBERO_LeRobot_v3 \
  --dataset.root=/root/autodl-tmp/libero_lerobot/libero_spatial \
  --output_dir=/root/autodl-tmp/dp_baseline_s2 \
  --wandb.enable=false \
  --steps=200000 \
  --save_freq=100000 \
  --batch_size=64 \
  2>&1 | tee /root/autodl-tmp/dp_baseline_s2.log
