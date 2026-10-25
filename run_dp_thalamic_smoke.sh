#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONPATH=/root/autodl-tmp/dp_code/dp_thalamic

# Smoke 1: dp_thalamic minimal training (20 steps, one save at the end)
lerobot-train \
  --policy.type=dp_thalamic \
  --policy.push_to_hub=false \
  --dataset.repo_id=nvidia/LIBERO_LeRobot_v3 \
  --dataset.root=/root/autodl-tmp/libero_lerobot/libero_spatial \
  --output_dir=/root/autodl-tmp/dp_thalamic_smoke \
  --wandb.enable=false \
  --seed=1000 \
  --steps=20 \
  --save_freq=20 \
  2>&1 | tee /root/autodl-tmp/dp_thalamic_smoke.log

SMOKE1=$?
echo "dp_thalamic smoke exit code: $SMOKE1"
[ $SMOKE1 -ne 0 ] && exit $SMOKE1

# Smoke 2: original diffusion policy must still run from the same workspace (2 steps, no save)
lerobot-train \
  --policy.type=diffusion \
  --policy.push_to_hub=false \
  --dataset.repo_id=nvidia/LIBERO_LeRobot_v3 \
  --dataset.root=/root/autodl-tmp/libero_lerobot/libero_spatial \
  --output_dir=/root/autodl-tmp/dp_diffusion_smoke_ws \
  --wandb.enable=false \
  --seed=1000 \
  --steps=2 \
  --save_freq=1000000 \
  2>&1 | tee /root/autodl-tmp/dp_diffusion_smoke_ws.log

echo "diffusion smoke exit code: $?"
