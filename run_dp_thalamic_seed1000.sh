#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONPATH=/root/autodl-tmp/dp_code/dp_thalamic

# 等待基线 dp_baseline_s2 训练进程结束（避免单卡 4090 显存冲突）
echo "waiting for baseline dp_baseline_s2 to finish..."
while pgrep -f "output_dir=/root/autodl-tmp/dp_baseline_s2" > /dev/null; do
  sleep 60
done
sleep 30  # 等待显存释放
echo "baseline finished, starting dp_thalamic training at $(date)"

# 与基线 dp_baseline_s2 完全同配置，仅改 policy.type 与输出目录，显式 seed=1000
# (基线未显式传 --seed，LeRobot 0.6.1 默认 seed=1000，与 dp_baseline_s1 的 train_config.json 一致)
lerobot-train \
  --policy.type=dp_thalamic \
  --policy.push_to_hub=false \
  --policy.use_topdown_gate=true \
  --policy.gate_rank_ratio=16 \
  --policy.gate_min_rank=16 \
  --dataset.repo_id=nvidia/LIBERO_LeRobot_v3 \
  --dataset.root=/root/autodl-tmp/libero_lerobot/libero_spatial \
  --output_dir=/root/autodl-tmp/dp_thalamic_seed1000 \
  --wandb.enable=false \
  --seed=1000 \
  --steps=200000 \
  --save_freq=100000 \
  --batch_size=64 \
  2>&1 | tee /root/autodl-tmp/dp_thalamic_seed1000.log
