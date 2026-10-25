#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export MUJOCO_GL=osmesa
lerobot-eval \
  --policy.path=/root/autodl-tmp/dp_baseline_s2/checkpoints/200000/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}' \
  --eval.n_episodes=10 \
  --eval.batch_size=5 \
  --output_dir=/root/autodl-tmp/dp_baseline_s2/eval_200000 \
  2>&1 | tee /root/autodl-tmp/dp_eval_s2.log
