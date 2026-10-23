#!/usr/bin/env python
"""diffusion_fwd 训练入口：等价于 lerobot-train，只是在解析 CLI 前注册新策略。

用法与 lerobot-train 完全一致，例如：
    python train_dp_fwd.py \
        --policy.type=diffusion_fwd \
        --policy.push_to_hub=false \
        --dataset.repo_id=nvidia/LIBERO_LeRobot_v3 \
        --dataset.root=/root/autodl-tmp/libero_lerobot/libero_spatial \
        --output_dir=/root/autodl-tmp/dp_fwd_s1 \
        --wandb.enable=false --steps=30000 --save_freq=5000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dp_fwd  # noqa: F401  # 注册 diffusion_fwd 到 lerobot 的策略注册表

from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
