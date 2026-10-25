#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("dp_thalamic")
@dataclass
class DpThalamicConfig(DiffusionConfig):
    """Diffusion Policy config with optional top-down thalamic gating on U-Net skip features.

    Inherits every field of DiffusionConfig unchanged. The only additions are the gate
    mechanism switches below.
    """

    use_topdown_gate: bool = True  # False 时网络结构与原 DiffusionPolicy 完全一致（诊断臂 G2）
    # gate 容量参数：bottleneck 维度 rank = max(gate_min_rank, channels // gate_rank_ratio)
    # 默认下 C=2048→rank=128，C=1024→rank=64，新增参数约 65.9 万（占全网 ~0.24%）
    gate_rank_ratio: int = 16
    gate_min_rank: int = 16

    def __post_init__(self):
        super().__post_init__()
        if self.gate_rank_ratio <= 0:
            raise ValueError("gate_rank_ratio must be > 0")
        if self.gate_min_rank <= 0:
            raise ValueError("gate_min_rank must be > 0")
