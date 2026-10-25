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
from typing import Any

import torch

from lerobot.lerobot_types import PolicyAction
from lerobot.policies.diffusion.processor_diffusion import (
    make_diffusion_pre_post_processors,
)
from lerobot.processor import PolicyProcessorPipeline

from .configuration_dp_thalamic import DpThalamicConfig


def make_dp_thalamic_pre_post_processors(
    config: DpThalamicConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Thin adapter reusing the diffusion pre/post processors unchanged."""
    # 与原 diffusion processor 无行为差异：归一化、图像变换、动作后处理全部复用，
    # 本函数仅为满足 factory 按命名约定解析 `make_{type}_pre_post_processors` 而存在。
    return make_diffusion_pre_post_processors(config=config, dataset_stats=dataset_stats)
