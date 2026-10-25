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
"""Residual top-down thalamic gate for 1D U-Net skip features.

Engineering abstraction inspired by the feedback-gating-of-feedforward-evidence
mechanism described in George et al. (2025), "A detailed theory of thalamic and
cortical microcircuits for predictive visual inference", Science Advances 11(6).
This module is a mechanism-level abstraction, not a line-by-line reproduction of
the paper's Noisy-OR belief propagation.
"""

import torch
from torch import Tensor, nn


class TopDownThalamicGate1d(nn.Module):
    """Residual top-down modulation of an encoder skip feature."""

    def __init__(self, channels: int, rank: int):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        self.channels = channels
        self.rank = rank
        # 门控信号通路（新增模块，原 Diffusion U-Net 中不存在）。
        # 类脑机制对应（George et al., 2025, Sci. Adv. 11(6):eadr6698）：高层皮层
        # L6 的反馈投射作用于 TRN 抑制性细胞，调节经丘脑中继的前馈驱动输入；
        # 这里用 bottleneck 1x1 卷积从 top-down context 计算逐通道、逐时间位置的
        # 调制 logits，是该回路的最小工程抽象（非 Noisy-OR 信念传播复现）。
        self.gate_logits = nn.Sequential(
            nn.Conv1d(channels, rank, kernel_size=1, bias=True),
            nn.Mish(),
            nn.Conv1d(rank, channels, kernel_size=1, bias=True),
        )
        # 可学习残差强度，初始为 0：初始化时 s_eff = s 严格恒等，
        # 整个机制版网络与原 DiffusionConditionalUnet1d 前向逐位一致（恒等回退），
        # gamma 先被更新，随后 gate_logits 才参与学习（数学预期，非 bug）。
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, skip: Tensor, context: Tensor) -> Tensor:
        if skip.ndim != 3 or context.ndim != 3:
            raise ValueError(
                f"Expected [B,C,L], got {skip.shape} and {context.shape}"
            )
        if skip.shape != context.shape:
            raise ValueError(
                f"skip/context mismatch: {skip.shape} vs {context.shape}"
            )
        if skip.shape[1] != self.channels:
            raise ValueError(
                f"Expected C={self.channels}, got {skip.shape}"
            )

        # g = 2*sigmoid(u) ∈ (0,2)：以 1 为中心的分级门控（graded gating），
        # g<1 为抑制、g>1 为增强——对应论文 "suppression, or graded gating"；
        # 各位置独立 sigmoid，无 softmax 归一化，不是 Q-K-V attention。
        gate = 2.0 * torch.sigmoid(self.gate_logits(context))
        # 残差形式：原始前馈证据 skip 始终完整保留（直接前馈通路/丘脑中继不被切断），
        # 反馈只产生乘性修正项 gamma*(g-1)*skip；context 不 detach，梯度可回传。
        return skip + self.gamma * (gate - 1.0) * skip
