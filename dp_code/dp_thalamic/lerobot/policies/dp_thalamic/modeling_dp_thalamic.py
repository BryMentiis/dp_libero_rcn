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
"""Diffusion Policy with top-down thalamic gating on U-Net decoder skip features.

Only the U-Net decoder's routing of encoder skip features changes: before each
concat of an encoder skip feature, the current coarser-scale decoder feature
modulates that skip through a residual soft gate with zero-initialized gamma.
Everything else (diffusion objective, vision encoder, noise scheduler, optimizer
hooks, action queues) is inherited unchanged from the original diffusion policy.
"""

import einops
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from lerobot.utils.import_utils import require_package

from ..diffusion.modeling_diffusion import (
    DiffusionConditionalResidualBlock1d,
    DiffusionConv1dBlock,
    DiffusionModel,
    DiffusionPolicy,
    DiffusionRgbEncoder,
    DiffusionSinusoidalPosEmb,
    _make_noise_scheduler,
)
from ..pretrained import PreTrainedPolicy
from .configuration_dp_thalamic import DpThalamicConfig
from .topdown_thalamic_gate import TopDownThalamicGate1d


class DpThalamicPolicy(DiffusionPolicy):
    """Diffusion Policy whose U-Net decoder gates encoder skip features top-down."""

    # 与原 DiffusionPolicy 的区别仅在于 config_class 与 name；
    # __init__ 逻辑与源码 DiffusionPolicy.__init__ 完全相同，
    # 唯一行为差异来自 self.diffusion 被换成 DpThalamicModel（见下）。
    config_class = DpThalamicConfig
    name = "dp_thalamic"

    def __init__(
        self,
        config: DpThalamicConfig,
        **kwargs,
    ):
        require_package("diffusers", extra="diffusion")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.diffusion = DpThalamicModel(config)

        self.reset()


class DpThalamicModel(DiffusionModel):
    """DiffusionModel with a DpThalamicConditionalUnet1d. All methods are inherited."""

    def __init__(self, config: DpThalamicConfig):
        nn.Module.__init__(self)
        self.config = config

        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        # 与源码 DiffusionModel 的唯一替换点：U-Net 由 DiffusionConditionalUnet1d
        # 换成带丘脑式自上而下门控的 DpThalamicConditionalUnet1d；
        # 视觉编码器、噪声调度器、条件生成等其余部分与源码逐一相同。
        self.unet = DpThalamicConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)

        if config.compile_model:
            # Compile the U-Net. "reduce-overhead" is preferred for the small-batch repetitive loops
            # common in diffusion inference.
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps


class DpThalamicConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Identical to DiffusionConditionalUnet1d except that, when
    `config.use_topdown_gate` is True, each encoder skip feature is modulated by a
    TopDownThalamicGate1d (conditioned on the current coarser decoder feature)
    right before the decoder concat.

    Note: this removes local conditioning as compared to the original diffusion policy code.
    """

    def __init__(self, config: DpThalamicConfig, global_cond_dim: int):
        super().__init__()

        self.config = config

        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we
        # just reverse these.
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # Unet encoder.
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # 【新增模块，原源码无】每个 decoder block 一个 TopDownThalamicGate1d。
        # 类脑机制对应：门控块位于层级接口（论文："a different block that sits at
        # the interface between two levels"）；以当前较粗时间尺度的 decoder feature
        # 作为 top-down 反馈（高层皮层 L6 → TRN），调制即将向上传递的 encoder skip
        # （低层前馈证据，相当于经丘脑中继的 driving feedforward input）。
        # channels 必须取 dim_in（该层 skip/context 的通道数），rank 为 bottleneck 维度。
        self.topdown_gates = nn.ModuleList([])
        if config.use_topdown_gate:
            for dim_out, dim_in in reversed(in_out[1:]):
                channels = dim_in
                rank = max(config.gate_min_rank, channels // config.gate_rank_ratio)
                self.topdown_gates.append(
                    TopDownThalamicGate1d(channels=channels, rank=rank)
                )

        if config.use_topdown_gate:
            assert len(self.topdown_gates) == len(self.up_modules)
        else:
            assert len(self.topdown_gates) == 0

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        use_gc = self.config.gradient_checkpointing and self.training

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            if use_gc:
                x = checkpoint(resnet, x, global_feature, use_reentrant=False)
                x = checkpoint(resnet2, x, global_feature, use_reentrant=False)
            else:
                x = resnet(x, global_feature)
                x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            if use_gc:
                x = checkpoint(mid_module, x, global_feature, use_reentrant=False)
            else:
                x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            # 【与源码的核心区别】源码直接 x = torch.cat((x, encoder_skip_features.pop()), dim=1)；
            # 这里在拼接之前，用 decoder 当前（较粗时间尺度）特征 x 作为 top-down context，
            # 对 encoder skip 做残差软门控。类脑机制：反馈消息调节前馈消息的上传强度
            # （explaining away / gating / top-down attention 的工程抽象）；
            # gamma=0 时 s_eff = s，严格退化为源码行为。gate 很小，保持直接执行、
            # 不包 checkpoint；原 residual blocks 的 gradient checkpointing 不变。
            skip = encoder_skip_features.pop()
            if self.config.use_topdown_gate:
                skip = self.topdown_gates[idx](skip=skip, context=x)

            x = torch.cat((x, skip), dim=1)
            if use_gc:
                x = checkpoint(resnet, x, global_feature, use_reentrant=False)
                x = checkpoint(resnet2, x, global_feature, use_reentrant=False)
            else:
                x = resnet(x, global_feature)
                x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")
        return x
