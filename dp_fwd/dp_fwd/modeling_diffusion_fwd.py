"""diffusion_fwd: 类脑前向模型组增强的 Diffusion Policy（规格书 §1 主机制）。

实现策略：全部通过继承完成，lerobot 源文件零改动。
- DiffusionFwdModel(DiffusionModel)：在 ResNet layer3 / layer4 上挂 forward hook
  捕获 z_mid / z_high（层级绑定见规格书 §1.1），外挂三个独立 MLP 预测头；
  compute_loss = 原版扩散损失 + λ·前向预测损失 + λq·本体预测损失。
- DiffusionFwdPolicy(DiffusionPolicy)：仅替换 self.diffusion 为上面的子类。

推理等价性（规格书 §0-1 / §3-Step2）：
- hook 仅在 compute_loss 内打开（self._capture），推理路径逐算子不变；
- 预测头只被 compute_loss 调用；drop_forward_heads() 可整组移除以验证
  参数量 / 输出与基线一致。

未来帧数据：batch 的观测 s 维 = [过去 n_obs 步] + [sorted(k_mid, k_q, k_high)]，
由 DiffusionFwdConfig.observation_delta_indices 驱动 dataloader 提供。
"""

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel, DiffusionPolicy
from lerobot.policies.utils import get_output_shape
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from .configuration_diffusion_fwd import DiffusionFwdConfig


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class DiffusionFwdModel(DiffusionModel):
    """原版 DiffusionModel + 前向模型组（P_high / P_mid / P_q，独立参数不共享）。"""

    def __init__(self, config: DiffusionFwdConfig):
        super().__init__(config)
        self.config: DiffusionFwdConfig = config

        # ---- z_mid / z_high 捕获：backbone = Sequential(conv1..layer4)，
        # backbone[-2] = layer3（中层），backbone[-1] = layer4（高层）----
        self._capture = False
        self._z_cache: dict[str, list[Tensor]] = {"mid": [], "high": []}
        self._hook_handles = []
        for enc in self._encoders():
            self._hook_handles.append(
                enc.backbone[-2].register_forward_hook(self._make_hook("mid"))
            )
            self._hook_handles.append(
                enc.backbone[-1].register_forward_hook(self._make_hook("high"))
            )

        # ---- 特征维度（GAP 后的通道数，多相机时按相机 concat）----
        images_shape = next(iter(config.image_features.values())).shape
        if config.crop_shape is not None:
            hw = config.crop_shape
        elif config.resize_shape is not None:
            hw = config.resize_shape
        else:
            hw = images_shape[1:]
        dummy = (1, images_shape[0], *hw)
        enc0 = self._encoders()[0]
        c_mid = get_output_shape(enc0.backbone[:-1], dummy)[1]   # layer3 输出通道
        c_high = get_output_shape(enc0.backbone, dummy)[1]       # layer4 输出通道
        n_cam = len(config.image_features)
        self.z_mid_dim = c_mid * n_cam
        self.z_high_dim = c_high * n_cam

        # ---- 三个预测头（规格书 §1.1：独立参数、2–3 层 MLP、动作 concat 注入）----
        action_dim = config.action_feature.shape[0]
        state_dim = config.robot_state_feature.shape[0]
        h = config.fwd_hidden_dim
        self.p_mid = _mlp(self.z_mid_dim + config.k_mid * action_dim, h, self.z_mid_dim)
        self.p_high = _mlp(self.z_high_dim + config.k_high * action_dim, h, self.z_high_dim)
        self.p_q = _mlp(
            self.z_high_dim + state_dim + config.k_q * action_dim, h, state_dim
        )

        self._last_aux: dict[str, float] = {}

    # ---------- 内部工具 ----------

    def _encoders(self) -> list[nn.Module]:
        if isinstance(self.rgb_encoder, nn.ModuleList):
            return list(self.rgb_encoder)
        return [self.rgb_encoder]

    def _make_hook(self, key: str):
        def hook(module, inputs, output):
            if self._capture:
                self._z_cache[key].append(output)
        return hook

    def _encode_and_pool_z(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """编码一组观测帧并返回 GAP 后的 (z_mid, z_high)，形状 (b, s, n_cam*C)。"""
        cfg = self.config
        b, s, n_cam = images.shape[:3]
        self._z_cache = {"mid": [], "high": []}
        self._capture = True
        if cfg.use_separate_rgb_encoder_per_camera:
            per_cam = einops.rearrange(images, "b s n ... -> n (b s) ...")
            for enc, imgs in zip(self.rgb_encoder, per_cam, strict=True):
                enc(imgs)
        else:
            self.rgb_encoder(einops.rearrange(images, "b s n ... -> (b s n) ..."))
        self._capture = False

        mid = torch.stack([f.mean(dim=(-2, -1)) for f in self._z_cache["mid"]], dim=1)
        high = torch.stack([f.mean(dim=(-2, -1)) for f in self._z_cache["high"]], dim=1)
        if not cfg.use_separate_rgb_encoder_per_camera:
            # 共享编码器：单条目形状 (b*s*n, C)，还原相机维
            mid = mid.reshape(b * s, n_cam, -1)
            high = high.reshape(b * s, n_cam, -1)
        # (b*s, n_cam, C) -> (b, s, n_cam*C)
        mid = mid.reshape(b, s, -1)
        high = high.reshape(b, s, -1)
        return mid, high

    def _current_only(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """切出原版 DP 需要的过去 n_obs 步，去掉前向模型专用的未来帧。"""
        n_obs = self.config.n_obs_steps
        cur = dict(batch)
        cur[OBS_STATE] = batch[OBS_STATE][:, :n_obs]
        if OBS_IMAGES in batch:
            cur[OBS_IMAGES] = batch[OBS_IMAGES][:, :n_obs]
        if OBS_ENV_STATE in batch:
            cur[OBS_ENV_STATE] = batch[OBS_ENV_STATE][:, :n_obs]
        return cur

    # ---------- 训练 ----------

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        cfg = self.config
        if not cfg.enable_forward_model:
            return super().compute_loss(self._current_only(batch))

        n_obs = cfg.n_obs_steps
        future_ks = sorted({cfg.k_mid, cfg.k_q, cfg.k_high})
        fut_pos = {k: n_obs + i for i, k in enumerate(future_ks)}

        # 1) 原版扩散损失（同时经 hook 捕获当前窗口的 z_mid / z_high）
        self._z_cache = {"mid": [], "high": []}
        self._capture = True
        loss_diff = super().compute_loss(self._current_only(batch))
        self._capture = False
        mid_all, high_all = self._z_cache["mid"], self._z_cache["high"]
        b = batch[OBS_STATE].shape[0]
        n_cam = len(cfg.image_features)
        # 每相机条目形状 (b*n_obs, C, h, w) -> GAP -> (b, n_obs*n_cam, C)，
        # 取当前帧 t（最后一个观测步）对应的所有相机特征
        z_mid_t = torch.stack(
            [f.mean(dim=(-2, -1)) for f in mid_all], dim=1
        ).reshape(b, n_obs * n_cam, -1)[:, -n_cam:].reshape(b, -1)
        z_high_t = torch.stack(
            [f.mean(dim=(-2, -1)) for f in high_all], dim=1
        ).reshape(b, n_obs * n_cam, -1)[:, -n_cam:].reshape(b, -1)

        # 2) 未来真值编码（sg：no_grad 不回传梯度，规格书 §1.1）
        with torch.no_grad():
            fut_z = {}
            for k in (cfg.k_mid, cfg.k_high):
                imgs_k = batch[OBS_IMAGES][:, fut_pos[k] : fut_pos[k] + 1]
                z_mid_k, z_high_k = self._encode_and_pool_z(imgs_k)
                fut_z[k] = (z_mid_k[:, 0], z_high_k[:, 0])
            q_future = batch[OBS_STATE][:, fut_pos[cfg.k_q]]

        # 3) 动作块 a(t:t+k)：ACTION 的 s 维从 t-(n_obs-1) 开始，t 位于 n_obs-1
        def action_block(k: int) -> Tensor:
            return batch[ACTION][:, n_obs - 1 : n_obs - 1 + k].flatten(start_dim=1)

        q_t = batch[OBS_STATE][:, n_obs - 1]

        # 4) 辅助损失（规格书 §1.2）
        l_mid = F.mse_loss(self.p_mid(torch.cat([z_mid_t, action_block(cfg.k_mid)], dim=-1)), fut_z[cfg.k_mid][0])
        l_high = F.mse_loss(self.p_high(torch.cat([z_high_t, action_block(cfg.k_high)], dim=-1)), fut_z[cfg.k_high][1])
        l_q = F.mse_loss(
            self.p_q(torch.cat([z_high_t, q_t, action_block(cfg.k_q)], dim=-1)), q_future
        )

        total = loss_diff + cfg.lambda_fwd * (l_mid + l_high) + cfg.lambda_q * l_q
        self._last_aux = {
            "l_diff": loss_diff.detach().item(),
            "l_mid": l_mid.detach().item(),
            "l_high": l_high.detach().item(),
            "l_q": l_q.detach().item(),
            "l_total": total.detach().item(),
        }
        return total

    # ---------- 推理等价验证（规格书 §3-Step2）----------

    def drop_forward_heads(self):
        """整组移除前向模型组：删预测头、摘除 hook，此后与基线逐算子一致。"""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
        del self.p_mid, self.p_high, self.p_q
        self.config.enable_forward_model = False


class DiffusionFwdPolicy(DiffusionPolicy):
    """工厂按命名约定解析：DiffusionFwdConfig -> DiffusionFwdPolicy。"""

    config_class = DiffusionFwdConfig
    name = "diffusion_fwd"

    def __init__(self, config: DiffusionFwdConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.diffusion = DiffusionFwdModel(config)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        # 与原版相同的相机键堆叠；s 维允许含未来帧（由 compute_loss 内部切分）
        if self.config.image_features:
            batch = dict(batch)
            for key in self.config.image_features:
                if batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        loss = self.diffusion.compute_loss(batch)
        return loss, (self.diffusion._last_aux or None)
