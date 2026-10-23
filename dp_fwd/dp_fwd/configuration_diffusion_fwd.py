"""Config for diffusion_fwd: 类脑前向模型组增强的 Diffusion Policy。

继承 lerobot 原版 DiffusionConfig，新增内容全部是"追加字段 + 一个属性覆盖"：
- 不改动任何 lerobot 源文件；
- 通过 register_subclass("diffusion_fwd") 注册为新 policy type，
  lerobot 工厂按命名约定从 dp_fwd.modeling_diffusion_fwd / dp_fwd.processor_diffusion_fwd 解析。

未来帧数据通路：observation_delta_indices 覆盖为 [过去 n_obs 步] + [k_mid, k_q, k_high]，
lerobot.datasets.factory.resolve_delta_timestamps 会据此让 dataloader 额外送出
z(t+k) 所需的未来图像 / 本体状态。推理（rollout）时 batch 只含过去帧，不受影响。
"""

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("diffusion_fwd")
@dataclass
class DiffusionFwdConfig(DiffusionConfig):
    """Diffusion Policy + 前向模型组（仅训练期挂载，见规格书 §1）。"""

    # ---- 辅助损失权重（规格书 §1.2，初值 λ=1.0、λq=1.0）----
    lambda_fwd: float = 1.0
    lambda_q: float = 1.0

    # ---- 预测头 horizon 绑定（规格书 §1.1）----
    k_high: int = 8  # P_high: z_high(t) + a(t:t+8) -> z_high(t+8)
    k_mid: int = 2   # P_mid:  z_mid(t)  + a(t:t+2) -> z_mid(t+2)
    k_q: int = 6     # P_q:    z_high(t) + q_t + a(t:t+6) -> q(t+6)

    # ---- 预测头结构 ----
    fwd_hidden_dim: int = 512  # 各预测头 MLP 隐层宽度
    enable_forward_model: bool = True  # False 时退化为纯基线（用于对照调试）

    @property
    def observation_delta_indices(self) -> list:
        # 过去观测（与原版一致）+ 未来帧（仅供前向模型组取 sg 目标）
        past = list(range(1 - self.n_obs_steps, 1))
        future = sorted({self.k_mid, self.k_q, self.k_high})
        return past + future
