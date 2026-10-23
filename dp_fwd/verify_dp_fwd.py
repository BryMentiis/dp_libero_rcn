#!/usr/bin/env python
"""diffusion_fwd 自检（CPU，小尺寸）：
1. 工厂能按名字解析策略类与 processor；
2. 带未来帧的 batch 能算总损失 + 辅助损失明细；
3. drop_forward_heads() 后参数量与原版一致，且同权重同噪声下推理输出逐元素相同。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import dp_fwd  # noqa: F401
from dp_fwd.configuration_diffusion_fwd import DiffusionFwdConfig
from dp_fwd.modeling_diffusion_fwd import DiffusionFwdPolicy

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import get_policy_class
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def make_cfg(cls, **kw):
    return cls(
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (8,)),
            "observation.images.cam0": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (7,))},
        n_obs_steps=2,
        horizon=8,
        n_action_steps=4,
        down_dims=(64, 128, 256),
        spatial_softmax_num_keypoints=8,
        pretrained_backbone_weights=None,
        num_inference_steps=3,
        device="cpu",
        **kw,
    )


def main():
    # 1) 工厂解析
    assert get_policy_class("diffusion_fwd") is DiffusionFwdPolicy
    print("[1] factory resolves diffusion_fwd OK")

    torch.manual_seed(0)
    cfg = make_cfg(DiffusionFwdConfig, k_mid=2, k_q=3, k_high=4)
    policy = DiffusionFwdPolicy(cfg)
    policy.train()

    # 2) 训练前向：s 维 = 过去 2 步 + 未来 {2,3,4}
    b, s = 4, 2 + 3
    batch = {
        OBS_STATE: torch.randn(b, s, 8),
        OBS_IMAGES: torch.rand(b, s, 1, 3, 64, 64),
        ACTION: torch.randn(b, cfg.horizon, 7),
        "action_is_pad": torch.zeros(b, cfg.horizon, dtype=torch.bool),
    }
    loss, aux = policy.forward(batch)
    loss.backward()
    assert aux and {"l_diff", "l_mid", "l_high", "l_q", "l_total"} <= set(aux), aux
    print(f"[2] forward OK: {aux}")

    # 梯度检查：编码器与预测头有梯度；未来目标端无额外回传（no_grad 天然保证）
    enc_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in policy.diffusion.rgb_encoder.parameters()
    )
    head_grad = all(
        p.grad is not None for p in policy.diffusion.p_high.parameters()
    )
    assert enc_grad and head_grad
    print("[3] gradients flow to encoder + heads OK")

    # 4) 推理等价：移除预测头后与原版同权重同噪声输出一致
    base_policy = DiffusionPolicy(make_cfg(DiffusionConfig))
    missing, unexpected = base_policy.diffusion.load_state_dict(
        policy.diffusion.state_dict(), strict=False
    )
    assert not missing and [k for k in unexpected if "p_" in k], (missing, unexpected)

    policy.diffusion.drop_forward_heads()
    assert len(list(policy.diffusion.parameters())) == len(
        list(base_policy.diffusion.parameters())
    )
    policy.eval()
    base_policy.eval()

    obs_batch = {
        OBS_STATE: torch.randn(2, 2, 8),
        OBS_IMAGES: torch.rand(2, 2, 1, 3, 64, 64),
    }
    noise = torch.randn(2, cfg.horizon, 7)
    with torch.no_grad():
        a_new = policy.diffusion.generate_actions(obs_batch, noise=noise)
        a_base = base_policy.diffusion.generate_actions(obs_batch, noise=noise)
    assert torch.equal(a_new, a_base), (a_new - a_base).abs().max()
    print("[4] inference equivalence after drop_forward_heads OK (bitwise equal)")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
