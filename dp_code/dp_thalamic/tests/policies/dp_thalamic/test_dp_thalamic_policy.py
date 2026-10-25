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
import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionModel,
)
from lerobot.policies.dp_thalamic.configuration_dp_thalamic import DpThalamicConfig
from lerobot.policies.dp_thalamic.modeling_dp_thalamic import (
    DpThalamicConditionalUnet1d,
    DpThalamicModel,
    DpThalamicPolicy,
)
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def make_config(cls=DpThalamicConfig, **overrides):
    kwargs = dict(
        n_obs_steps=2,
        horizon=8,
        n_action_steps=4,
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (4,)),
            "observation.image": PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        },
        output_features={
            ACTION: PolicyFeature(FeatureType.ACTION, (3,)),
        },
        down_dims=(64, 128, 256),
        spatial_softmax_num_keypoints=8,
        pretrained_backbone_weights=None,
        num_train_timesteps=10,
    )
    kwargs.update(overrides)
    return cls(**kwargs)


def make_batch(config, batch_size=2):
    return {
        OBS_STATE: torch.randn(batch_size, config.n_obs_steps, 4),
        "observation.image": torch.rand(batch_size, config.n_obs_steps, 3, 96, 96),
        ACTION: torch.randn(batch_size, config.horizon, 3),
        "action_is_pad": torch.zeros(batch_size, config.horizon, dtype=torch.bool),
    }


def test_registration():
    assert "dp_thalamic" in PreTrainedConfig.get_known_choices()
    assert get_policy_class("dp_thalamic") is DpThalamicPolicy
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    assert get_policy_class("diffusion") is DiffusionPolicy
    cfg = make_policy_config("dp_thalamic")
    assert isinstance(cfg, DpThalamicConfig)
    assert cfg.use_topdown_gate is True
    assert cfg.gate_rank_ratio == 16
    assert cfg.gate_min_rank == 16


def test_config_validation():
    with pytest.raises(ValueError):
        make_config(gate_rank_ratio=0)
    with pytest.raises(ValueError):
        make_config(gate_min_rank=0)


@pytest.mark.parametrize("use_topdown_gate", [True, False])
@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_policy_forward_backward(use_topdown_gate, gradient_checkpointing):
    torch.manual_seed(0)
    config = make_config(
        use_topdown_gate=use_topdown_gate, gradient_checkpointing=gradient_checkpointing
    )
    policy = DpThalamicPolicy(config)
    policy.train()
    loss, _ = policy(make_batch(config))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)

    unet = policy.diffusion.unet
    expected_gates = len(unet.up_modules) if use_topdown_gate else 0
    assert len(unet.topdown_gates) == expected_gates


def test_gate_channels_match_decoder_levels():
    config = make_config()
    unet = DpThalamicModel(config).unet
    # in_out = [(3, 64), (64, 128), (128, 256)]; decoder levels use skip/context channels 256, 128.
    assert [g.channels for g in unet.topdown_gates] == [256, 128]
    assert [g.rank for g in unet.topdown_gates] == [16, 16]


@pytest.mark.parametrize("use_topdown_gate", [True, False])
def test_zero_gamma_unet_equivalence(use_topdown_gate):
    """With gamma=0 (default) the gated U-Net must match the original bitwise."""
    torch.manual_seed(0)
    cfg_base = make_config(DiffusionConfig)
    cfg_gate = make_config(use_topdown_gate=use_topdown_gate)
    global_cond_dim = 17

    unet_base = DiffusionConditionalUnet1d(cfg_base, global_cond_dim=global_cond_dim)
    unet_gate = DpThalamicConditionalUnet1d(cfg_gate, global_cond_dim=global_cond_dim)

    missing, unexpected = unet_gate.load_state_dict(unet_base.state_dict(), strict=False)
    assert unexpected == []
    assert all(k.startswith("topdown_gates.") for k in missing)
    assert (len(missing) > 0) == use_topdown_gate

    unet_base.eval()
    unet_gate.eval()
    x = torch.randn(2, cfg_base.horizon, 3)
    timesteps = torch.randint(0, cfg_base.num_train_timesteps, (2,))
    global_cond = torch.randn(2, global_cond_dim)
    with torch.no_grad():
        out_base = unet_base(x, timesteps, global_cond=global_cond)
        out_gate = unet_gate(x, timesteps, global_cond=global_cond)
    torch.testing.assert_close(out_base, out_gate, rtol=0, atol=0)


def test_extra_param_count():
    cfg_base = make_config(DiffusionConfig)
    cfg_gate = make_config()
    n_base = sum(p.numel() for p in DiffusionModel(cfg_base).parameters())
    n_gate = sum(p.numel() for p in DpThalamicModel(cfg_gate).parameters())
    expected = 0
    for channels in (256, 128):
        rank = max(16, channels // 16)
        expected += channels * rank + rank  # conv1 weight + bias
        expected += rank * channels + channels  # conv2 weight + bias
        expected += 1  # gamma
    assert n_gate - n_base == expected


def test_default_arch_param_count():
    """Spec estimate: ~655,360 new weights (+ biases + 2 gammas) at down_dims=(512,1024,2048)."""
    config = make_config(down_dims=(512, 1024, 2048), horizon=16)
    unet = DpThalamicModel(config).unet
    gate_params = sum(p.numel() for g in unet.topdown_gates for p in g.parameters())
    weights = 2048 * 128 + 128 * 2048 + 1024 * 64 + 64 * 1024
    biases = (128 + 2048) + (64 + 1024)
    assert gate_params == weights + biases + 2


def test_save_load_pretrained(tmp_path):
    torch.manual_seed(0)
    config = make_config()
    policy = DpThalamicPolicy(config)
    policy.save_pretrained(tmp_path)

    loaded = DpThalamicPolicy.from_pretrained(tmp_path).to("cpu")
    assert isinstance(loaded.config, DpThalamicConfig)
    assert loaded.config.type == "dp_thalamic"
    assert loaded.config.use_topdown_gate is True
    assert loaded.config.gate_rank_ratio == 16
    assert loaded.config.gate_min_rank == 16

    gate_keys = [k for k in loaded.state_dict() if "topdown_gates" in k]
    assert len(gate_keys) > 0
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_pre_post_processors_match_diffusion():
    from lerobot.policies.diffusion.processor_diffusion import (
        make_diffusion_pre_post_processors,
    )
    from lerobot.policies.dp_thalamic.processor_dp_thalamic import (
        make_dp_thalamic_pre_post_processors,
    )

    stats = {
        OBS_STATE: {"min": torch.zeros(4), "max": torch.ones(4)},
        "observation.image": {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
        ACTION: {"min": torch.zeros(3), "max": torch.ones(3)},
    }
    cfg_base = make_config(DiffusionConfig, device="cpu")
    cfg_gate = make_config(device="cpu")

    pre_b, post_b = make_diffusion_pre_post_processors(cfg_base, stats)
    pre_g, post_g = make_dp_thalamic_pre_post_processors(cfg_gate, stats)

    assert [type(s) for s in pre_g.steps] == [type(s) for s in pre_b.steps]
    assert [type(s) for s in post_g.steps] == [type(s) for s in post_b.steps]

    # Factory resolution must find the dp_thalamic processors by naming convention.
    pre_f, post_f = make_pre_post_processors(cfg_gate, dataset_stats=stats)
    assert [type(s) for s in pre_f.steps] == [type(s) for s in pre_g.steps]

    batch = {
        OBS_STATE: torch.rand(4),
        "observation.image": torch.rand(3, 96, 96),
    }
    out_b = pre_b(dict(batch))
    out_g = pre_g(dict(batch))
    assert out_b.keys() == out_g.keys()
    for key in out_b:
        if isinstance(out_b[key], torch.Tensor):
            torch.testing.assert_close(out_b[key], out_g[key])
