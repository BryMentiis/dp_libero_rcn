"""diffusion_fwd 的 pre/post processor：完全复用原版 diffusion 的归一化管线。

lerobot 工厂按命名约定寻找 make_diffusion_fwd_pre_post_processors，
这里直接委托给 lerobot.policies.diffusion.processor_diffusion，
保证训练/推理的数据预处理与基线逐算子一致（规格书 §0-2 变量隔离）。
"""

from lerobot.policies.diffusion.processor_diffusion import (
    make_diffusion_pre_post_processors,
)

from .configuration_diffusion_fwd import DiffusionFwdConfig


def make_diffusion_fwd_pre_post_processors(
    config: DiffusionFwdConfig,
    dataset_stats: dict | None = None,
    **kwargs,
):
    return make_diffusion_pre_post_processors(
        config, dataset_stats=dataset_stats, **kwargs
    )
