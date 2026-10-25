# DP-Thalamic: Top-Down Gated Diffusion Policy on LIBERO-Spatial

A controlled, single-variable comparison on **LIBERO-Spatial**:

- **Baseline**: LeRobot 0.6.1 native `diffusion` policy (DP).
- **Variant**: `dp_thalamic`, the same Diffusion Policy with a thalamic-style **top-down gate** added to the U-Net decoder's skip connections.

Everything else — dataset, seed (1000), training steps (200k), batch size (64), optimizer, noise scheduler, data pipeline, and evaluation protocol — is held identical.

## The mechanism

In the U-Net decoder, right before every

```python
torch.cat((x, encoder_skip_features.pop()), dim=1)
```

the coarser decoder feature `x` modulates the corresponding encoder skip feature through a **low-rank top-down gate** (`gate_rank_ratio=16`, `gate_min_rank=16`). The gate is residual with a learnable strength `gamma` initialized to **0**, so the modified network is exactly the original U-Net at initialization.

Nothing else is changed: no auxiliary losses, no future-observation prediction, no iterative belief propagation, and no changes to DDPM/DDIM, horizon, optimizer, or data augmentation.

## Results (LIBERO-Spatial, 200k steps, seed=1000, 10 tasks × 10 episodes)

| Policy | Success rate |
|---|---|
| `diffusion` (baseline) | **73%** (73/100) |
| `dp_thalamic` (top-down gate) | **62%** (62/100) |

Per-task successes (out of 10):

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 8 | 9 | 10 | 10 | 4 | 2 | 9 | 5 | 9 | 7 |
| dp_thalamic | 8 | 9 | 10 | 10 | 5 | **0** | 8 | 4 | 5 | 3 |

In this single-seed setting the gated variant **underperforms** the baseline, driven largely by task 5 (2/10 → 0/10). Results come from one seed only; no statistical significance is claimed.

Full metrics: [`dp_baseline_s2/eval_200000/eval_info.json`](dp_baseline_s2/eval_200000/eval_info.json), [`dp_thalamic_seed1000/eval_200000/eval_info.json`](dp_thalamic_seed1000/eval_200000/eval_info.json). Per-episode videos are included under each `eval_200000/videos/`.

## Repository layout

```
dp_code/
  SOURCE.md            # provenance of the source snapshot (lerobot 0.6.1 site-packages)
  original/            # pristine lerobot 0.6.1 source, for diffing
  dp_thalamic/         # modified workspace, loaded via PYTHONPATH
    lerobot/policies/dp_thalamic/
      configuration_dp_thalamic.py
      modeling_dp_thalamic.py
      processor_dp_thalamic.py
      topdown_thalamic_gate.py   # the gate module
    tests/policies/dp_thalamic/  # unit tests
run_dp_baseline_s2.sh           # baseline training entry
run_dp_thalamic_seed1000.sh     # gated-variant training entry (waits for baseline GPU)
run_dp_eval_s2.sh               # baseline evaluation entry
run_dp_eval_thalamic.sh         # variant evaluation entry
*.log                           # full training / evaluation logs
```

Code changes vs. the pristine source are limited to:

1. New `dp_thalamic` policy package (`lerobot/policies/dp_thalamic/`) and its registration in `lerobot/policies/__init__.py` — the original `diffusion` policy is untouched and still usable.
2. Two evaluation-only fixes in `lerobot/envs/libero.py`: remapping the gripper action sign (dataset stores 0=close/1=open, robosuite expects ±1) and flipping robosuite camera frames to the dataset orientation.

## Setup

Tested on Ubuntu, single RTX 4090, conda environment `lerobot`, Python 3.12.

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
pip install lerobot==0.6.1

# LIBERO benchmark (robosuite-based evaluation env)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
```

Offscreen rendering for evaluation uses osmesa:

```bash
export MUJOCO_GL=osmesa   # requires libosmesa6 (e.g. apt install libosmesa6-dev)
```

## Dataset

`nvidia/LIBERO_LeRobot_v3` (LeRobot dataset format v3.0), **libero_spatial** subset:

- 432 episodes, 52,970 frames, 10 tasks, 20 FPS, Franka Panda arm
- Observations: `agentview_image` + `robot0_eye_in_hand_image` (resized to 256×256) and 8-dim proprioceptive state
- Actions: 7-dim (6-DoF end-effector + gripper)

Download only the libero_spatial subset:

```bash
huggingface-cli download nvidia/LIBERO_LeRobot_v3 \
  --repo-type dataset \
  --include "libero_spatial/*" \
  --local-dir <DATA_ROOT>
# dataset root for the commands below is then <DATA_ROOT>/libero_spatial
```

## Reproduce

All commands below are also collected in the `run_*.sh` scripts (adjust paths to your machine).

### 1. Train the baseline

```bash
lerobot-train \
  --policy.type=diffusion \
  --policy.push_to_hub=false \
  --dataset.repo_id=nvidia/LIBERO_LeRobot_v3 \
  --dataset.root=<DATA_ROOT>/libero_spatial \
  --output_dir=<OUT>/dp_baseline \
  --wandb.enable=false \
  --steps=200000 \
  --save_freq=100000 \
  --batch_size=64
```

### 2. Train the gated variant

The modified source tree is loaded via `PYTHONPATH`, leaving the installed `lerobot` package untouched:

```bash
export PYTHONPATH=<REPO>/dp_code/dp_thalamic

lerobot-train \
  --policy.type=dp_thalamic \
  --policy.push_to_hub=false \
  --policy.use_topdown_gate=true \
  --policy.gate_rank_ratio=16 \
  --policy.gate_min_rank=16 \
  --dataset.repo_id=nvidia/LIBERO_LeRobot_v3 \
  --dataset.root=<DATA_ROOT>/libero_spatial \
  --output_dir=<OUT>/dp_thalamic \
  --wandb.enable=false \
  --seed=1000 \
  --steps=200000 \
  --save_freq=100000 \
  --batch_size=64
```

### 3. Evaluate

Identical protocol for both policies (10 episodes per task, 10 LIBERO-Spatial tasks):

```bash
export MUJOCO_GL=osmesa
# only needed for the dp_thalamic checkpoint:
export PYTHONPATH=<REPO>/dp_code/dp_thalamic

lerobot-eval \
  --policy.path=<OUT>/<RUN>/checkpoints/200000/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}' \
  --eval.n_episodes=10 \
  --eval.batch_size=5 \
  --output_dir=<OUT>/<RUN>/eval_200000
```

### 4. Unit tests

```bash
cd dp_code/dp_thalamic
PYTHONPATH=$PWD pytest tests/policies/dp_thalamic -q
```

## Notes

- Model checkpoints (~2 GB each) are **not** included in this repository; only evaluation metrics and videos are. Regenerate them with the training commands above.
- The baseline run relies on the LeRobot 0.6.1 default seed (1000), which matches the explicitly passed `--seed=1000` of the variant run.
- Training/evaluation logs for both runs are committed as `dp_baseline_s2.log`, `dp_thalamic_seed1000.log`, `dp_eval_s2.log`, and `dp_eval_thalamic.log`.
