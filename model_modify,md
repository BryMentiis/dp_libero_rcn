# 模型修改实施规格书：类脑机制增强的 Diffusion Policy

> 用途：交给 VS Code / Kimi Code 等代码工具作为改造依据。本文只保留结构定义、损失函数、插入点与验收协议；叙事与文献归属见技术报告 HTML 版。
> 基座：Diffusion Policy（ResNet 视觉编码 + FiLM 条件化去噪头），基准 LIBERO-Spatial（每任务 50 条演示）。

## 0. 总原则（硬约束）

1. **主机制只参与训练**：前向模型组在推理时整组移除，推理路径与原版 DP 逐算子一致。
2. **变量隔离**：基线与机制臂同架构、同数据、同 seed、同步数，唯一变量是机制本身。
3. **门控臂（可选）是受控例外**：它在推理时生效，所有含门控的结果单独成表归因，不与主机制混排。

## 1. 需要新增的模块

### 1.1 前向模型组（主机制，仅训练期挂载）

| 预测头 | 输入 | 预测目标 | horizon |
|---|---|---|---|
| P_high | z_high(t), 动作块 a(t:t+8) | sg[z_high(t+8)] | k=8 |
| P_mid  | z_mid(t), 动作块 a(t:t+2)  | sg[z_mid(t+2)]  | k=2 |
| P_q    | z_high(t), q_t, 动作块 a(t:t+6) | q(t+6)（本体真值，无 sg 需求外的坍缩风险） | k=6 |

- z_high / z_mid：ResNet 编码器的高层 / 中层特征（层级来源）。
- **三个预测头独立参数、不共享**（clone 结构启发）。
- 建议结构：每层预测头为 2–3 层 MLP（动作条件通过 concat 或 FiLM 注入）。
- sg[·] 为停梯度：未来真值编码不回传梯度，梯度只更新编码器与预测头。

### 1.2 辅助损失（仅训练）

```
L = L_diffusion
  + λ  * Σ_{l ∈ {mid, high}} ‖ sg[z_l(t + k_l)] − P_l(z_l(t), a(t:t+k_l)) ‖²
  + λq * ‖ q(t+6) − P_q(z_high(t), q_t, a(t:t+6)) ‖²
```

- λ、λq 为新超参，初始建议 λ=1.0、λq=1.0，后续做敏感性消融。

### 1.3 跨模态一致性门控（可选诊断臂，时间盒半天）

```
ê = h(z_mid)                     # 辅助头：视觉估计的末端位姿
s = exp( −‖ ê − q_eef ‖² / τ )   # 一致性分数
c_film = FiLM( s · z_vis ⊕ q_t ) # 视觉条件按 s 缩放后进入 FiLM
```

- 挂载点：FiLM 条件化之前——视觉信息进入动作头的唯一闸门，最小侵入。
- **训练时必须注入错位**（否则学到 s≡1 平凡解）：对 20–30% 样本注入视觉-本体错位（用 t−Δ 帧图像配 t 时刻本体，或图像空间扰动）。
- 仅在主臂结果指向视觉条件依赖时启动。

## 2. 插入点定位清单（改代码前先跑）

```bash
grep -rn "diffusion" libero/lifelong/ --include="*.py" -il   # 确认 DP 落点
grep -rn "class.*Policy" libero/lifelong/algos/ --include="*.py"
grep -rn "FiLM\|film" libero/ --include="*.py" -l             # 条件融合位置
grep -rn "def loss\|compute_loss" libero/lifelong/ --include="*.py"  # 损失汇总处
```

在策略类中记录三处「文件:行号」：①编码器中层/高层特征出口 ②损失函数汇总处 ③FiLM 条件化之前（门控预留）。

## 3. 实施顺序与验收门

### Step 1 — 过拟合冒烟（半天，不过不进下一步）
- 10 条演示、几十步训练。
- **防坍缩验收协议（三项全过才算过）**：
  1. 辅助 loss 下降但不趋零；
  2. 预测误差随 horizon 单调上升（k=2 误差 < k=8）；
  3. 目标端参数确认无梯度流（检查 sg 生效）。
- 本体预测支无坍缩风险，不受此协议约束。

### Step 2 — 推理路径等价验证（半小时）
- 加载同一 checkpoint，推理时移除全部预测头；
- 验证：参数量与基线一致、相同输入下动作输出与基线逐算子一致（或差异为 0）、延迟一致。

### Step 3 — 正式实验矩阵

| 实验臂 | 配置 | 数据规模 | 角色 |
|---|---|---|---|
| 基线 | DP 原版 | 50 / 25 / 10 | 对照 |
| +前向模型组（主） | P_high(k=8) + P_mid(k=2) + P_q | 50 / 25 / 10 | 主机制 |
| +一致性门控（可选） | 辅助头 + s + FiLM 门控（含错位注入） | 50 | 单独归因 |
| 消融 | 层级×horizon 绑定 vs 平铺同 k；sg 开/关 | 50 | 机制拆解 |
| 扰动评测 | 遮挡（上下文遮块）vs 删除（纯灰遮块） | — | 生成式机制判别性检验 |

- 评测：每臂 50 个保留初始状态 rollout，报成功率；记录训练曲线、辅助 loss、逐层预测误差随 horizon 变化。
- 对照硬约束：同 seed、同步数、同数据；所有命令与种子入附录。

### Step 4 — 回填报告
- 结果回填技术报告 §06 表 1/表 2/图 2/图 3；负结果如实写入 §08。

## 4. 环境复用提醒（已验证，勿重排）

- Python 3.10 独立 env；robosuite==1.4.1；mujoco==2.3.7；setuptools<81。
- 训练冒烟：`python libero/lifelong/main.py seed=1 benchmark_name=LIBERO_SPATIAL policy=<policy> lifelong=base train.n_epochs=3 train.num_workers=0`。
- evaluate.py 的 env_num=20 需按卡阶梯测试（EGL_BAD_ALLOC 先例）。
- 过夜点火前先用 `eval.eval_every=1 eval.num_procs=2 eval.n_eval=2` 验证一个评测点。
