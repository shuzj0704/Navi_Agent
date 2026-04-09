# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NaviAgent -- 跨环境长程导航（CELN）研究项目，目标投稿 CoRL/NeurIPS/RSS 2026。当前阶段为 proposal 规划 + 仿真环境搭建。文档以中文为主。

核心卖点：统一导航 Token 架构——将 Agent 的工具交互、空间记忆、推理过程和动作输出统一到 VLM 的 embedding 空间中（tool tokens + differentiable memory + latent reasoning），端到端可微，2Hz 实时部署。

## 方案演进

1. **Plan B（已放弃）** `plan_B/` -- NavigateAnywhere，以 benchmark 为核心贡献，方法上用 GPS HDOP 滞回规则做模式切换。觉得 novelty 不够，放弃。
2. **Plan A v1** `docs/proposal/NaviAgent_v1.md` -- 初版，以 agent 范式为卖点，ReAct-style SFT 文本工具调用。
3. **Plan A v2** `docs/proposal/NaviAgent_v2.md` -- 系统审阅后修改：补 POMDP 形式化、Adaptive Thinking（回应 Aux-Think）、新增 ARNA 对比。但文本工具接口仍是"工程拼接"。
4. **Plan A v3（当前方向）** `docs/proposal/NaviAgent_v3.md` -- 架构重构：用 learned tool tokens（ToolkenGPT 范式）替代文本工具调用，用 differentiable memory bank 替代外部拓扑图，用 latent thinking tokens（LaRA-VLA 范式）替代文本 CoT，用 spatial/latent goal dual-head 替代文本 pixel goal。

## 技术要点

- **VLM**: Qwen3.5-9B（原生多模态 early fusion，256K context 可扩展至 1M，75% GDN线性注意力+25% Full Attention 混合架构，hidden_size=4096，vocab=248K），扩展 ~130 NaVocab tokens（Re-Initialization 对齐词空间）
- **架构**: DualVLN 双系统 -- System 2（VLM + NaVocab + Differentiable Memory + Dual-Head, 2Hz）+ System 1（DiT, 30Hz, conditioned on pixel goal + latent goal）
- **Tool Tokens**: `<tool:mem_w>`, `<tool:mem_r>`, `<tool:route>`, `<tool:progress>`, `<tool:floor>`（learned embeddings，非文本函数调用）
- **Differentiable Spatial Memory**: Memory Bank 作为 VLM 上下文 tokens，写入/读取均可微
- **Latent Adaptive Reasoning**: 3-stage curriculum（explicit CoT → latent thinking tokens → adaptive depth）
- **Depth**: Depth Positional Encoding（借鉴 SpatialVLA），D435i 实测 depth → backproject → 正弦编码 → 加到 ViT patch token 上，不改 ViT 架构
- **训练数据**: Teacher Model（Qwen3.5-72B/9B）生成因果推理链（thinking → tool → action），而非规则模板标注
- **训练**: Stage 1 Explicit CoT SFT → Stage 2 Latent Distillation → Stage 3 RFT（IQL/GRPO）
- **训练基础设施**: 8×A100 80GB, BF16, ZeRO-2, flash_attention_2, 基于 InternVLA-N1 代码库适配

## Conda 环境

| 环境名 | Python | 用途 | 激活方式 |
|--------|--------|------|---------|
| `habitat` | 3.9 | 室内仿真渲染（Habitat-Sim 0.3.3 + Habitat-Lab 0.3.3） | `conda activate habitat` |
| `metaurban` | 3.10 | 室外仿真渲染（MetaUrban 0.0.1 + SB3 + PyTorch） | `conda activate metaurban` |

两个环境不可混装（渲染引擎冲突：Habitat 用 EGL/OpenGL，MetaUrban 用 Panda3D）。

## 关键文档

| 文件 | 内容 |
|------|------|
| `docs/proposal/NaviAgent_v3.md` | **当前版本 proposal**（架构、方法、实验、timeline） |
| `docs/proposal/数据流水线.md` | 数据准备全流程（仿真器→渲染→Teacher标注→训练数据→验证） |
| `docs/proposal/参考文献简介.md` | 55+ 篇参考论文分类简介（输入/输出/创新点/与NaviAgent关联） |
| `docs/proposal/Framework_v3.png` | 系统架构图（matplotlib 生成） |
| `docs/proposal/gen_framework_v3.py` | 架构图生成脚本 |
| `docs/AMAP/AMAP_CV_Lab_导航论文全景解读.md` | AMAP CV Lab 10 篇导航论文全景分析（技术脉络、继承关系、设计模式） |
| `docs/outdoor_paper/仿真器papers.md` | UrbanVerse + Urban-Sim 对比分析，含 NaviAgent 集成策略 |
| `docs/outdoor_paper/OmniNav_解读.md` | OmniNav 解读 + 与 InternVLA-N1 对比 |
| `docs/outdoor_paper/SocialNav_解读.md` | SocialNav Brain-Action 架构、SAFE-GRPO |
| `docs/outdoor_paper/zero_shot_outdoor_nav_方案.md` | 零样本室外导航方案（Qwen3.5-9B + MetaUrban） |

## 已知坑点

### 训练相关
- Qwen3.5-9B 初始 loss 可能异常高：LR 须降至 1e-5（非 2e-5），warmup 设 0.01（沿用 Qwen3-VL 经验，需实测确认）
- Qwen3.5 的 Gated DeltaNet 混合注意力需要适配 InternVLA-N1 训练代码（原代码假设纯 Full Attention）
- liger-kernel 极慢，不要用
- 需要 transformers >= 4.57.0（当前需从 GitHub main 分支安装：`pip install transformers @ git+https://github.com/huggingface/transformers.git@main`）

### MetaUrban 相关
- 首次运行需填表获取 asset code（https://forms.office.com/r/tFBRFk7u4E），assets 已下载完成
- `image_observation=True` 时某些 seed 会在 reset 后立即 `arrive_dest=True`，需要检查跳过
- depth_camera 输出可能不在 obs dict 的顶层（需进一步调查）

### Isaac Sim 相关
- 脚本必须通过 Isaac Sim 自带 Python 运行（`./python.sh`），不可在 conda 环境中运行
- `SimulationApp` 必须在所有 `omni.*` import 之前创建，`IsaacSimEnv` 已封装此约束
- Policy joint order 与 Isaac Sim articulation DOF order 不同，`Go2WRobot` 内部通过 `joint_index_map` 做双向映射

## 仿真数据资源

| 资源 | 路径 | 状态 |
|------|------|------|
| MetaUrban 测试轨迹 | `data/metaurban_test/episode_0000/` | 已采集（231 步，SR 95.2%） |
| CraftBench 12 场景 | `data/UrbanVerse-CraftBench/` | 已下载 7/12 场景（scene_01~07），USD 格式，IsaacSim 4.5.0 打开 |
| CraftBench 原始副本 | `/home/shu22/navigation/urban_verse/CraftBench/` | 全 12 场景 + 场景文件说明文档 |
| Urban-Sim（程序化仿真） | 未 clone | 开发阶段首选平台，github.com/metadriverse/urban-sim |
| HM3D 室内场景 | 未下载 | 需 Matterport API token |

## 参考论文 `docs/paper/`

| 简称 | 文件前缀 | 与本项目的关系 |
|---|---|---|
| DualVLN | Ground Slow, Move Fast | 基础架构（双系统 VLN） |
| InternVLA-N1 | InternVLA_N1 | 训练方案迁移来源 |
| UrbanVLA | Li et al. 2025 | 室外 Route Following + IQL RFT |
| CogNav | Cao et al. 2025 | 认知推理导航（Plan A 最近的 prior work） |
| MapNav | Zhang et al. 2025 MapNav | 语义地图记忆 |
| NavFoM | Zhang et al. 2025 Embodied | 导航基础模型，viewpoint indicator |
| ASCENT | Gong et al. 2026 | 多楼层零样本 ObjectNav |
| BridgeNav | Zhao et al. 2026 | 室内外过渡（仅 out-to-in） |

AMAP CV Lab 10 篇系列论文（CE-Nav, Nav-R², AstraNav-Memory, OmniNav, FantasyVLN, BridgeNav, ABot-N0, SocialNav, JanusVLN, NavForesee）详见 `docs/AMAP/`。

## Isaac Sim 室外仿真 `src/sim_vln_outdoor/`

模块化结构，env / robot / nav 与 scripts / assets 平级：

```
src/sim_vln_outdoor/
├── scripts/                     # 入口脚本
│   ├── load_scene.py            # 仅加载 USD 场景
│   ├── load_scene_robot.py      # 场景 + Go2W + RL policy 完整 pipeline
│   ├── load_scene_view.py       # D435i 相机视角渲染 + 键盘/Socket 控制
│   └── nav_eval.py              # 闭环导航控制器评估
├── nav/                         # 导航控制器接口
│   ├── controller.py            # NavController ABC + Observation / Action 定义
│   └── demo_controllers.py      # ForwardOnly / RandomWalk 示例控制器
├── env/
│   └── isaac_env.py             # IsaacSimEnv: SimulationApp + USD 加载 + World
├── robot/
│   ├── go2w.py                  # Go2WRobot: URDF 导入 + 关节映射 + 状态读写
│   └── go2w_policy.py           # Go2WPolicy: obs 构建 + TorchScript 推理 + PD torque
├── data/
│   └── view/make_video.py       # 快照拼接为 30fps H.264 视频
└── assets/
    ├── policy/<robot>/          # RL policy（base.yaml + framework/config.yaml + *.pt）
    └── rl_sar_zoo/<robot>_description/  # URDF/MJCF/mesh
```

### 关键约束

- `IsaacSimEnv` 必须在所有 `omni.*` import 之前实例化（Isaac Sim 要求 `SimulationApp` 先创建）
- 脚本通过 `sys.path.insert` 把 `sim_vln_outdoor/` 加入搜索路径，用 `from env import IsaacSimEnv` 导入
- 运行方式：`./python.sh src/sim_vln_outdoor/scripts/<script>.py`（Isaac Sim 自带 Python，不用 conda）

### nav_eval.py 闭环评估

渲染 D435i 图像 → 控制器输出动作 → 更新相机位姿 → 循环。用于验证导航控制器在仿真场景中的表现。

```bash
# 直行 + headless + 保存帧（默认 ForwardOnly 控制器）
./python.sh src/sim_vln_outdoor/scripts/nav_eval.py --headless --save-frames --max-steps 200

# 自定义控制器，2Hz
./python.sh src/sim_vln_outdoor/scripts/nav_eval.py \
    --headless --controller "my_module:MyController" --controller-freq 2.0
```

自定义控制器继承 `NavController`，实现 `act(obs) -> Action`。输出保存到 `data/urbanverse/nav_eval/<timestamp>/`（trajectory.jsonl + summary.json + 可选帧图片）。详见 `src/sim_vln_outdoor/README.md`。

## 脚本 `scripts/`

| 脚本 | 用途 | 环境 |
|------|------|------|
| `scripts/metaurban/single_trajectory.py` | MetaUrban 单条轨迹采集（RGB + 位姿 + 动作） | `conda activate metaurban` |
| `scripts/utils/reorganize_refs.py` | 参考文献重组工具 | 任意 |
| `scripts/utils/update_refs.py` | 参考文献更新工具 | 任意 |
| `scripts/reference/app.py` | VLN Demo 参考脚本（外部依赖，不可直接运行） | — |
| `docs/proposal/gen_framework_v3.py` | 生成 Framework_v3.png 架构图 | 任意（matplotlib） |

## 文件约定

- Proposal 以文件名后缀版本号递增（v1, v2, ...），始终以最高版本为准
- `plan_B/论文Proposal审阅提示词.txt` 是用于模拟顶会审稿的 prompt
- 数据存放在 `data/`（已 gitignore），README.md 记录每次采集的配置和结果
- 所有 markdown 文档保持飞书兼容格式（无 jsonc、无 HTML 标签）
- `.claudeignore` 排除了 PDF/DOCX 等二进制文件，避免读入上下文
