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

## 代码架构

项目分三个独立运行层，通过 HTTP 通信解耦：

```
naviagent (任意 Python 环境, 无 habitat 依赖)
  ├── perception: 观测读取 (ObsBundle, SimClientObsReader) + 点云 + YOLOE/SAM3 分割 + 语义地图
  ├── decision:   NavigationEngine + DWA + 转向控制 + TaskOrchestrator
  ├── vlm:        VLMNavigator (System1) + System2Planner
  └── common:     坐标变换 + NavState + 视角常量 + 可视化

sim_vln_indoor (conda activate habitat, Python 3.9)
  └── env/server/ — FastAPI 仿真 HTTP 服务器 (端口 5100)
  └── env/client/ — SimClient (httpx), naviagent 通过此客户端调用仿真

sim_vln_outdoor (Isaac Sim 自带 Python, ./python.sh)
  └── Isaac Sim 渲染 + NavController 接口
```

## 室内导航运行方式

采用**双进程架构**，仿真服务器和导航 Agent 分开运行：

```bash
# Terminal 1: 启动仿真服务器 (habitat 环境)
conda activate habitat
cd src
python -m sim_vln_indoor.env.server
# 默认 0.0.0.0:5100, 可选 --port 5200

# Terminal 2: 启动导航 Agent (任意环境)
python src/scripts/nav_main.py --sim-url http://localhost:5100 --scene 17DRP5sb8fy --steps 100
python src/scripts/nav_main.py --mock --scene 17DRP5sb8fy   # 不调 VLM
python src/scripts/batch_eval.py --split val_seen --max-episodes 5 --steps 100
python src/scripts/batch_eval.py --eval-set quick_16 --steps 100  # 快速评测集 (16 eps, baseline SR≈50%)
```

### System1 消融开关

`batch_eval.py` 强制 `--no-planner` 表示只跑快系统。通过以下 CLI 可切换 `AblationConfig`
(定义在 `src/naviagent/vlm/vlm_navigator.py`, 默认 = 当前主干行为):

| Flag | 取值 | 含义 |
|------|------|------|
| `--views` | `front,left,right` (默认) / `front` / `front,left,right,back` | RGB 输入视角, back 需在 `sim_server.yaml` 启用 `back_rgb` 并重启仿真 |
| `--output-mode` | `direction` (默认) / `pixel` | VLM 输出 F/L/R/STOP 或 v,vx,vy 像素目标 (pixel 模式复用老 DWA 反投影路径) |
| `--semantic-mode` | `text` (默认) / `image` / `none` | 语义图作为结构化文字 / 俯视图片 / 不传 |
| `--image-memory-len N` | 默认 8 | 前视图记忆长度 (0 禁用), 长度采样阈值见 `FRONT_MEMORY_MIN_DIST/_ANGLE` |
| `--action-history-len N` | 默认 20 | 决策动作历史长度 (0 禁用) |
| `--pose-history-len N` | 默认 20 | 位姿历史长度 (0 禁用) |
| `--ablation-tag NAME` | — | 结果目录后缀, 便于 diff 多次 run |

批量跑全部消融实验: `bash src/scripts/run_ablation_matrix.sh` — 14 个配置, 跑完用
`python src/scripts/aggregate_ablation.py` 汇总成 `output/eval/ablation_0419_report.md`。

验证仿真服务器：
```bash
curl http://localhost:5100/health
curl http://localhost:5100/scenes
```

室内 VLM 推理默认连接 `http://10.100.0.1:8000/v1`（代码中 `VLM_API_URL` 常量）。

## Conda 环境

| 环境名 | Python | 用途 | 激活方式 |
|--------|--------|------|---------|
| `naviagent` | 3.12 | 运行 naviagent client 端所有代码（perception/decision/vlm）+ SAM3 + YOLOE；**不含仿真** | `conda activate naviagent` |
| `habitat` | 3.9 | 室内仿真服务器（Habitat-Sim 0.3.3 + FastAPI + uvicorn） | `conda activate habitat` |
| `metaurban` | 3.10 | 室外仿真渲染（MetaUrban 0.0.1 + SB3 + PyTorch） | `conda activate metaurban` |

仿真环境不可互装（渲染引擎冲突：Habitat 用 EGL/OpenGL，MetaUrban 用 Panda3D），也不要和 `naviagent` 混装。

`naviagent` env 关键依赖：
- torch 2.9.0+cu128 / torchvision 0.24+cu128（RTX 5060 Ti Blackwell 必须 cu128）
- `sam3`（源码安装自 `third_party/sam3/`，官方仓库 facebookresearch/sam3）
- `ultralytics`（YOLOE 备用）
- numpy<2 / opencv-python<4.11（与 SAM3 的 numpy<2 约束兼容）
- setuptools<81（SAM3 用 `pkg_resources`，Py3.12 + setuptools 81 会移除）
- openai, httpx, scipy, pyyaml, matplotlib, pillow, timm, einops, pycocotools

搭建步骤（新机器）：
```bash
conda create -n naviagent python=3.12 -y
conda activate naviagent
pip install torch==2.9.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple   # pypi.nvidia.com 被墙时必备
git clone https://github.com/facebookresearch/sam3.git third_party/sam3
pip install -e third_party/sam3 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install "opencv-python<4.11" "numpy<2" "setuptools<81" \
    openai scipy ultralytics matplotlib einops pycocotools \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
huggingface-cli login   # 需在 HF facebook/sam3 页面同意许可后才能下载权重
```

## 模型权重 (git-lfs)

`models/` 目录下的 `.pt` 文件通过 git-lfs 管理。克隆后必须拉取：

```bash
git lfs install
git lfs pull
```

当前模型：
- `models/yoloe-11l-seg.pt` (68MB, YOLOE 开放词汇分割)
- SAM3 checkpoint — 非 lfs，首次调用 `Sam3Segmentor` 时自动从 HF `facebook/sam3` 下载到 `~/.cache/huggingface/`。需先 `huggingface-cli login` 并在网页同意许可。

## 语义分割器选择

`src/naviagent/perception/` 下两个分割器共用 `Segment` 接口，可互换：

| 分割器 | 优势 | 代价 |
|--------|------|------|
| `YOLOESegmentor` | 小模型快（68MB，CPU 可跑），在 habitat env 也能用 | 需要预先 `set_classes`，开放词汇泛化弱于 SAM3 |
| `Sam3Segmentor` (base) | 概念级 open-vocab 零样本分割，mask 质量高 | 848M 参数、必须 cu128 + `naviagent` env，图片级单轮只接受一个 text prompt（内部循环 classes） |

用法示例（图片模式）：
```bash
conda activate naviagent
python src/scripts/test_sam3.py --image path/to/img.jpg --classes "chair,table,sofa"
```

## 机器与远程访问

| 角色 | 主机 | 用途 |
|------|------|------|
| 本机（local） | `shu22@shu22` | Isaac Sim 5.1 仿真 + 数据采集 + 调试，单卡 RTX 4070 Ti SUPER (16GB) |
| 远程服务器 | `ps@ps2` | VLM 推理（Qwen3.5-9B / Qwen3-VL-8B 的 vLLM 服务）、训练 |

- 连接方式：本机已经在 `~/.ssh/config` 配好别名,直接 `ssh ps2` 即可登录,无需 `user@host` 写法
- **典型工作流**：仿真在本机跑,VLM 推理走远程,通过 SSH 端口转发把远程 vLLM 暴露到本机:
  ```bash
  # 本机 terminal: 把 ps2 上的 vLLM (本例端口 8000) 映射到本机 18004
  ssh -fN -L 18004:localhost:8000 ps2
  curl http://localhost:18004/v1/models   # 验证 tunnel 通了
  # 然后本机跑仿真,显式 --base-url http://localhost:18004/v1
  ```
- **常见坑**：远程 vLLM 的 `--port` 取决于实际启动命令(可能不是仓库默认 8004),用前 `ssh ps2 'ss -tlnp | grep python'` 确认真实端口

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
- `python.sh` wrapper 默认走 `/home/shu22/nvidia/isaacsim_5.1.0`；新机器需 `export ISAACSIM_ROOT=...` 覆盖
- 用 VLM controller 前必须 `./python.sh -m pip install openai`，否则 `from vlm_serve.client import VLMClient` 会 ModuleNotFoundError

### 室内仿真相关
- 仿真服务器 (`sim_vln_indoor.env.server`) 必须在 `habitat` conda 环境中运行
- 导航 Agent (`nav_main.py` / `batch_eval.py`) 可在任意 Python 环境运行，不需要 habitat
- Habitat 返回的 numpy float32 不能直接 `json.dumps`，已在 `habitat_backend.py` 中转为 Python float
- `SimClient.get_observations()` 返回的 RGB 图已经是 BGR 格式（服务端做了 RGB→BGR 转换），不要重复转换

### 全新服务器部署
- 室内仿真部署只需 `src/sim_vln_indoor/` (~50KB) + 场景数据 + habitat conda 环境
- 室外仿真部署见 [README.md](README.md) 第 4 节
- **必须改的硬编码路径**：
  - `src/sim_vln_indoor/env/config/sim_server.yaml` — `scenes.base_dir` (MP3D 场景路径)
  - `src/scripts/batch_eval.py` — `DATASET_DIR`, `SCENE_DIR` (VLN-CE 数据集路径)
  - `src/scripts/nav_main.py` / `batch_eval.py` — `VLM_API_URL` (VLM 服务地址)
  - [python.sh](python.sh) — `ISAACSIM_ROOT`
  - [src/vlm_serve/configs/qwen3_5_9b.yaml](src/vlm_serve/configs/qwen3_5_9b.yaml) — `model_path/gpu`
  - [src/vlm_serve/configs/qwen3_vl_8b.yaml](src/vlm_serve/configs/qwen3_vl_8b.yaml) — `model_path/gpu`
- `data/` 整体被 gitignore，但 `data/urbanverse/trajectory/scene_*/blog_point.txt`（人工标注的导航路线 waypoint 文件）通过 `.gitignore` 例外规则**入库**；由它生成的 `dense_trajectory.json/.png` 不入库，每次拉下来后需要跑一次 `scripts/utils/interpolate_trajectory.py` 重新生成

## 仿真数据资源

| 资源 | 路径 | 状态 |
|------|------|------|
| MP3D 室内场景 | `data/scene_data/mp3d/` | Habitat 室内仿真 (~21GB) |
| R2R VLN-CE 数据集 | `data/vln_ce/R2R_VLNCE_v1-3/` | 室内导航评测数据 |
| MetaUrban 测试轨迹 | `data/metaurban_test/episode_0000/` | 已采集（231 步，SR 95.2%） |
| CraftBench 12 场景 | `data/UrbanVerse-CraftBench/` | 已下载 7/12 场景（scene_01~07），USD 格式，IsaacSim 4.5.0 打开 |
| CraftBench 原始副本 | `/home/shu22/navigation/urban_verse/CraftBench/` | 全 12 场景 + 场景文件说明文档 |
| YOLOE 模型 | `models/yoloe-11l-seg.pt` | git-lfs 管理 (68MB) |

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
│   ├── nav_eval.py              # 闭环导航控制器评估（通用，--controller 插件式）
│   └── vlm_gps_nav.py           # GPS 引导 VLM 闭环导航（任务专属入口）
├── nav/                         # 导航控制器接口
│   ├── controller.py            # NavController ABC + Observation / Action 定义
│   ├── demo_controllers.py      # ForwardOnly / RandomWalk 示例控制器
│   ├── vlm_controller.py        # 基础 VLM 控制器（仅看 RGB，无目标感知）
│   └── gps_vlm_controller.py    # GPS 引导 VLM 控制器（RGB + 结构化文本 prompt）
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

### vlm_gps_nav.py — GPS 引导 VLM 闭环导航

```bash
# 1. 稀疏 waypoint → 稠密轨迹
python scripts/utils/interpolate_trajectory.py \
    --input data/urbanverse/trajectory/scene_09/blog_point.txt \
    --step 0.5 --visualize

# 2. 起 Qwen3-VL 服务（另一终端，lwy_swift 环境）
python scripts/serve/start_qwen3vl.py

# 3. 跑闭环
xvfb-run -a ./python.sh src/sim_vln_outdoor/scripts/vlm_gps_nav.py \
    --headless --max-steps 200 --controller-freq 1.0 \
    --trajectory data/urbanverse/trajectory/scene_09/dense_trajectory.json
```

输出在 `data/urbanverse/vlm_gps_nav/<timestamp>/`。

## VLM 部署 `src/vlm_serve/`

封装 vLLM 服务启动 + OpenAI client 调用，供 Teacher Model 标注、推理评估、交互测试统一复用。**新增需要调 VLM 的代码请直接 `from vlm_serve.client import VLMClient`，不要再写新的 OpenAI client 包装。**

```bash
# 启动 Qwen3-VL-8B 服务（需 lwy_swift 环境）
python scripts/serve/start_qwen3vl.py

# 交互测试
python scripts/serve/chat_test.py --base-url http://localhost:8004/v1 --model qwen3-vl
```

服务进程需在装了 vLLM 的 conda 环境（`lwy_swift`）下运行；client 调用方只需 `openai` 包，可在任意环境中执行。

## 脚本 `scripts/`

| 脚本 | 用途 | 环境 |
|------|------|------|
| `scripts/metaurban/single_trajectory.py` | MetaUrban 单条轨迹采集（RGB + 位姿 + 动作） | `conda activate metaurban` |
| `scripts/serve/start_qwen35.py` | 启动 Qwen3.5-9B vLLM 服务 | `conda activate lwy_swift` |
| `scripts/serve/start_qwen3vl.py` | 启动 Qwen3-VL-8B-Instruct vLLM 服务 | `conda activate lwy_swift` |
| `scripts/serve/chat_test.py` | 通用 vLLM 服务交互测试客户端 | 任意（仅需 openai 包） |
| `scripts/utils/interpolate_trajectory.py` | 把稀疏 waypoint 文件线性插值成稠密轨迹 JSON（含 PIL 可视化） | 任意（numpy + Pillow） |
| `scripts/utils/reorganize_refs.py` | 参考文献重组工具 | 任意 |
| `scripts/utils/update_refs.py` | 参考文献更新工具 | 任意 |
| `scripts/reference/app.py` | VLN Demo 参考脚本（外部依赖，不可直接运行） | -- |
| `docs/proposal/gen_framework_v3.py` | 生成 Framework_v3.png 架构图 | 任意（matplotlib） |

## 评测集

评测集配置存放在 `src/scripts/eval_sets/` 下，JSON 格式，通过 `--eval-set` 参数加载：

| 名称 | Episodes | 场景数 | Baseline SR | 用途 |
|------|----------|--------|-------------|------|
| `quick_16` | 16 | 11 | ≈50% | 快速迭代验证（8 成功 + 8 near-miss 失败） |

## 文件约定

- Proposal 以文件名后缀版本号递增（v1, v2, ...），始终以最高版本为准
- `plan_B/论文Proposal审阅提示词.txt` 是用于模拟顶会审稿的 prompt
- 数据存放在 `data/`（已 gitignore），README.md 记录每次采集的配置和结果
- 所有 markdown 文档保持飞书兼容格式（无 jsonc、无 HTML 标签）
- `.claudeignore` 排除了 PDF/DOCX 等二进制文件，避免读入上下文
- 模型权重 (`.pt`) 通过 git-lfs 管理，克隆后需 `git lfs install && git lfs pull`
