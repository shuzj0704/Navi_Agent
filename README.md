# NaviAgent

跨环境长程导航（CELN）研究项目。当前阶段：proposal 规划 + 仿真环境搭建。完整研究方案见 [docs/proposal/NaviAgent_v3.md](docs/proposal/NaviAgent_v3.md)。

---

## 1. 项目结构

```
Navi_Agent/
├── src/
│   ├── naviagent/                        # 导航 Agent 核心（与仿真器完全解耦）
│   │   ├── common/                       # 共享工具
│   │   │   ├── coordinate_transform.py   # Habitat 坐标 ↔ 导航坐标系转换
│   │   │   ├── nav_state.py              # NavState 导航状态机
│   │   │   ├── view_constants.py         # 视角常量 (VIEW_ORDER, VIEW_LABELS 等)
│   │   │   └── visualizer.py             # 可视化面板 (draw_debug_frame, build_panel_info)
│   │   ├── perception/                   # 感知层
│   │   │   ├── pixel_to_3d.py            # 深度图 → 3D 点云 + 相机内参
│   │   │   ├── obs_reader.py             # ObsBundle + HabitatObsReader + SimClientObsReader
│   │   │   ├── yoloe_segmentor.py        # YOLOE 开放词汇分割
│   │   │   └── semantic_mapper.py        # 语义地图 (俯视图渲染)
│   │   ├── decision/                     # 决策层
│   │   │   ├── nav_engine.py             # NavigationEngine 导航主循环引擎
│   │   │   ├── dwa_planner.py            # DWA 局部路径规划
│   │   │   ├── turn_controller.py        # 转向控制器
│   │   │   └── orchestrator.py           # TaskOrchestrator 双系统任务编排
│   │   └── vlm/                          # VLM 模块
│   │       ├── vlm_navigator.py          # System1 VLM 导航 (Qwen3-VL 4 视角预测)
│   │       └── planner.py                # System2 规划器
│   │
│   ├── scripts/                          # 室内导航入口脚本
│   │   ├── nav_main.py                   # 单次导航主循环 (HTTP 模式)
│   │   └── batch_eval.py                 # 批量评测 (HTTP 模式)
│   │
│   ├── sim_vln_indoor/                   # 室内仿真 (Habitat, 独立进程)
│   │   ├── env/
│   │   │   ├── server/                   # FastAPI 仿真服务器
│   │   │   │   ├── __main__.py           # 入口: python -m sim_vln_indoor.env.server
│   │   │   │   ├── app.py                # FastAPI 路由 (/obs, /action, /scene 等)
│   │   │   │   ├── habitat_backend.py    # Habitat-Sim 线程安全封装
│   │   │   │   ├── encoding.py           # RGB JPEG / Depth float32 编码
│   │   │   │   └── config.py             # YAML 配置加载
│   │   │   ├── client/                   # HTTP 客户端
│   │   │   │   └── client.py             # SimClient (httpx) + AgentState
│   │   │   ├── config/
│   │   │   │   └── sim_server.yaml       # 仿真服务器配置
│   │   │   └── habitat_utils.py          # Habitat 工具函数
│   │   └── scripts/                      # 仿真侧独立测试脚本
│   │       ├── habitat_render.py         # 渲染测试
│   │       ├── habitat_viewer.py         # 场景浏览器
│   │       └── test_yoloe.py             # YOLOE 分割测试
│   │
│   ├── sim_vln_outdoor/                  # 室外仿真 (Isaac Sim)
│   │   ├── scripts/                      # 入口脚本
│   │   │   ├── load_scene.py             # 仅加载 USD 场景
│   │   │   ├── load_scene_view.py        # D435i 视角渲染 + 键盘/Socket 控制
│   │   │   ├── load_scene_robot.py       # 场景 + Go2W + RL locomotion
│   │   │   ├── nav_eval.py               # 通用 NavController 闭环评估
│   │   │   └── vlm_gps_nav.py            # GPS 引导 VLM 闭环导航
│   │   ├── nav/                          # NavController 接口与实现
│   │   ├── env/isaac_env.py              # IsaacSimEnv: SimulationApp 封装
│   │   └── robot/                        # Go2WRobot / Go2WPolicy
│   │
│   └── vlm_serve/                        # VLM 后端：vLLM 启动 + OpenAI client
│       ├── server.py                     # VLLMServerConfig + launch
│       ├── client.py                     # VLMClient (chat / chat_with_image)
│       └── configs/                      # YAML 配置 (qwen3_5_9b / qwen3_vl_8b)
│
├── models/                               # 模型权重 (git-lfs 管理)
│   └── yoloe-11l-seg.pt                  # YOLOE 分割模型 (68MB)
├── scripts/                              # 全局工具脚本
│   ├── serve/                            # vLLM 启动入口
│   ├── utils/                            # 轨迹插值等工具
│   └── metaurban/                        # MetaUrban 数据采集
├── data/                                 # 数据目录 (gitignore, LFS 模型除外)
├── docs/                                 # Proposal + 论文解读
├── python.sh                             # Isaac Sim Python wrapper
├── .gitattributes                        # git-lfs 配置 (*.pt)
└── README.md
```

### 架构分层

```
naviagent (任意 Python 环境, 无 habitat 依赖)
  ├── perception: 观测读取 + 点云 + 语义分割/建图
  ├── decision:   导航引擎 + DWA + 转向控制 + 任务编排
  ├── vlm:        VLM 导航预测 + System2 规划
  └── common:     坐标变换 + 状态机 + 可视化

sim_vln_indoor (conda activate habitat, Python 3.9)
  └── HTTP Server (FastAPI) ← naviagent 通过 SimClient 调用

sim_vln_outdoor (Isaac Sim 自带 Python)
  └── Isaac Sim 渲染 + NavController 接口
```

所有模块单向依赖，无循环引用。naviagent 与仿真器之间**完全通过 HTTP 通信**。

---

## 2. 环境与依赖

### 2.1 Conda 环境

| 环境名 | Python | 用途 | 关键依赖 |
|--------|--------|------|----------|
| `habitat` | 3.9 | 室内仿真服务器 (Habitat-Sim 0.3.3) | habitat-sim, fastapi, uvicorn, numpy, opencv |
| `metaurban` | 3.10 | 室外仿真 (MetaUrban, 非主线) | metaurban, stable-baselines3, pytorch |

> `habitat` 和 `metaurban` 渲染引擎冲突（EGL vs Panda3D），不可混装。

### 2.2 导航 Agent 依赖

naviagent 运行在**任意 Python 环境**（不需要 habitat），需要：

```
numpy, opencv-python, httpx, openai, ultralytics
```

### 2.3 模型权重 (git-lfs)

模型文件通过 git-lfs 管理。克隆后需要拉取：

```bash
git lfs install
git lfs pull
```

如果未拉取 LFS 文件，运行时会提示：
```
FileNotFoundError: YOLOE 模型文件不存在: Navi_Agent/models/yoloe-11l-seg.pt
如果是从 Git 仓库克隆的项目，请先拉取 LFS 文件:
  git lfs install && git lfs pull
```

---

## 3. 室内导航 (Habitat + HTTP)

室内导航采用**双进程架构**：仿真服务器（Habitat）和导航 Agent 分别运行，通过 HTTP 通信。

### 3.1 启动仿真服务器

```bash
conda activate habitat
cd Navi_Agent/src
python -m sim_vln_indoor.env.server
# 默认监听 0.0.0.0:5100, 可选: --port 5200 --config <path>
```

验证：
```bash
curl http://localhost:5100/health        # 健康检查
curl http://localhost:5100/scenes        # 可用场景列表
```

HTTP API：

| 方法 | 路径 | 用途 |
|------|------|------|
| GET | /health | 健康检查 |
| GET | /scenes | 可用场景列表 |
| POST | /scene | 加载场景 |
| GET | /obs | 获取全部传感器观测 + agent 状态 |
| POST | /action | 执行单个动作 |
| POST | /actions | 执行多个动作 |
| GET | /agent | 获取 agent 状态 |
| PUT | /agent | 设置 agent 位姿 |

### 3.2 启动导航 Agent

在**任意 Python 环境**中运行（不需要 habitat）：

```bash
cd /home/nuc/vln

# 单次导航
python Navi_Agent/src/scripts/nav_main.py \
  --sim-url http://localhost:5100 \
  --scene 17DRP5sb8fy \
  --steps 100 \
  --save-vis

# Mock 模式（不调 VLM, 测试仿真连通性）
python Navi_Agent/src/scripts/nav_main.py --mock --scene 17DRP5sb8fy

# 批量评测
python Navi_Agent/src/scripts/batch_eval.py \
  --sim-url http://localhost:5100 \
  --split val_seen \
  --max-episodes 5 \
  --steps 100 \
  --save-vis
```

### 3.3 nav_main.py 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--sim-url` | `http://localhost:5100` | 仿真服务器地址 |
| `--scene` | None | 场景名 (e.g. `17DRP5sb8fy`) |
| `--scene-idx` | 0 | 场景编号 (`--scene` 优先) |
| `--steps` | 100 | 导航步数上限 |
| `--instruction` | 默认探索指令 | 导航任务自然语言指令 |
| `--mock` | False | 不调 VLM, 使用 mock 预测 |
| `--save-vis` | False | 保存可视化视频 |
| `--no-planner` | False | 禁用 System2 任务规划器 |
| `--plan-heartbeat` | 15 | 规划器心跳步数 |
| `--no-thinking` | False | 禁用 VLM thinking |

### 3.4 batch_eval.py 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--sim-url` | `http://localhost:5100` | 仿真服务器地址 |
| `--split` | `val_seen` | 数据集分割 (train/val_seen/val_unseen/test) |
| `--max-episodes` | None | 最大 episode 数 |
| `--episode-ids` | None | 指定 episode ID (逗号分隔) |
| `--steps` | 100 | 每个 episode 步数上限 |
| `--mock` | False | 不调 VLM |
| `--save-vis` | False | 保存可视化视频 |
| `--output-dir` | `/home/nuc/vln/output/eval` | 评测结果输出目录 |
| `--success-threshold` | 3.0 | 成功判定距离阈值 (米) |

评测输出：`results.csv` (逐 episode 指标) + `summary.json` (汇总 SR/SPL/Avg Dist)。

### 3.5 部署到其他机器

仿真服务器可独立部署，只需：

1. 拷贝 `src/sim_vln_indoor/` (~50KB) + 场景数据 `data/scene_data/mp3d/` (~21GB)
2. 创建 habitat conda 环境 (habitat-sim 0.3.3 + fastapi + uvicorn)
3. 修改 `sim_vln_indoor/env/config/sim_server.yaml` 中的 `base_dir` 路径
4. 启动服务: `python -m sim_vln_indoor.env.server`

导航 Agent 通过 `--sim-url http://<remote-ip>:5100` 连接远程仿真。

### 3.6 VLM 服务

室内导航的 VLM 推理默认连接 `http://192.168.1.137:8000/v1`（代码中 `VLM_API_URL` 常量）。VLM 服务不在线时使用 `--mock` 跳过。

---

## 4. 室外导航 (Isaac Sim + VLM)

### 4.1 概述

在 [UrbanVerse CraftBench](https://huggingface.co/datasets/Oatmealliu/UrbanVerse-CraftBench) 城市街道 USD 场景中，用 Qwen3-VL 作为导航大脑，跑端到端 VLM 闭环导航。

所有脚本必须通过 Isaac Sim 自带的 Python 运行 (`./python.sh`)，**不能在 conda 环境中运行**。SSH 远程时命令前加 `xvfb-run -a`。

### 4.2 脚本列表

| 脚本 | 用途 |
|------|------|
| `load_scene.py` | 最小 smoke test，验证 Isaac Sim + USD 加载 |
| `load_scene_view.py` | D435i 视角渲染 + 键盘/Socket 手动控制 |
| `load_scene_robot.py` | 场景 + Go2W + RL locomotion policy |
| `nav_eval.py` | 通用 NavController 闭环评估 |
| `vlm_gps_nav.py` | GPS 引导 VLM 闭环导航 (主线入口) |

### 4.3 GPS 引导 VLM 闭环 (vlm_gps_nav.py)

```bash
# 1. 生成稠密轨迹
python scripts/utils/interpolate_trajectory.py \
    --input data/urbanverse/trajectory/scene_09/blog_point.txt \
    --step 0.5 --visualize

# 2. 启动 vLLM (另一终端)
ssh ps2
conda activate lwy_swift
python scripts/serve/start_qwen3vl.py

# 3. 跑闭环
xvfb-run -a ./python.sh src/sim_vln_outdoor/scripts/vlm_gps_nav.py \
    --headless --max-steps 200 --controller-freq 1.0 \
    --trajectory data/urbanverse/trajectory/scene_09/dense_trajectory.json
```

输出在 `data/urbanverse/vlm_gps_nav/<timestamp>/`：`frames/` + `vlm_io.jsonl` + `trajectory.jsonl` + `summary.json` + `nav.mp4`。

### 4.4 Isaac Sim 环境配置

详见 [CLAUDE.md](CLAUDE.md) 中 Isaac Sim 相关章节。关键点：

- Isaac Sim 5.1.0，通过 `python.sh` wrapper 运行
- 新机器需设置: `export ISAACSIM_ROOT=/path/to/isaacsim`
- VLM 控制器需: `./python.sh -m pip install openai`

---

## 5. VLM 服务 (vlm_serve)

封装 vLLM 服务启动 + OpenAI client 调用。新增 VLM 调用请复用 `VLMClient`。

```bash
# 启动 Qwen3-VL-8B 服务 (需 lwy_swift 环境)
python scripts/serve/start_qwen3vl.py

# 交互测试
python scripts/serve/chat_test.py --base-url http://localhost:8004/v1 --model qwen3-vl
```

```python
from vlm_serve.client import VLMClient
client = VLMClient(base_url="http://localhost:8004/v1", model="qwen3-vl")
reply = client.chat_with_image(prompt="describe this image", image_path="rgb.png")
```

---

## 6. 数据资源

| 资源 | 路径 | 说明 |
|------|------|------|
| MP3D 室内场景 | `data/scene_data/mp3d/` | Habitat 室内仿真 (~21GB) |
| R2R VLN-CE 数据集 | `data/vln_ce/R2R_VLNCE_v1-3/` | 室内导航评测数据 |
| CraftBench USD 场景 | 见 CLAUDE.md | 室外城市街道 (12 场景) |
| YOLOE 模型 | `models/yoloe-11l-seg.pt` | git-lfs 管理 (68MB) |

---

## 7. 硬编码路径

新机器部署时需要修改的路径：

| 文件 | 字段 | 说明 |
|------|------|------|
| `src/sim_vln_indoor/env/config/sim_server.yaml` | `scenes.base_dir` | MP3D 场景数据路径 |
| `src/scripts/batch_eval.py` | `DATASET_DIR`, `SCENE_DIR` | VLN-CE 数据集和场景路径 |
| `src/scripts/nav_main.py` | `VLM_API_URL` | VLM 服务地址 |
| `src/scripts/batch_eval.py` | `VLM_API_URL` | VLM 服务地址 |
| `python.sh` | `ISAACSIM_ROOT` | Isaac Sim 安装路径 |
| `src/vlm_serve/configs/*.yaml` | `model_path`, `gpu` | VLM 模型路径和 GPU |
