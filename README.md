# NaviAgent

跨环境长程导航研究项目。当前阶段：在 [UrbanVerse CraftBench](https://huggingface.co/datasets/Oatmealliu/UrbanVerse-CraftBench) 城市街道 USD 场景中，把 Qwen3-VL 当成"导航大脑"，跑通端到端 VLM 闭环导航。完整研究方案见 [docs/proposal/NaviAgent_v3.md](docs/proposal/NaviAgent_v3.md)。

---

## 1. 项目结构

```
Navi_Agent/
├── src/
│   ├── sim_vln_outdoor/              # Isaac Sim 仿真前端
│   │   ├── scripts/                  # 入口脚本（详见第 3 节）
│   │   │   ├── load_scene.py         # 仅加载 USD 场景
│   │   │   ├── load_scene_view.py    # D435i 视角渲染 + 键盘/Socket 手动控制
│   │   │   ├── load_scene_robot.py   # 场景 + Go2W + RL locomotion policy
│   │   │   ├── nav_eval.py           # 通用 NavController 闭环评估
│   │   │   └── vlm_gps_nav.py        # GPS 引导 VLM 闭环（主线入口）
│   │   ├── nav/                      # NavController 接口与实现
│   │   │   ├── controller.py         # NavController ABC + Observation / Action
│   │   │   ├── demo_controllers.py   # ForwardOnly / RandomWalk
│   │   │   ├── vlm_controller.py     # 基础 VLM 控制器（仅看 RGB，无目标）
│   │   │   └── gps_vlm_controller.py # GPS 引导 VLM 控制器（RGB + 文本 prompt）
│   │   ├── env/isaac_env.py          # IsaacSimEnv: SimulationApp 封装
│   │   ├── robot/                    # Go2WRobot / Go2WPolicy
│   │   └── assets/                   # USD / URDF / mesh / policy 权重 (gitignore)
│   └── vlm_serve/                    # VLM 后端：vLLM 启动 + OpenAI client
│       ├── server.py                 # VLLMServerConfig + launch
│       ├── client.py                 # VLMClient (chat / chat_with_image)
│       └── configs/                  # YAML 配置 (qwen3_5_9b / qwen3_vl_8b)
├── scripts/
│   ├── serve/                        # vLLM 启动入口
│   │   ├── start_qwen35.py
│   │   ├── start_qwen3vl.py
│   │   └── chat_test.py              # 通用交互测试客户端
│   ├── utils/
│   │   └── interpolate_trajectory.py # waypoint 稀疏 → 稠密插值
│   └── metaurban/                    # 周边：MetaUrban 数据采集（独立 conda 环境）
├── data/                             # 仿真数据 (整体 gitignore)
│   └── urbanverse/
│       ├── trajectory/scene_*/blog_point.txt  # 入库：人工标注的导航路线
│       └── vlm_gps_nav/<timestamp>/           # 主线输出
├── docs/                             # Proposal + 论文解读
├── python.sh                         # Isaac Sim Python wrapper
├── README.md
└── CLAUDE.md
```

---

## 2. 环境与依赖

### 2.1 系统级工具

| 工具 | 用途 | 安装 |
|---|---|---|
| `xvfb` | SSH headless 跑 Isaac Sim | `sudo apt install xvfb` |
| `ffmpeg` | 把 frames 拼成 mp4 | `sudo apt install ffmpeg` |
| Conda / Miniconda | 隔离 vLLM / metaurban 环境 | 自行安装 |

### 2.2 Conda 环境

主线只需要一个装了 vLLM 的环境（约定叫 `lwy_swift`，名字随意）：

```bash
conda create -n lwy_swift python=3.10 -y
conda activate lwy_swift
pip install vllm openai pyyaml
```

周边对照仿真器需要单独环境（**与主线无关，可跳过**）：

| 环境 | Python | 用途 |
|---|---|---|
| `metaurban` | 3.10 | MetaUrban 室外仿真（Panda3D） |
| `habitat` | 3.9 | Habitat-Sim 室内仿真（EGL/OpenGL，与 MetaUrban 渲染冲突） |

### 2.3 Isaac Sim

1. 装 Isaac Sim 5.1.0（NVIDIA 官方，~5GB），假设装在 `/home/<user>/nvidia/isaacsim_5.1.0`
2. 项目根的 [python.sh](python.sh) 是个 wrapper，默认走 `/home/shu22/nvidia/isaacsim_5.1.0`，路径不同时改 [python.sh](python.sh) 的 `ISAACSIM_ROOT` 或 export 环境变量：

```bash
export ISAACSIM_ROOT=/path/to/your/isaacsim_5.1.0
```

3. 给 Isaac Sim 自带 Python 装额外依赖（**用 `./python.sh -m pip`，不是系统 pip**）：

```bash
./python.sh -m pip install typing_extensions filelock fsspec
./python.sh -m pip install openai     # 主线必须，VLMClient 依赖
```

### 2.4 外部资源下载

```bash
# (1) CraftBench USD 场景（12 个城市街道）—— 主线必须
huggingface-cli download Oatmealliu/UrbanVerse-CraftBench \
    --repo-type dataset \
    --local-dir ~/navigation/urban_verse/CraftBench
# 默认路径 ~/navigation/urban_verse/CraftBench/，可用 CRAFTBENCH_ROOT 环境变量覆盖

# (2) Qwen3-VL-8B-Instruct 权重 —— 主线必须
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir /your/path/Qwen3-VL-8B-Instruct

# (3) Go2W RL policy 权重 + URDF —— 仅 load_scene_robot.py 需要
git clone https://github.com/fan-ziqi/rl_sar.git /tmp/rl_sar
cp -r /tmp/rl_sar/policy/* src/sim_vln_outdoor/assets/policy/
git clone https://github.com/fan-ziqi/rl_sar_zoo.git \
    src/sim_vln_outdoor/assets/rl_sar_zoo/
```

### 2.5 必须修改的硬编码路径

新机器拉下来后，下面这些写死的路径必须改：

| 文件 | 字段 | 默认值 | 改成 |
|---|---|---|---|
| [python.sh](python.sh):3 | `ISAACSIM_ROOT` | `/home/shu22/nvidia/isaacsim_5.1.0` | 你的 Isaac Sim 路径 |
| [src/vlm_serve/configs/qwen3_vl_8b.yaml](src/vlm_serve/configs/qwen3_vl_8b.yaml) | `model_path` | `/mnt/sda/szj/navi_dataset/checkpoints/Qwen3-VL-8B-Instruct/` | 你的 Qwen3-VL 权重路径 |
| [src/vlm_serve/configs/qwen3_vl_8b.yaml](src/vlm_serve/configs/qwen3_vl_8b.yaml) | `gpu` | `"2"` | 空闲 GPU 索引 |
| [src/vlm_serve/configs/qwen3_5_9b.yaml](src/vlm_serve/configs/qwen3_5_9b.yaml) | `model_path` / `gpu` | — | （仅 Teacher 标注用，主线非必须） |

CraftBench 路径用环境变量覆盖（不是改代码）：

```bash
export CRAFTBENCH_ROOT=/your/custom/path/CraftBench
```

CLI 也支持临时覆盖 vLLM 配置：

```bash
python scripts/serve/start_qwen3vl.py --model-path /your/path --gpu 0 --port 8014
```

### 2.6 重新生成稠密导航轨迹

仓库**入库**了人工标注的稀疏 waypoint 文件 [data/urbanverse/trajectory/scene_09/blog_point.txt](data/urbanverse/trajectory/scene_09/blog_point.txt)，由它生成的 `dense_trajectory.json` / `.png` **不入库**。新机器第一次跑主线之前必须生成一次：

```bash
python scripts/utils/interpolate_trajectory.py \
    --input data/urbanverse/trajectory/scene_09/blog_point.txt \
    --step 0.5 --visualize
```

---

## 3. sim_vln_outdoor/scripts 测试脚本

所有脚本必须通过 Isaac Sim 自带的 Python 解释器运行（`./python.sh`），**不能在 conda 环境中运行**。SSH 远程时所有命令前面加 `xvfb-run -a`。

### 3.1 load_scene.py — 仅加载 USD 场景

最小 smoke test，验证 Isaac Sim 能起来 + USD 能加载，不放任何机器人/相机。

```bash
xvfb-run -a ./python.sh src/sim_vln_outdoor/scripts/load_scene.py --headless --gpu 0
# 期望看到 "[Info] Scene loaded successfully"
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--usd-path` | scene_09 | USD 场景路径 |
| `--headless` | False | 无头运行（SSH 必须） |
| `--gpu` | 0 | 渲染 GPU |

### 3.2 load_scene_view.py — D435i 视角 + 手动控制

加载场景 + D435i 相机，**不经过任何 controller**，用键盘或 TCP socket 直接控制相机视角。主要用于：
- **给新场景标定 waypoint 坐标**（按 `P` 拍照，终端打印当前 `pos`）
- 调试 USD 场景渲染问题
- 给外部 agent 集成做 socket 接口测试

```bash
# 键盘控制：WASD 移动、QE 升降、方向键旋转、+/- 改速、P 拍照
./python.sh src/sim_vln_outdoor/scripts/load_scene_view.py \
    --keyboard --camera-pos -730.0 490.0 1.5

# TCP socket 控制（外部 agent 用）
./python.sh src/sim_vln_outdoor/scripts/load_scene_view.py --socket-port 9090
# 另一个进程发：echo '{"x":-730,"y":490,"z":1.5,"yaw":90}' | nc localhost 9090

# 保存帧到目录
./python.sh src/sim_vln_outdoor/scripts/load_scene_view.py --save-dir /tmp/frames
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--camera-pos` | scene_09 默认 | 初始相机位置 [x y z] |
| `--keyboard` | False | 启用键盘控制 |
| `--socket-port` | None | 启用 socket 控制（端口号） |
| `--save-dir` | None | 帧保存目录 |
| `--headless` | False | 无头运行 |

### 3.3 load_scene_robot.py — 场景 + Go2W + RL locomotion

加载场景 + Go2W 四足机器人 + RL locomotion policy，验证机器人能在 USD 场景里走路。**当前主线用 D435i 相机视角作为 stand-in**，Go2W 集成是 future work。

```bash
xvfb-run -a ./python.sh src/sim_vln_outdoor/scripts/load_scene_robot.py \
    --headless --spawn-pos -730.0 490.0 0.0 --cmd-vel 1.0 0.0 0.0
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--spawn-pos` | `-730 490 0` | 机器人初始位置 [x y z] |
| `--cmd-vel` | `0.5 0 0` | 速度指令 [vx vy vyaw] |
| `--headless` | False | 无头运行 |

> 需要先下载 RL policy 权重（见 2.4 节第 (3) 项）。

### 3.4 nav_eval.py — 通用 NavController 闭环评估

通用版闭环评估：渲染 D435i 图像 → 控制器输出动作 → 更新相机位姿 → 循环。可加载任意 NavController（包括自定义的）。`vlm_gps_nav.py` 实际上就是它的 GPS 特化版。

```bash
# 默认 ForwardOnly 控制器，保存帧
xvfb-run -a ./python.sh src/sim_vln_outdoor/scripts/nav_eval.py \
    --headless --save-frames --max-steps 200

# 随机游走
xvfb-run -a ./python.sh src/sim_vln_outdoor/scripts/nav_eval.py \
    --headless --max-steps 100 \
    --controller "nav.demo_controllers:RandomWalkController"

# 基础 VLM 控制器（仅看图，无目标感知；先要起 vLLM 服务）
xvfb-run -a ./python.sh src/sim_vln_outdoor/scripts/nav_eval.py \
    --headless --save-frames --max-steps 100 --controller-freq 1.0 \
    --controller "nav.vlm_controller:VLMNavController" \
    --controller-kwargs '{"instruction":"explore the construction site"}'
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--controller` | `nav.demo_controllers:ForwardOnlyController` | `module:Class` 形式的控制器 |
| `--controller-kwargs` | `{}` | JSON 字符串，传给控制器 `__init__` |
| `--controller-freq` | 2.0 | 控制器频率 (Hz) |
| `--camera-pos` | scene_09 默认 | 初始相机位置 |
| `--max-steps` | 500 | 步数上限 |
| `--save-frames` | False | 保存每步 RGB 帧 |
| `--headless` | False | 无头运行 |

输出在 `data/urbanverse/nav_eval/<timestamp>/`：`trajectory.jsonl` + `summary.json` + 可选 `frames/`。

**自定义 controller** 只需继承 `NavController`，实现 `act()`：

```python
from nav import NavController, Observation, Action

class MyController(NavController):
    def act(self, obs: Observation) -> Action:
        rgb = obs.rgb       # (480, 640, 3) uint8
        pose = obs.pose     # (x, y, z, roll, pitch, yaw)
        return Action(forward=0.5, yaw=2.0)
```

### 3.5 vlm_gps_nav.py — GPS 引导 VLM 闭环（主线入口）

详见第 4 节。

---

## 4. UrbanVerse + VLM 闭环完整流程

主线目标：让 Qwen3-VL 在 CraftBench 城市场景里**跟着预定义的 GPS 路径**走到终点。每步 VLM 收到 FPV 图像 + 结构化文本 prompt（当前 pose / 进度 / 前 5 个 lookahead 点的 ego 坐标 / next turn 提示），输出 `FORWARD / TURN_LEFT / TURN_RIGHT / STOP` 之一。

### 4.1 准备稠密轨迹（一次性）

人工标注的稀疏 waypoint 文件 [blog_point.txt](data/urbanverse/trajectory/scene_09/blog_point.txt) 是 position-only 格式，跟真实导航 APP 的 polyline 一致：

```
# format: [point N] x y z    (z is the camera height in meters)
[point 1] -693.51  496.81  2.0
[point 2] -635.51  496.81  2.0
[point 3] -635.51  485.31  2.0
[point 4] -629.32  485.22  2.0
```

线性插值成 0.5m 间距的稠密路径：

```bash
python scripts/utils/interpolate_trajectory.py \
    --input data/urbanverse/trajectory/scene_09/blog_point.txt \
    --step 0.5 --visualize
# -> dense_trajectory.json (153 个稠密点) + dense_trajectory.png (俯视图)
```

机器人初始 yaw 由稠密路径**第一段方向**（`atan2`）推出来，模拟"被放下时已经面朝路线"。

> **要标注新路线** 时：用 [load_scene_view.py](src/sim_vln_outdoor/scripts/load_scene_view.py) 开 `--keyboard` 飞到合适位置按 `P` 拍照（终端打印 `pos`），把 4-N 个关键转折点写入 `data/urbanverse/trajectory/<scene>/blog_point.txt`，再跑 `interpolate_trajectory.py`。

### 4.2 启动 vLLM 服务（另开一个终端）

```bash
conda activate lwy_swift
python scripts/serve/start_qwen3vl.py
# 默认 GPU 2，端口 8004，等到看到 "Uvicorn running on http://0.0.0.0:8004"

# 临时换 GPU/端口
python scripts/serve/start_qwen3vl.py --gpu 0 --port 8014
```

**连通性测试**（任意 conda 环境，仅需 `openai` 包）：

```bash
python scripts/serve/chat_test.py --base-url http://localhost:8004/v1 --model qwen3-vl
# 输入 "hi"，应能收到 Qwen3-VL 回复
```

### 4.3 跑 GPS 引导闭环

```bash
xvfb-run -a ./python.sh src/sim_vln_outdoor/scripts/vlm_gps_nav.py \
    --headless --max-steps 200 --controller-freq 1.0 --gpu 0 \
    --trajectory data/urbanverse/trajectory/scene_09/dense_trajectory.json \
    --instruction "Walk along the sidewalk to the destination, avoid the construction barriers"
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--trajectory` | （必填） | `dense_trajectory.json` 路径 |
| `--instruction` | "Follow the navigation route to the destination." | 任务自然语言指令 |
| `--max-steps` | 200 | 控制器步数上限 |
| `--controller-freq` | 1.0 | 控制器频率 (Hz)，建议 0.5-1.0（VLM 单次推理 1-2s） |
| `--lookahead` | 5 | 前瞻点数（5 × 0.5m = 2.5m 路径预览） |
| `--goal-tol` | 2.0 | 距终点 < 该值判定成功（米） |
| `--forward-step` | 0.5 | 单次 FORWARD 移动距离（米） |
| `--yaw-step` | 15.0 | 单次 TURN 旋转角度（度） |
| `--start-yaw` | None | 强制覆盖初始 yaw；默认从路径第一段几何推算 |
| `--base-url` | `http://localhost:8004/v1` | vLLM 服务地址 |
| `--model` | `qwen3-vl` | 服务端模型名 |
| `--gpu` | 0 | Isaac Sim 渲染 GPU |
| `--usd-path` | scene_09 默认 | USD 场景路径 |
| `--headless` | False | 无头运行（SSH 必须） |

### 4.4 输出结构

每次运行产物在 `data/urbanverse/vlm_gps_nav/<timestamp>/` 下：

```
20260411_153012/
├── frames/
│   ├── frame_000000.png   # FPV 帧（每个控制步一张）
│   └── ...
├── vlm_io.jsonl           # 每行 = 一步 VLM 决策的完整审计
│                          # {step, pose, progress_idx, prompt, reply, action}
├── trajectory.jsonl       # 每行 = 一步实际 pose + 动作 + 距离终点
├── summary.json           # success / total_steps / final_dist_to_goal_m / route_length_m
└── nav.mp4                # 30fps 第一视角运动视频（ffmpeg 自动拼接）
```

`vlm_io.jsonl` 的 `prompt` 字段是给 VLM 的**完整文本**，可以直接 grep/重放调试 VLM 决策质量。

### 4.5 在自己的代码中调用 VLM

新增需要调 VLM 的代码请直接复用 `VLMClient`，不要再写新的 OpenAI client 包装：

```python
from vlm_serve.client import VLMClient

client = VLMClient(base_url="http://localhost:8004/v1", model="qwen3-vl")
reply = client.chat_with_image(prompt="describe this image", image_path="rgb_0.png")
```

---

## 5. CraftBench 可用场景

| ID | 场景描述 |
|---|---|
| scene_01 | 住宅街道，双侧人行道，密集路桩和垃圾袋 |
| scene_02 | 住宅街道，单侧人行道，沿街建筑 |
| scene_03 | 住宅街道，双侧人行道，整洁环境 |
| scene_04 | 商业街道，不对称人行道，长椅和自行车 |
| scene_05 | 日式旅游街道，双侧人行道，车辆阻挡 |
| scene_06 | 酒吧街道，双侧人行道，倒落的滑板车 |
| scene_07 | 窄巷，户外餐饮区 |
| scene_08 | 商业街道，自行车道，双侧人行道 |
| scene_09 | CBD T 字路口，施工区域（**主线默认场景**） |
| scene_10 | CBD 十字路口，多样化障碍物 |
| scene_11 | 城市公园，遛狗和运动区 |
| scene_12 | 庭院广场，中央喷泉 |
