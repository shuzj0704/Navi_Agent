# NaviAgent

跨环境长程导航（Cross-Environment Long-horizon Navigation）研究项目。

核心思路：统一导航 Token 架构——将 Agent 的工具交互、空间记忆、推理过程和动作输出统一到 VLM 的 embedding 空间中，端到端可微，2Hz 实时部署。详见 `docs/proposal/NaviAgent_v3.md`。

## 1. 项目结构

```
Navi_Agent/
├── docs/
│   ├── proposal/                  # 研究 proposal（v1→v2→v3 演进）
│   ├── AMAP/                      # AMAP CV Lab 导航论文解读
│   └── outdoor_paper/             # 室外导航相关论文解读
├── src/
│   ├── sim_vln_outdoor/           # Isaac Sim 室外仿真模块
│   │   ├── scripts/               # 入口脚本（场景加载、机器人控制、相机视角、闭环评估）
│   │   ├── nav/                   # 导航控制器接口（NavController ABC + 示例控制器）
│   │   ├── env/                   # 仿真环境封装（IsaacSimEnv）
│   │   ├── robot/                 # 机器人控制（Go2WRobot、Go2WPolicy）
│   │   ├── data/view/             # 快照拼接视频工具（make_video.py）
│   │   └── assets/
│   │       ├── policy/            # RL policy 权重 + 配置（rl_sar 格式）
│   │       └── rl_sar_zoo/        # 机器人 URDF/MJCF/mesh 描述包
│   └── vlm_serve/                 # VLM 部署模块（vLLM server + OpenAI client）
│       ├── server.py              # VLLMServerConfig + launch
│       ├── client.py              # VLMClient（chat / chat_stream_text / chat_with_image）
│       └── configs/               # YAML 配置（qwen3_5_9b、qwen3_vl_8b）
├── scripts/
│   ├── metaurban/                 # MetaUrban 数据采集脚本
│   ├── serve/                     # vLLM 服务启动 + 交互测试入口
│   ├── utils/                     # 文献整理等工具脚本
│   └── reference/                 # 参考脚本（外部依赖，不可直接运行）
├── data/                          # 仿真采集数据（gitignore）
│   └── urbanverse/
│       ├── load_scene_view/       # 相机视角快照
│       └── nav_eval/              # 闭环评估轨迹 + 帧图片
└── python.sh                      # Isaac Sim Python 解释器 wrapper
```

## 2. 仿真环境

项目使用两套仿真器，**不可混装**（渲染引擎冲突）：

| 环境名 | Python | 仿真器 | 用途 | 激活方式 |
|--------|--------|--------|------|---------|
| `habitat` | 3.9 | Habitat-Sim 0.3.3 | 室内导航 | `conda activate habitat` |
| `metaurban` | 3.10 | MetaUrban 0.0.1 (Panda3D) | 室外导航 | `conda activate metaurban` |

Isaac Sim 脚本须通过自带的 Python 运行，**不要在 conda 环境中运行**：

```bash
conda deactivate
./python.sh src/sim_vln_outdoor/scripts/<script>.py [args]
```

### 远程服务器（SSH）运行 Isaac Sim

通过 SSH 连接远程服务器运行 Isaac Sim 脚本时，需要额外准备：

**1. 安装 xvfb（虚拟帧缓冲区）**

Isaac Sim 的 GPU 渲染管线即使在 `--headless` 模式下也需要 X display。SSH 会话没有图形界面，需要 `xvfb-run` 提供虚拟显示：

```bash
# Ubuntu/Debian
sudo apt install xvfb
```

**2. 安装 Isaac Sim Python 缺失依赖**

Isaac Sim 自带的 Python 可能缺少部分依赖，需手动补装：

```bash
# 用 Isaac Sim 自带的 python.sh 安装（不是系统 pip）
./python.sh -m pip install typing_extensions filelock fsspec
```

**3. 运行命令**

所有 Isaac Sim 脚本前加 `xvfb-run -a`，用 `--gpu` 指定空闲 GPU（默认 0）：

```bash
# 远程 headless 运行（默认 GPU 0）
xvfb-run -a ./python.sh src/sim_vln_outdoor/scripts/nav_eval.py \
    --headless --save-frames --max-steps 170

# 指定 GPU 2 运行
xvfb-run -a ./python.sh src/sim_vln_outdoor/scripts/nav_eval.py \
    --headless --save-frames --max-steps 170 --gpu 2

# 其他脚本同理
xvfb-run -a ./python.sh src/sim_vln_outdoor/scripts/load_scene.py --headless --gpu 2
xvfb-run -a ./python.sh src/sim_vln_outdoor/scripts/load_scene_robot.py --headless --gpu 2
```

> 本地有图形界面时不需要 `xvfb-run`，直接运行即可。`--gpu` 默认为 0，单卡机器无需指定。

## 3. 快速开始

### 3.1 Isaac Sim — 场景加载与机器人仿真

```bash
# 仅加载 USD 场景
./python.sh src/sim_vln_outdoor/scripts/load_scene.py

# 加载场景 + 机器人 + RL locomotion policy
./python.sh src/sim_vln_outdoor/scripts/load_scene_robot.py

# 指定速度指令 / 出生点 / 无头模式
./python.sh src/sim_vln_outdoor/scripts/load_scene_robot.py \
    --cmd-vel 1.0 0.0 0.0 \
    --spawn-pos -730.0 490.0 0.0 \
    --headless
```

CraftBench 场景数据（12 个城市街道 USD 场景）需单独下载，详见[第 4 节](#4-外部资源下载)。可通过 `--usd-path` 指定场景路径，或设置 `CRAFTBENCH_ROOT` 环境变量。

### 3.2 Isaac Sim — D435i 相机视角查看

```bash
# 启用键盘控制（WASD 移动、方向键旋转、P 拍照）
./python.sh src/sim_vln_outdoor/scripts/load_scene_view.py \
    --keyboard --camera-pos -730.0 490.0 1.5

# TCP socket 控制（用于 agent 集成）
./python.sh src/sim_vln_outdoor/scripts/load_scene_view.py --socket-port 9090
```

### 3.3 Isaac Sim — 闭环导航控制器评估

```bash
# 直行 + headless + 保存每步帧图片（默认 ForwardOnly 控制器）
./python.sh src/sim_vln_outdoor/scripts/nav_eval.py \
    --headless --save-frames --max-steps 200

# 随机游走
./python.sh src/sim_vln_outdoor/scripts/nav_eval.py \
    --headless --save-frames --max-steps 200 \
    --controller "nav.demo_controllers:RandomWalkController"

# 自定义控制器，2Hz 控制频率
./python.sh src/sim_vln_outdoor/scripts/nav_eval.py \
    --headless --controller "my_module:MyNavController" \
    --controller-freq 2.0 --max-steps 100 --save-frames
```

自定义控制器继承 `NavController`，实现 `act()` 即可：

```python
from nav import NavController, Observation, Action

class MyNavController(NavController):
    def act(self, obs: Observation) -> Action:
        rgb = obs.rgb          # (480, 640, 3) uint8
        pose = obs.pose        # (x, y, z, roll, pitch, yaw)
        return Action(forward=0.5, yaw=2.0)
```

输出保存到 `data/urbanverse/nav_eval/<timestamp>/`（trajectory.jsonl + summary.json + 可选帧图片）。详见 `src/sim_vln_outdoor/README.md`。

### 3.4 MetaUrban — 室外轨迹采集

```bash
conda activate metaurban
python scripts/metaurban/single_trajectory.py
```

采集结果保存在 `data/metaurban_test/`。

### 3.5 VLM 部署 — vLLM 服务 + 交互测试

`src/vlm_serve/` 封装了 vLLM 服务启动和 OpenAI client 调用，供 Teacher Model 标注、推理评估、交互测试统一复用。所有参数走 YAML，CLI 可覆盖。

**启动服务**（在装了 vLLM 的环境中，例如 `lwy_swift`）：

```bash
conda activate lwy_swift

# 启动 Qwen3.5-9B（默认 GPU 1，端口 8003）
python scripts/serve/start_qwen35.py

# 启动 Qwen3-VL-8B-Instruct（默认 GPU 2，端口 8004）
python scripts/serve/start_qwen3vl.py

# 临时覆盖 GPU / 端口 / max_model_len
python scripts/serve/start_qwen35.py --gpu 0 --port 8013 --max-model-len 16384
```

默认配置在 [src/vlm_serve/configs/](src/vlm_serve/configs/)，模型路径等长期参数改 YAML，临时调整走 CLI。

**交互测试**（任意环境，仅需 `openai` 包）：

```bash
# 默认连 Qwen3.5（localhost:8003）
python scripts/serve/chat_test.py

# 连 Qwen3-VL
python scripts/serve/chat_test.py --base-url http://localhost:8004/v1 --model qwen3-vl
```

**在自己的代码中调用**（Teacher 标注、推理 eval 等场景）：

```python
from vlm_serve.client import VLMClient

# 文本对话
client = VLMClient(base_url="http://localhost:8003/v1", model="qwen3.5")
resp = client.chat(messages=[{"role": "user", "content": "你好"}])

# 单图视觉推理
vl_client = VLMClient(base_url="http://localhost:8004/v1", model="qwen3-vl")
reply = vl_client.chat_with_image(prompt="描述这张图", image_path="rgb_0.png")
```

> 新增需要调 VLM 的代码请直接复用 `VLMClient`，不要再写新的 OpenAI client 包装。

## 4. 外部资源下载

仿真模块依赖以下外部资源（已 gitignore，需手动下载）：

```bash
# 1. CraftBench 城市街道 USD 场景（12 个场景，来自 UrbanVerse）
#    下载地址：https://huggingface.co/datasets/Oatmealliu/UrbanVerse-CraftBench
#    默认存放路径：~/navigation/urban_verse/CraftBench/
#    可通过环境变量 CRAFTBENCH_ROOT 自定义路径，或运行时用 --usd-path 指定
huggingface-cli download Oatmealliu/UrbanVerse-CraftBench \
    --repo-type dataset --local-dir ~/navigation/urban_verse/CraftBench

# 2. RL policy 权重 + 配置
git clone https://github.com/fan-ziqi/rl_sar.git /tmp/rl_sar
cp -r /tmp/rl_sar/policy/* src/sim_vln_outdoor/assets/policy/

# 3. 机器人 URDF/MJCF/mesh 描述包
git clone https://github.com/fan-ziqi/rl_sar_zoo.git src/sim_vln_outdoor/assets/rl_sar_zoo/
```

如果 CraftBench 放在非默认路径，设置环境变量：

```bash
export CRAFTBENCH_ROOT=/your/custom/path/CraftBench
```

## 5. 扩展新机器人

1. 将机器人描述包放入 `src/sim_vln_outdoor/assets/rl_sar_zoo/<robot>_description/`
2. 将 policy 配置和权重放入 `src/sim_vln_outdoor/assets/policy/<robot>/`
3. 在 `src/sim_vln_outdoor/robot/` 下参照 `go2w.py` / `go2w_policy.py` 新建对应模块
4. `assets/` 中已包含 A1, B2, B2W, D1, G1, Go2, GR1T1, GR1T2, L4W4, Lite3, Tita 的 URDF 和 policy

## 6. CraftBench 可用场景

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
| scene_09 | CBD T 字路口，施工区域（默认场景） |
| scene_10 | CBD 十字路口，多样化障碍物 |
| scene_11 | 城市公园，遛狗和运动区 |
| scene_12 | 庭院广场，中央喷泉 |
