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
│   └── sim_vln_outdoor/           # Isaac Sim 室外仿真模块
│       ├── scripts/               # 入口脚本（场景加载、机器人控制、相机视角、闭环评估）
│       ├── nav/                   # 导航控制器接口（NavController ABC + 示例控制器）
│       ├── env/                   # 仿真环境封装（IsaacSimEnv）
│       ├── robot/                 # 机器人控制（Go2WRobot、Go2WPolicy）
│       ├── data/view/             # 快照拼接视频工具（make_video.py）
│       └── assets/
│           ├── policy/            # RL policy 权重 + 配置（rl_sar 格式）
│           └── rl_sar_zoo/        # 机器人 URDF/MJCF/mesh 描述包
├── scripts/
│   ├── metaurban/                 # MetaUrban 数据采集脚本
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

CraftBench 场景数据（12 个城市街道 USD 场景）位于 `~/navigation/urban_verse/CraftBench/`，可通过 `--usd-path` 指定。

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

## 4. 扩展新机器人

1. 将机器人描述包放入 `src/sim_vln_outdoor/assets/rl_sar_zoo/<robot>_description/`
2. 将 policy 配置和权重放入 `src/sim_vln_outdoor/assets/policy/<robot>/`
3. 在 `src/sim_vln_outdoor/robot/` 下参照 `go2w.py` / `go2w_policy.py` 新建对应模块
4. `assets/` 中已包含 A1, B2, B2W, D1, G1, Go2, GR1T1, GR1T2, L4W4, Lite3, Tita 的 URDF 和 policy

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
| scene_09 | CBD T 字路口，施工区域（默认场景） |
| scene_10 | CBD 十字路口，多样化障碍物 |
| scene_11 | 城市公园，遛狗和运动区 |
| scene_12 | 庭院广场，中央喷泉 |
