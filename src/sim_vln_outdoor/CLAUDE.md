# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Isaac Sim 室外仿真模块，用于在 [UrbanVerse CraftBench](https://huggingface.co/datasets/Oatmealliu/UrbanVerse-CraftBench) USD 城市场景中加载机器人并运行 RL locomotion policy。代码从 [UrbanVerse](https://github.com/metadriverse/urban-verse) 移植并重构为模块化架构。

## Module Structure

```
sim_vln_outdoor/
├── scripts/                          # 入口脚本
│   ├── load_scene.py                 # 仅加载 USD 场景
│   ├── load_scene_robot.py           # 场景 + 机器人 + policy 完整 pipeline
│   └── load_scene_view.py            # D435i 相机视角渲染 + 键盘/Socket 控制
├── env/                              # 仿真环境
│   └── isaac_env.py                  # IsaacSimEnv 类
├── robot/                            # 机器人控制
│   ├── go2w.py                       # Go2WRobot 类
│   └── go2w_policy.py                # Go2WPolicy 类 + quat_rotate_inverse
└── assets/
    ├── policy/<robot>/               # RL policy 权重 + 配置
    │   ├── base.yaml                 # 共享配置（dt, joint_names, default_dof_pos）
    │   └── <framework>/config.yaml   # Policy 特定参数（obs, scales, PD gains）
    └── rl_sar_zoo/<robot>_description/  # URDF/MJCF/mesh，来自 rl_sar 项目
```

## Running Scripts

所有脚本必须通过 Isaac Sim 自带的 Python 解释器运行（不可在 conda 环境中）：

```bash
# 从项目根目录
conda deactivate
./python.sh src/sim_vln_outdoor/scripts/load_scene.py
./python.sh src/sim_vln_outdoor/scripts/load_scene_robot.py

# 或直接用 Isaac Sim Python
cd /home/shu22/nvidia/isaacsim_5.1.0
./python.sh /home/shu22/navigation/Navi_Agent/src/sim_vln_outdoor/scripts/load_scene_robot.py

# 常用参数
./python.sh .../load_scene_robot.py --headless
./python.sh .../load_scene_robot.py --cmd-vel 1.0 0.0 0.0
./python.sh .../load_scene_robot.py --spawn-pos -730.0 490.0 0.0
./python.sh .../load_scene_robot.py --usd-path /path/to/scene.usd
```

## Architecture

### Import 顺序约束

Isaac Sim 要求 `SimulationApp` 在所有 `omni.*` import 之前创建。模块设计遵循此约束：

1. 入口脚本先 `from env import IsaacSimEnv` 并实例化（内部创建 `SimulationApp`）
2. 然后才 `from robot import Go2WRobot, Go2WPolicy`（内部延迟 import omni 模块）

脚本通过 `sys.path.insert(0, sim_vln_outdoor/)` 使 `from env/robot import ...` 可用。

### 核心类

**`IsaacSimEnv`** (`env/isaac_env.py`)
- 封装 SimulationApp 生命周期、USD 场景加载、World 创建
- 提供 `step()` / `is_running` / `close()` / `stage` 接口

**`Go2WPolicy`** (`robot/go2w_policy.py`)
- 纯计算，不依赖 Isaac Sim，可独立测试
- 从 rl_sar 格式 YAML 加载配置，运行 TorchScript 推理
- 57 维 obs: `[ang_vel(3), gravity_vec(3), commands(3), dof_pos(16), dof_vel(16), actions(16)]`
- PD torque 计算：leg joints (0-11) 位置控制，wheel joints (12-15) 速度控制

**`Go2WRobot`** (`robot/go2w.py`)
- 持有 `IsaacSimEnv` 和 `Go2WPolicy` 引用
- URDF 导入、articulation 初始化、关节映射（policy order ↔ sim DOF order）
- `get_state()` 返回 policy 顺序的关节状态，`apply_torques()` 映射回 sim 顺序

### 数据流

```
env.step() → robot.get_state() → policy.compute_observation()
→ policy.infer() → policy.compute_torques() → robot.apply_torques()
```

## External Data Dependencies

- **CraftBench 场景（USD）**: `~/navigation/urban_verse/CraftBench/`（12 个城市街道场景）
- 默认使用 `scene_09_cbd_t_intersection_construction_sites`
- 路径通过 `--usd-path` 参数覆盖

## Conventions

- YAML 配置使用 rl_sar 格式：顶层 key 为 `<robot_name>`（base.yaml）或 `<robot_name>/<framework>`（policy config.yaml）
- Quaternion 约定：(w, x, y, z)
- 物理时间步：dt=0.005s（200Hz physics），decimation=4（50Hz policy）
- 关节驱动设为 effort mode（kp=0, kd=0），torque 由 policy 的 PD 控制器计算

## Adding a New Robot

1. 放置 URDF/mesh 到 `assets/rl_sar_zoo/<robot>_description/`
2. 放置 policy 配置和权重到 `assets/policy/<robot>/`
3. 在 `robot/` 下新建 `<robot>.py`（参照 `go2w.py`，主要改 PRIM_PATH 和 URDF 路径）
4. 在 `robot/` 下新建 `<robot>_policy.py`（参照 `go2w_policy.py`，主要改 config key 和 obs 构建）
5. 在 `robot/__init__.py` 中注册导出

当前已有 URDF 和 policy 的机器人：A1, B2, B2W, D1, G1, Go2, Go2W, GR1T1, GR1T2, L4W4, Lite3, Tita。其中 Go2W 已完成完整集成。
