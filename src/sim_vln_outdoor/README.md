# sim_vln_outdoor

Isaac Sim 室外仿真模块，用于在 UrbanVerse CraftBench USD 城市场景中加载机器人、运行 RL locomotion policy、模拟 D435i 相机视角，以及闭环评估导航控制器。

## 1. 环境要求

- Isaac Sim 5.1.0（路径：`/home/shu22/nvidia/isaacsim_5.1.0`）
- 不使用 conda 环境，通过 Isaac Sim 自带 Python 运行
- CraftBench 场景数据：`~/navigation/urban_verse/CraftBench/`

## 2. 运行方式

所有脚本必须通过 Isaac Sim 的 Python 解释器运行。项目根目录提供了 wrapper：

```bash
cd ~/navigation/Navi_Agent

# 方式一：使用项目 wrapper（推荐）
./python.sh src/sim_vln_outdoor/scripts/<script>.py [args]

# 方式二：直接使用 Isaac Sim Python
cd /home/shu22/nvidia/isaacsim_5.1.0
./python.sh /home/shu22/navigation/Navi_Agent/src/sim_vln_outdoor/scripts/<script>.py [args]
```

## 3. 脚本说明

### 3.1 load_scene.py — 场景加载

仅加载 USD 场景，用于验证场景文件是否正常。

```bash
./python.sh src/sim_vln_outdoor/scripts/load_scene.py
./python.sh src/sim_vln_outdoor/scripts/load_scene.py --usd-path /path/to/scene.usd
./python.sh src/sim_vln_outdoor/scripts/load_scene.py --headless
```

### 3.2 load_scene_robot.py — 机器人仿真

加载场景 + 生成 Go2W 机器人 + 运行 RL locomotion policy。

```bash
# 默认运行
./python.sh src/sim_vln_outdoor/scripts/load_scene_robot.py

# 指定速度指令 [vx, vy, yaw_rate]
./python.sh src/sim_vln_outdoor/scripts/load_scene_robot.py --cmd-vel 1.0 0.0 0.0

# 指定生成位置
./python.sh src/sim_vln_outdoor/scripts/load_scene_robot.py --spawn-pos -730.0 490.0 0.0

# 无头模式
./python.sh src/sim_vln_outdoor/scripts/load_scene_robot.py --headless
```

### 3.3 load_scene_view.py — 相机视角查看器

加载场景，创建模拟 D435i 相机的 prim，Isaac Sim viewport 直接显示该相机视角。支持键盘和 TCP socket 两种实时位姿控制。

```bash
# 仅查看（viewport 显示指定视角）
./python.sh src/sim_vln_outdoor/scripts/load_scene_view.py --camera-pos -730.0 490.0 1.5

# 启用键盘控制（点击 viewport 获取焦点后使用）
./python.sh src/sim_vln_outdoor/scripts/load_scene_view.py --keyboard --camera-pos -730.0 490.0 1.5

# 指定初始朝向 [roll, pitch, yaw]（度）
./python.sh src/sim_vln_outdoor/scripts/load_scene_view.py --keyboard --camera-pos -730.0 490.0 1.5 --camera-rot 0 0 90

# TCP socket 控制（用于 agent 集成）
./python.sh src/sim_vln_outdoor/scripts/load_scene_view.py --socket-port 9090
# 另一个终端发送位姿：
echo '{"x":-730,"y":490,"z":1.5,"yaw":90}' | nc localhost 9090

# 保存渲染帧
./python.sh src/sim_vln_outdoor/scripts/load_scene_view.py --save-dir /tmp/frames
```

#### 3.3.1 键盘控制

5-DOF + 拍照，需先点击 viewport 获取焦点。坐标系为 Isaac Sim 的 Z-up 右手系，按键方向与视觉直觉一致。

| 按键 | 功能 |
|------|------|
| W / S | 沿朝向前进 / 后退 |
| A / D | 左平移 / 右平移 |
| Q / E | 上升 / 下降 |
| ↑ / ↓ | 抬头 / 低头（pitch） |
| ← / → | 左转 / 右转（yaw） |
| = / - | 加速 / 减速（当前移动步长） |
| P | 拍照快照，保存到 `data/urbanverse/load_scene_view/<启动时间>/snap_XXXX.png` |
| Ctrl+C | 退出 |

#### 3.3.2 Socket 协议

每行一个 JSON，字段均可选（缺省保持当前值），服务端返回更新后的完整位姿。

```json
{"x": -730, "y": 490, "z": 1.5, "roll": 0, "pitch": 0, "yaw": 90}
```

#### 3.3.3 D435i 相机参数

| 参数 | 值 |
|------|-----|
| 分辨率 | 640 x 480 |
| fx / fy | 615.0 |
| cx / cy | 320.0 / 240.0 |
| 裁剪范围 | 0.1 ~ 100.0 m |

### 3.4 nav_eval.py — 闭环导航控制器评估

渲染 D435i 相机图像 → 输入控制器 → 输出动作 → 更新相机位姿 → 循环。用于验证导航控制器在仿真场景中的表现。

```bash
# 直行 + headless + 保存每步帧图片（默认 ForwardOnly 控制器）
./python.sh src/sim_vln_outdoor/scripts/nav_eval.py \
    --headless --save-frames --max-steps 200

# 指定初始位姿，直行 500 步
./python.sh src/sim_vln_outdoor/scripts/nav_eval.py \
    --camera-pos -730.0 490.0 1.5 --max-steps 500

# 随机游走 + headless + 保存每步帧图片
./python.sh src/sim_vln_outdoor/scripts/nav_eval.py \
    --headless --save-frames --max-steps 200 \
    --controller "nav.demo_controllers:RandomWalkController"

# 自定义控制器，2Hz 控制频率
./python.sh src/sim_vln_outdoor/scripts/nav_eval.py \
    --headless --controller "my_module:MyNavController" \
    --controller-freq 2.0 --max-steps 100 --save-frames
```

#### 3.4.1 自定义控制器

继承 `NavController`，实现 `act()` 方法即可接入评估循环：

```python
from nav import NavController, Observation, Action

class MyNavController(NavController):
    def act(self, obs: Observation) -> Action:
        rgb = obs.rgb          # (480, 640, 3) uint8
        pose = obs.pose        # (x, y, z, roll, pitch, yaw)
        step = obs.step        # 当前步数
        return Action(forward=0.5, yaw=2.0)
```

可选重写 `reset()`（episode 开始时调用）和 `on_episode_end(trajectory)`（结束时调用）。

#### 3.4.2 Action 正负号约定

与键盘控制一致，Z-up 右手系：

| 字段 | 正值含义 | 对应按键 |
|------|----------|----------|
| forward | 前进 | W |
| strafe | 左移 | A |
| vertical | 上升 | Q |
| pitch | 抬头 | ↑ |
| yaw | 左转 | ← |

设置 `action.done = True` 可让控制器主动结束 episode。

#### 3.4.3 输出格式

输出保存到 `data/urbanverse/nav_eval/<timestamp>/`：

| 文件 | 内容 |
|------|------|
| `trajectory.jsonl` | 每行一条 `{step, pose, action}` 记录（增量写入，中断不丢数据） |
| `summary.json` | 运行配置（控制器、初始位姿、总步数等） |
| `frame_XXXXXX.png` | 每步 RGB 帧图片（需 `--save-frames`） |

#### 3.4.4 CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--controller` | `nav.demo_controllers:ForwardOnlyController` | 控制器类（`module:ClassName` 格式） |
| `--max-steps` | 1000 | 最大控制步数 |
| `--controller-freq` | 20.0 | 控制器频率（Hz），支持低频如 2Hz |
| `--save-dir` | auto | 输出目录，默认 `data/urbanverse/nav_eval/<timestamp>/` |
| `--save-frames` | False | 保存每步 RGB 帧 |
| `--camera-pos` | [-693.5, 496.8, 2.0] | 初始相机位置 |
| `--camera-rot` | [0, 0, 360] | 初始相机朝向 (roll, pitch, yaw) |
| `--usd-path` | scene_09 | USD 场景路径 |
| `--headless` | False | 无头模式 |

## 4. 模块结构

```
sim_vln_outdoor/
├── scripts/
│   ├── load_scene.py           # 仅加载 USD 场景
│   ├── load_scene_robot.py     # 场景 + Go2W 机器人 + RL policy
│   ├── load_scene_view.py      # 场景 + D435i 相机视角查看器
│   └── nav_eval.py             # 闭环导航控制器评估
├── nav/
│   ├── controller.py           # NavController ABC + Observation / Action 定义
│   └── demo_controllers.py     # ForwardOnly / RandomWalk 示例控制器
├── env/
│   └── isaac_env.py            # IsaacSimEnv：SimulationApp + World 封装
├── robot/
│   ├── go2w.py                 # Go2WRobot：URDF 导入 + 关节映射
│   └── go2w_policy.py          # Go2WPolicy：TorchScript 推理 + PD 力矩
├── data/
│   └── view/
│       └── make_video.py       # 快照拼接为 30fps H.264 视频
└── assets/
    ├── policy/<robot>/         # RL policy 权重 + YAML 配置
    └── rl_sar_zoo/             # URDF / MJCF / mesh 资源
```

## 5. 注意事项

- 必须先 `conda deactivate`，Isaac Sim 自带 Python 与 conda 环境冲突
- Isaac Sim 要求 `SimulationApp` 在所有 `omni.*` import 之前创建，模块已遵循此约束
- 默认场景为 `scene_09_cbd_t_intersection_construction_sites`，通过 `--usd-path` 切换
- `--headless` 模式下没有 GUI viewport，键盘控制不可用，请使用 socket 控制或 nav_eval 闭环评估
