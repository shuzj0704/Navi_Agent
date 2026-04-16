# NaviAgent 运行指南

本文档记录室内外导航的运行方式。

---

## 1. 环境准备

### Conda 环境
使用 `internav_habitat` 环境（位于 `~/workspace/ll/env/miniforge3/envs/internav_habitat`）：
```bash
source ~/workspace/ll/env/miniforge3/bin/activate internav_habitat
```

依赖已安装：
- habitat-sim 0.2.4
- numpy 1.26.4
- opencv-python 4.9.0.80
- pillow 10.4.0
- fastapi, uvicorn
- pyyaml, httpx, openai, ultralytics

### 数据目录
```
/home/ps/workspace/ll/workspace/Navi_Agent/data/
├── mp3d_ce/          # 90+ 室内场景 (glb + navmesh)
├── vln_ce/          # VLN-CE 数据集
└── urbanverse/       # 室外轨迹数据
    └── trajectory/
        └── scene_09/
            ├── blog_point.txt
            └── dense_trajectory.json
```

### Isaac Sim (室外导航)
路径：`/home/ps/sources/isaacsim_4.5.0`（Isaac Sim 5.1 在当前环境有兼容性问题）

**必要依赖安装**：
```bash
/home/ps/sources/isaacsim_4.5.0/python.sh -m pip install openai httpx -q
```

---

## 2. 室内导航 (Habitat + HTTP)

### 2.1 启动仿真服务器

```bash
# Terminal 1
cd /home/ps/workspace/ll/workspace/Navi_Agent
source ~/workspace/ll/env/miniforge3/bin/activate internav_habitat

# 使用 xvfb-run 启动 headless 服务器
xvfb-run -a python -m sim_vln_indoor.env.server --port 5100
```

验证：
```bash
curl http://localhost:5100/health
curl http://localhost:5100/scenes
```

### 2.2 启动导航 Agent

```bash
# Terminal 2
cd /home/ps/workspace/ll/workspace/Navi_Agent
source ~/workspace/ll/env/miniforge3/bin/activate internav_habitat

# Mock 模式（不调 VLM）- 使用场景名
python src/scripts/nav_main.py --mock --scene 17DRP5sb8fy --steps 100 --save-vis

# 带 VLM（需要启动 vLLM）
python src/scripts/nav_main.py --scene 17DRP5sb8fy --steps 100 --save-vis
```

### 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `--sim-url` | `http://localhost:5100` | 仿真服务器地址 |
| `--scene` | - | 场景名 (e.g. `17DRP5sb8fy`) |
| `--steps` | 100 | 导航步数上限 |
| `--mock` | False | Mock 模式（不调 VLM） |
| `--save-vis` | False | 保存可视化视频 |
| `--instruction` | 默认探索指令 | 导航任务指令 |

### 输出

运行后会在 `output/nav/<scene_name>/<timestamp>/` 生成：
- `nav_debug.mp4` - 可视化视频
- `frame_XXXX.jpg` - 逐帧图像

---

## 3. 室外导航 (Isaac Sim + VLM)

### 3.1 启动 vLLM

```bash
# Terminal 1 (需要 vLLM 环境)
source ~/workspace/ll/env/miniforge3/bin/activate /home/ps/miniconda3/envs/lwy_swift
CUDA_VISIBLE_DEVICES=3 vllm serve /mnt/sda/szj/navi_dataset/checkpoints/Qwen3-VL-8B-Instruct/ \
  --served-model-name qwen3-vl \
  --port 8004 \
  --host 0.0.0.0 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.5
```

验证：
```bash
curl http://localhost:8004/v1/models
```

### 3.2 启动室外导航

```bash
# Terminal 2
cd /home/ps/workspace/ll/workspace/Navi_Agent
xvfb-run -a ./python.sh src/sim_vln_outdoor/scripts/vlm_gps_nav.py \
  --trajectory data/urbanverse/trajectory/scene_09/dense_trajectory.json \
  --usd-path /mnt/sda/szj/urban_verse/CraftBench/scene_09_cbd_t_intersection_construction_sites/Collected_export_version/export_version.usd \
  --headless \
  --max-steps 200 \
  --gpu 3 \
  --base-url http://localhost:8004 \
  --model qwen3-vl
```

**参数说明：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--trajectory` | - | 稠密轨迹 JSON 路径 |
| `--usd-path` | - | USD 场景文件路径 |
| `--headless` | False | 无窗口模式 |
| `--max-steps` | 200 | 导航步数上限 |
| `--gpu` | 0 | GPU 设备 ID |
| `--base-url` | `http://localhost:8004` | vLLM 服务器地址 |
| `--model` | `qwen3-vl` | vLLM 模型名 |
| `--controller-freq` | 1.0 | 控制频率 (Hz) |
| `--goal-tol` | 2.0 | 目标容差 (m) |

**Isaac Sim 路径：** `/home/ps/sources/isaacsim_4.5.0`
**USD 场景：** `/mnt/sda/szj/urban_verse/CraftBench/scene_09_*/Collected_export_version/export_version.usd`

### 输出

运行后会在 `data/urbanverse/vlm_gps_nav/<timestamp>/` 生成：
- `frames/` - 逐帧图像
- `vlm_io.jsonl` - VLM 输入输出
- `trajectory.jsonl` - 轨迹数据
- `summary.json` - 汇总统计
- `nav.mp4` - 可视化视频

## 2.3 快速启动（室内导航）

```bash
# 终端 1: 启动仿真服务器
cd /home/ps/workspace/ll/workspace/Navi_Agent
source ~/workspace/ll/env/miniforge3/bin/activate internav_habitat
xvfb-run -a python -m sim_vln_indoor.env.server --port 5100

# 终端 2: 启动导航 Agent
cd /home/ps/workspace/ll/workspace/Navi_Agent
source ~/workspace/ll/env/miniforge3/bin/activate internav_habitat
python src/scripts/nav_main.py --scene 17DRP5sb8fy --steps 100
```

---

## 4. 配置文件路径

| 文件 | 关键配置 |
|------|----------|
| `src/sim_vln_indoor/env/config/sim_server.yaml` | `scenes.base_dir` |
| `python.sh` | `ISAACSIM_ROOT` |
| `src/vlm_serve/configs/qwen3_vl_8b.yaml` | `model_path` |

---

## 5. 已知问题与限制

### 5.1 Habitat-sim EGL 初始化失败

**症状：** 
```
Platform::WindowlessEglApplication::tryCreateContext(): unable to find CUDA device X among Y EGL devices
WindowlessContext: Unable to create windowless context
```
或
```
Platform::WindowlessEglApplication::tryCreateContext(): cannot get default EGL display: EGL_BAD_PARAMETER
```

**根因：** habitat-sim headless 模式依赖 EGL 渲染，但 conda 环境中的 EGL 库与 NVIDIA 驱动不兼容。

**解决方案（已验证）：**

1. **设置 `gpu_device_id = -1`**（强制 CPU 渲染）
   ```yaml
   # src/sim_vln_indoor/env/config/sim_server.yaml
   gpu_device_id: -1
   ```

2. **配置正确的 NVIDIA EGL 库**
   ```python
   # src/sim_vln_indoor/env/server/__main__.py
   os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/home/ps/workspace/ll/workspace/Navi_Agent/data/10_nvidia.json"
   os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")
   ```

3. **创建 NVIDIA EGL 配置 JSON**
   ```json
   // data/10_nvidia.json
   {
       "file_format_version": "1.0.0",
       "ICD": {
           "library_path": "/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0"
       }
   }
   ```

4. **使用 `xvfb-run` 启动**
   ```bash
   xvfb-run -a python -m sim_vln_indoor.env.server --port 5100
   ```

**参考：** [GitHub Issue #2554](https://github.com/facebookresearch/habitat-sim/issues/2554), [CSDN 博客](https://blog.csdn.net/weixin_73189486/article/details/156989647)

### 5.2 Isaac Sim 相机频率问题

**症状：** `Exception: frequency of the camera sensor needs to be a divisible by the rendering frequency.`

**解决方案：** 将相机频率从 20Hz 改为 10Hz：
```python
# src/sim_vln_outdoor/scripts/nav_eval.py
camera = Camera(..., frequency=10, ...)  # 原来是 20
```

### 5.3 vLLM 端点 404 错误

**症状：** VLM 调用返回 `Error code: 404 - {'detail': 'Not Found'}`

**原因：** base_url 缺少 `/v1` 后缀

**解决方案：** 代码已自动处理，会自动添加 `/v1` 后缀

### 5.4 vLLM 内存不足

**解决方案：**
- 使用空闲 GPU：`CUDA_VISIBLE_DEVICES=3`
- 降低内存利用率：`--gpu-memory-utilization 0.5`
- 降低序列长度：`--max-model-len 2048`

---

## 6. 服务状态

| 服务 | 端口 | 状态 |
|------|------|------|
| vLLM (Qwen3-VL-8B) | 8004 | ✅ 可用 |
| Isaac Sim 4.5.0 | - | ✅ 可用 (室外导航) |
| Habitat Sim | 5100 | ✅ 可用 |

---

## 6. 服务状态检查

| 服务 | 端口 | 检查命令 |
|------|------|----------|
| Habitat Sim | 5100 | `curl http://localhost:5100/health` |
| vLLM | 8004 | `curl http://localhost:8004/v1/models` |

---

## 7. 室内外合并导航

### 7.1 配置文件

编辑 `configs/indoor_outdoor.yaml`:

```yaml
indoor:
  scene: "17DRP5sb8fy"           # 室内场景名
  max_steps: 100
  door_distance_threshold: 2.0   # 门距离阈值 (m)

outdoor:
  usd_path: "/path/to/export_version.usd"
  trajectory: "data/urbanverse/trajectory/scene_09/dense_trajectory.json"
  max_steps: 200

coordinate_transform:
  offset_x: -730.0
  offset_y: 490.0

vlm:
  api_url: "http://localhost:8004/v1"
  model: "qwen3-vl"
```

### 7.2 启动

```bash
# 终端 1: 启动 Habitat 仿真服务器
conda activate internav_habitat
xvfb-run -a python -m sim_vln_indoor.env.server --port 5100

# 终端 2: 启动 vLLM (可选，如需 VLM)
conda activate lwy_swift
CUDA_VISIBLE_DEVICES=3 vllm serve /path/to/Qwen3-VL-8B-Instruct/ \
  --served-model-name qwen3-vl --port 8004

# 终端 3: 运行室内外合并导航
conda activate internav_habitat
python src/scripts/indoor_outdoor_nav.py --config configs/indoor_outdoor.yaml
```

### 7.3 输出

| 类型 | 路径 |
|------|------|
| 合并输出 | `output/indoor_outdoor/<timestamp>/` |
| 室内视频 | `output/indoor_outdoor/<timestamp>/indoor/nav_debug.mp4` |
| 室外视频 | `data/urbanverse/vlm_gps_nav/<timestamp>/nav.mp4` |
| 评测指标 | `output/indoor_outdoor/<timestamp>/summary.json` |

### 7.4 评测指标 (summary.json)

```json
{
  "indoor": {
    "scene": "17DRP5sb8fy",
    "steps": 45,
    "success": true,
    "door_distance_m": 1.2,
    "door_position": [x, y, z]
  },
  "outdoor": {
    "scene": "scene_09",
    "steps": 80,
    "success": true,
    "final_distance_m": 1.5
  },
  "overall": {
    "total_steps": 125,
    "success": true
  }
}
```

---

## 8. 输出目录

| 类型 | 路径 |
|------|------|
| 室内导航可视化 | `output/nav/<scene>/<timestamp>/` |
| 室外导航数据 | `data/urbanverse/vlm_gps_nav/<timestamp>/` |