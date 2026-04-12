"""
Habitat 仿真工具函数
====================
从 nav_main.py 提取的传感器配置、仿真器创建、动作转换等工具。
独立于主循环, 供 nav_main.py 和 batch_eval.py 共享。
"""

import os
import math
import habitat_sim
import numpy as np


# ========== 传感器配置 ==========

_CAM_BASE = {
    "position": [0.0, 0.5, 0.0],
    "pitch": 0.0,
    "roll": 0.0,
    "hfov": 120,
    "width": 640,
    "height": 480,
}

SENSOR_CONFIGS = {
    # 四视角 RGB (VLM 输入)
    "front_rgb":   {**_CAM_BASE, "type": "COLOR", "yaw": 0.0},
    "left_rgb":    {**_CAM_BASE, "type": "COLOR", "yaw": 90.0},
    "right_rgb":   {**_CAM_BASE, "type": "COLOR", "yaw": -90.0},
    "back_rgb":    {**_CAM_BASE, "type": "COLOR", "yaw": 180.0},
    # 前视深度 (DWA 目标点反投影)
    "front_depth": {**_CAM_BASE, "type": "DEPTH", "yaw": 0.0},
    # 低位深度 (DWA 障碍物点云)
    "low_depth": {
        "type": "DEPTH",
        "position": [0.0, 0.5, 0.0],
        "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
        "hfov": 90, "width": 640, "height": 480,
    },
}

# ========== 路径常量 ==========

SCENE_DIR = "/home/nuc/vln/data/scene_data/mp3d"
VLM_API_URL = "http://192.168.1.137:8000/v1"


# ========== Habitat 初始化 ==========

def make_sensor_spec(uuid, cfg):
    """从配置字典创建 Habitat 传感器"""
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = getattr(habitat_sim.SensorType, cfg["type"])
    spec.resolution = [cfg["height"], cfg["width"]]
    spec.hfov = cfg["hfov"]
    spec.position = cfg["position"]
    spec.orientation = [
        math.radians(cfg["pitch"]),
        math.radians(cfg["yaw"]),
        math.radians(cfg["roll"]),
    ]
    return spec


def create_sim(scene_path):
    """创建带全部传感器的仿真器"""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0

    sensors = [make_sensor_spec(k, v) for k, v in SENSOR_CONFIGS.items()]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


# ========== 运动执行 ==========

def velocity_to_action(v, omega, v_thresh=0.05, w_thresh=0.25):
    """DWA 输出 (v, omega) -> Habitat 离散动作列表
    坐标系 X=前 Y=右: omega>0 -> 右转, omega<0 -> 左转
    同时满足速度和方向阈值时返回两步动作"""
    actions = []
    if abs(omega) > w_thresh:
        actions.append("turn_right" if omega > 0 else "turn_left")
    if v > v_thresh:
        actions.append("move_forward")
    if not actions:
        actions.append("move_forward")  # 兜底前进，避免死锁
    return actions


# ========== 场景发现 ==========

def discover_scenes(scene_dir=None):
    """扫描场景目录, 返回 [(name, glb_path), ...] 列表"""
    scene_dir = scene_dir or SCENE_DIR
    scenes = []
    for name in sorted(os.listdir(scene_dir)):
        glb = os.path.join(scene_dir, name, f"{name}.glb")
        if os.path.exists(glb):
            scenes.append((name, glb))
    return scenes
