"""
Habitat 仿真客户端工具
======================
仅保留客户端用得到的辅助 (动作转换、场景发现、路径常量)。
传感器规格统一由仿真服务器通过 GET /sensors 提供 (源头是
`env/config/sim_server.yaml`), 客户端不再硬编码。
"""

import os
import numpy as np


# ========== 路径常量 ==========

SCENE_DIR = "data/scene_data/mp3d"
VLM_API_URL = "http://localhost:8004/v1"


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
