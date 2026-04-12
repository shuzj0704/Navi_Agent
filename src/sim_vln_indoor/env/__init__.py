"""
Indoor Simulation Package
=========================
独立的室内仿真模块, 可直接移植到其他项目中使用。

包含:
  - server/   FastAPI HTTP 仿真服务端
  - client/   HTTP 客户端
  - config/   传感器与服务器 YAML 配置
  - habitat_utils  Habitat 工具函数 (传感器配置, 仿真器创建, 动作转换)
"""

from .habitat_utils import (
    SENSOR_CONFIGS,
    SCENE_DIR,
    VLM_API_URL,
    create_sim,
    make_sensor_spec,
    velocity_to_action,
    discover_scenes,
)
from .client import SimClient, AgentState
