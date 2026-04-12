"""
导航状态数据结构
===============
DWA 规划器的统一输入/输出接口。
所有坐标在 2D 导航系下：x=前, y=左, yaw=逆时针(从x轴)。

仿真阶段：SimProvider 填充，DWA 读取
真机部署：ROS2Provider 通过话题填充，同一个 DWA 读取
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class NavState:
    # 机器人状态 (2D 导航系)
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0       # 弧度, 逆时针为正
    v: float = 0.0          # 线速度 m/s
    omega: float = 0.0      # 角速度 rad/s

    # 局部目标 (2D 导航系)
    goal_x: float = 0.0
    goal_y: float = 0.0
    goal_valid: bool = False

    # 障碍物 (2D 导航系)
    obstacles: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )

    # DWA 输出
    cmd_v: float = 0.0
    cmd_omega: float = 0.0

    timestamp: float = 0.0

    @property
    def state_vec(self):
        """[x, y, yaw, v, omega] 供 DWA 使用"""
        return np.array([self.x, self.y, self.yaw, self.v, self.omega])

    @property
    def goal(self):
        """[gx, gy]"""
        return np.array([self.goal_x, self.goal_y])
