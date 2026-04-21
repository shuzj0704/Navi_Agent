"""
观测读取模块
============
平台无关的观测数据结构 + 平台特定的读取器。
仿真直连: HabitatObsReader
仿真HTTP: SimClientObsReader
真机部署: 可替换为 ROS2ObsReader 等
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict

from .pixel_to_3d import get_camera_intrinsics, depth_to_pointcloud
from ..common.coordinate_transform import (
    habitat_pos_to_nav2d,
    habitat_quat_to_yaw,
    camera_points_to_robot2d,
    camera_points_to_nav2d,
)


@dataclass
class ObsBundle:
    """一帧处理后的观测, 平台无关。"""
    nav_x: float
    nav_y: float
    nav_yaw: float
    obstacles_local: np.ndarray     # (N, 2) 机器人坐标系 [前, 右]
    obstacles_global: np.ndarray    # (N, 2) 导航坐标系 [nav_x, nav_y]
    front_depth: np.ndarray         # (H, W) float32
    views_bgr: Dict[str, np.ndarray]  # {"front": (H,W,3), "left": ..., "right": ...}


class HabitatObsReader:
    """从 Habitat 仿真器读取观测并转换为 ObsBundle。

    封装了原 nav_main.py 中的 refresh_obs() 逻辑。
    """

    def __init__(self, sim, sensor_configs,
                 low_max_range=5.0, low_stride=8,
                 low_y_min=-1.5, low_y_max=0.3):
        self.sim = sim
        self.sensor_configs = sensor_configs

        # 低位深度内参 (障碍物点云)
        low_cfg = sensor_configs["low_depth"]
        self.low_fx, self.low_fy, self.low_cx, self.low_cy = get_camera_intrinsics(
            low_cfg["width"], low_cfg["height"], low_cfg["hfov"]
        )
        self.low_max_range = low_max_range
        self.low_stride = low_stride
        self.low_y_min = low_y_min
        self.low_y_max = low_y_max

    def read(self) -> ObsBundle:
        """读一帧观测, 返回 ObsBundle。"""
        obs = self.sim.get_sensor_observations()
        agent_state = self.sim.get_agent(0).get_state()
        nav_x, nav_y = habitat_pos_to_nav2d(agent_state.position)
        nav_yaw = habitat_quat_to_yaw(agent_state.rotation)

        # 低位深度 → 障碍物点云
        low_d = obs["low_depth"]
        if low_d.ndim == 3:
            low_d = low_d[:, :, 0]
        cam_cloud = depth_to_pointcloud(
            low_d, self.low_fx, self.low_fy, self.low_cx, self.low_cy,
            max_range=self.low_max_range, stride=self.low_stride,
            y_min=self.low_y_min, y_max=self.low_y_max,
        )
        obstacles_local = camera_points_to_robot2d(cam_cloud)
        obstacles_global = camera_points_to_nav2d(cam_cloud, nav_x, nav_y, nav_yaw)

        # 前视深度
        fd = obs["front_depth"]
        if fd.ndim == 3:
            fd = fd[:, :, 0]

        # RGB 视图 BGR (back 仅在传感器存在时读取)
        views_bgr = {}
        for vname in ["front", "left", "right", "back"]:
            sk = f"{vname}_rgb"
            if sk in obs:
                views_bgr[vname] = cv2.cvtColor(obs[sk][:, :, :3], cv2.COLOR_RGB2BGR)

        return ObsBundle(
            nav_x=nav_x, nav_y=nav_y, nav_yaw=nav_yaw,
            obstacles_local=obstacles_local,
            obstacles_global=obstacles_global,
            front_depth=fd,
            views_bgr=views_bgr,
        )


class SimClientObsReader:
    """通过 HTTP 客户端从 sim_server 读取观测。

    与 HabitatObsReader 接口完全一致, 可互换使用。
    """

    def __init__(self, client, sensor_configs,
                 low_max_range=5.0, low_stride=8,
                 low_y_min=-1.5, low_y_max=0.3):
        """
        Args:
            client: sim_client.SimClient 实例
            sensor_configs: 传感器配置字典 (与 HabitatObsReader 格��相同)
        """
        self.client = client
        self.sensor_configs = sensor_configs

        low_cfg = sensor_configs["low_depth"]
        self.low_fx, self.low_fy, self.low_cx, self.low_cy = get_camera_intrinsics(
            low_cfg["width"], low_cfg["height"], low_cfg["hfov"]
        )
        self.low_max_range = low_max_range
        self.low_stride = low_stride
        self.low_y_min = low_y_min
        self.low_y_max = low_y_max

    def read(self) -> ObsBundle:
        """读一帧观测, 返回 ObsBundle。"""
        obs, agent_state = self.client.get_observations()
        nav_x, nav_y = habitat_pos_to_nav2d(agent_state.position)
        nav_yaw = habitat_quat_to_yaw(agent_state.rotation)

        # 低位深度 → 障碍物点云
        low_d = obs["low_depth"]
        if low_d.ndim == 3:
            low_d = low_d[:, :, 0]
        cam_cloud = depth_to_pointcloud(
            low_d, self.low_fx, self.low_fy, self.low_cx, self.low_cy,
            max_range=self.low_max_range, stride=self.low_stride,
            y_min=self.low_y_min, y_max=self.low_y_max,
        )
        obstacles_local = camera_points_to_robot2d(cam_cloud)
        obstacles_global = camera_points_to_nav2d(cam_cloud, nav_x, nav_y, nav_yaw)

        # 前视深度
        fd = obs["front_depth"]
        if fd.ndim == 3:
            fd = fd[:, :, 0]

        # RGB 视图 (SimClient 返回的 RGB 图已经是 BGR, back 仅在传感器存在时读取)
        views_bgr = {}
        for vname in ["front", "left", "right", "back"]:
            sk = f"{vname}_rgb"
            if sk in obs:
                views_bgr[vname] = obs[sk]

        return ObsBundle(
            nav_x=nav_x, nav_y=nav_y, nav_yaw=nav_yaw,
            obstacles_local=obstacles_local,
            obstacles_global=obstacles_global,
            front_depth=fd,
            views_bgr=views_bgr,
        )
