"""
坐标变换模块
============
Habitat 相机系 ↔ 2D 导航系之间的变换。

坐标系约定:
  Habitat 3D : x=右, y=上, z=后. 默认朝向 -z.
  相机系     : x=右, y=下, z=前 (光轴).
  2D 导航系  : x=前, y=左, yaw 逆时针. (标准 ROS REP-103)

Habitat → 导航系映射:
  nav_x = -habitat_z     (前)
  nav_y = -habitat_x     (左)
  nav_yaw = habitat_yaw  (绕 y 轴旋转)

此映射保证标准运动学模型成立:
  x += v * cos(yaw) * dt
  y += v * sin(yaw) * dt
"""

import numpy as np
import math


def habitat_pos_to_nav2d(position):
    """Habitat 3D 位置 [x,y,z] → 导航系 (nav_x, nav_y)"""
    return float(-position[2]), float(-position[0])


def habitat_quat_to_yaw(rotation):
    """Habitat 四元数 → yaw (弧度)
    rotation: 带 x, y, z, w 属性的四元数
    """
    siny = 2.0 * (rotation.w * rotation.y + rotation.x * rotation.z)
    cosy = 1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z)
    return math.atan2(siny, cosy)


def nav2d_to_habitat_pos(nav_x, nav_y, height=0.07):
    """导航系 → Habitat 3D 位置 (逆变换, 用于设置 agent 位置)"""
    return np.array([-nav_y, height, -nav_x])


# ------------------------------------------------------------------
#  相机系 → 机器人局部 2D（DWA 用，无需里程计）
# ------------------------------------------------------------------

def camera_points_to_robot2d(cam_points):
    """
    相机/机器人系 3D 点云 → 机器人局部 2D。
    坐标系: X=前, Y=右, Z=上 (相机系=机器人系)
    2D 投影: 取 (X, Y) 即 (前, 右)

    参数: (N, 3) 点云 [X_前, Y_右, Z_上]
    返回: (N, 2) 机器人局部 2D [X_前, Y_右]
    """
    if len(cam_points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return cam_points[:, :2].astype(np.float32)


def camera_point_to_robot2d(cam_point):
    """单点版本: 3D → 机器人局部 2D (X_前, Y_右)"""
    return float(cam_point[0]), float(cam_point[1])


# ------------------------------------------------------------------
#  相机系 → 全局导航系 2D（语义建图用，需要里程计）
# ------------------------------------------------------------------

def camera_points_to_nav2d(cam_points, agent_nav_x, agent_nav_y, agent_yaw):
    """
    机器人系 3D 点云 → 全局导航系 2D (语义建图用)。

    坐标系: 点云 [X_前, Y_右, Z_上], 导航系 [nav_x=前, nav_y=左]
    公式:
      nav_x = agent_x + X * cos(yaw) + Y * sin(yaw)
      nav_y = agent_y + X * sin(yaw) - Y * cos(yaw)

    参数:
        cam_points: (N, 3) 点云 [X_前, Y_右, Z_上]
        agent_nav_x, agent_nav_y: agent 在导航系的位置
        agent_yaw: agent 航向角
    返回:
        (N, 2) 导航系 2D [nav_x, nav_y]
    """
    if len(cam_points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    fwd = cam_points[:, 0]   # X = 前
    rgt = cam_points[:, 1]   # Y = 右

    cos_yaw = np.cos(agent_yaw)
    sin_yaw = np.sin(agent_yaw)

    nav_x = agent_nav_x + fwd * cos_yaw + rgt * sin_yaw
    nav_y = agent_nav_y + fwd * sin_yaw - rgt * cos_yaw

    return np.stack([nav_x, nav_y], axis=-1).astype(np.float32)


def camera_point_to_nav2d(cam_point, agent_nav_x, agent_nav_y, agent_yaw):
    """单点版本: 机器人系 3D → 导航系 2D"""
    fwd, rgt = cam_point[0], cam_point[1]
    cos_yaw = math.cos(agent_yaw)
    sin_yaw = math.sin(agent_yaw)
    nav_x = agent_nav_x + fwd * cos_yaw + rgt * sin_yaw
    nav_y = agent_nav_y + fwd * sin_yaw - rgt * cos_yaw
    return float(nav_x), float(nav_y)
