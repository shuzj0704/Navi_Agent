"""
像素坐标 → 3D 坐标 (机器人/相机坐标系)
======================================
根据深度图将像素 (u, v) 反投影为 3D 点。

统一坐标系约定 (相机系 = 机器人系):
    X: 前 (forward, 光轴方向)
    Y: 右 (right)
    Z: 上 (up)

Habitat 原始相机输出为 (right, down, forward)，
本模块内部转换为 (forward, right, up) 后输出。

内参由 hfov 和图像尺寸推算:
    fx = fy = width / (2 * tan(hfov / 2))
    cx = width / 2
    cy = height / 2
"""

import numpy as np
import math


def get_camera_intrinsics(width, height, hfov_deg):
    """
    根据图像尺寸和水平视场角计算相机内参。

    返回:
        fx, fy, cx, cy
    """
    hfov_rad = math.radians(hfov_deg)
    fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx  # 正方形像素
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def pixel_to_camera_3d(u, v, depth, fx, fy, cx, cy, patch_size=5, max_range=10.0):
    """
    将像素 (u, v) 反投影为相机坐标系下的 3D 坐标。

    参数:
        u, v:       目标像素坐标 (u=列, v=行)
        depth:      (H, W) float32 深度图，单位：米
        fx, fy:     焦距（像素单位）
        cx, cy:     主点坐标
        patch_size: 取目标点附近 patch 的半径，用中位数更鲁棒
        max_range:  最大有效深度（米）

    返回:
        (x, y, z) 相机坐标系下的 3D 坐标，或 None（无有效深度时）
    """
    H, W = depth.shape

    # 取目标点附近区域的深度中位数
    v_min = max(0, v - patch_size)
    v_max = min(H, v + patch_size)
    u_min = max(0, u - patch_size)
    u_max = min(W, u + patch_size)
    patch = depth[v_min:v_max, u_min:u_max]

    valid = patch[(patch > 0) & (patch < max_range)]
    if len(valid) == 0:
        return None

    d = float(np.median(valid))
    # Habitat 原始: right = (u-cx)*d/fx, down = (v-cy)*d/fy, forward = d
    # 转为统一坐标系: X=前, Y=右, Z=上
    X = d                       # forward
    Y = (u - cx) * d / fx      # right
    Z = -(v - cy) * d / fy     # up (= -down)

    return np.array([X, Y, Z])


def depth_to_pointcloud(depth, fx, fy, cx, cy, max_range=10.0, stride=1,
                        y_min=None, y_max=None):
    """
    将整张深度图转换为相机坐标系下的 3D 点云。

    参数:
        depth:      (H, W) float32 深度图，单位：米
        fx, fy, cx, cy: 相机内参
        max_range:  最大有效深度
        stride:     降采样步长 (stride=4 → 每4个像素取1个，点数降为 1/16)
        y_min, y_max: 高度过滤 (沿用旧接口名, 内部转为 Z 轴过滤)。
                      相机高 0.5m 时, y_min=-1.5 y_max=0.3
                      → 保留 Z ∈ [-0.3, 1.5], 即离地 0.2m~2.0m 的点

    返回:
        points: (N, 3) 有效点云，每行 [X_前, Y_右, Z_上]
    """
    H, W = depth.shape

    u_coords = np.arange(0, W, stride)
    v_coords = np.arange(0, H, stride)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)

    d = depth[::stride, ::stride]
    # Habitat 原始 → 统一坐标系 X=前, Y=右, Z=上
    X = d                          # forward
    Y = (u_grid - cx) * d / fx     # right
    Z = -(v_grid - cy) * d / fy    # up (= -down)

    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    # 过滤无效深度
    d_flat = d.reshape(-1)
    valid = (d_flat > 0) & (d_flat < max_range)

    # 过滤高度 Z (去地面和天花板)
    # z_min/z_max 对应旧参数 y_min/y_max 但符号相反 (Z = -old_y)
    z_flat = Z.reshape(-1)
    if y_min is not None:
        valid &= z_flat <= -y_min   # old y_min → new Z upper bound
    if y_max is not None:
        valid &= z_flat >= -y_max   # old y_max → new Z lower bound

    return points[valid]
