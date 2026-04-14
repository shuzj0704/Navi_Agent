"""
语义建图模块
============
输入: 前视 RGB + 对齐的深度图 (LiDAR 点云等效)
输出: 全局语义地图 (含物体 3D 包围盒) + 俯视 2D 渲染

流程:
  1. 分割器 (SAM3/Mock) 对 RGB 分割 → 物体掩码列表
  2. 掩码 + 深度 → 相机系 3D 点 → 全局坐标系 AABB
  3. 与已有物体 3D IoU 匹配，低于阈值才新增
  4. 俯视图实时渲染: 彩色矩形 + 标签

坐标系: 导航系 x=前, y=左, z=上 (高度)
"""

import numpy as np
import cv2
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .pixel_to_3d import get_camera_intrinsics
from .yoloe_segmentor import Segment


# 20 种高区分度颜色 (BGR)
PALETTE = [
    (86, 180, 233),  (0, 158, 115),  (230, 159, 0),   (0, 114, 178),
    (213, 94, 0),    (204, 121, 167),(240, 228, 66),  (0, 204, 204),
    (148, 103, 189), (44, 160, 44),  (214, 39, 40),   (255, 127, 14),
    (31, 119, 180),  (127, 127, 127),(188, 189, 34),  (23, 190, 207),
    (140, 86, 75),   (227, 119, 194),(199, 199, 199), (255, 152, 150),
]


@dataclass
class Object3D:
    """全局地图中的一个物体"""
    id: int
    label: str
    center: np.ndarray      # [nav_x, nav_y, z_height] 全局坐标
    size: np.ndarray         # [dx, dy, dz] AABB 尺寸 (米)
    color: Tuple[int, ...]   # BGR
    confidence: float = 0.5

    @property
    def volume(self):
        return float(self.size[0] * self.size[1] * self.size[2])

    @property
    def footprint_area(self):
        """俯视投影面积 m²"""
        return float(self.size[0] * self.size[1])

    @property
    def min_corner(self):
        return self.center - self.size / 2

    @property
    def max_corner(self):
        return self.center + self.size / 2


class SemanticMapper:
    def __init__(self, segmentor=None, overlap_threshold=0.1,
                 camera_height=1.5, camera_pitch_deg=-20.0,
                 camera_hfov=90, image_width=640, image_height=640,
                 min_volume=0.01, max_volume=50.0):
        """
        Args:
            segmentor:          分割器 (默认 MockSegmentor, 可替换为 SAM3Segmentor)
            overlap_threshold:  3D IoU ≥ 此值视为同一物体，< 此值新增
            camera_height:      前视相机安装高度 (m)
            camera_pitch_deg:   前视相机俯仰角 (度, 负=下看)
            min_volume, max_volume: 过滤过小/过大的检测结果
        """
        self.segmentor = segmentor or MockSegmentor()
        self.overlap_threshold = overlap_threshold
        self.camera_height = camera_height
        self.camera_pitch = math.radians(camera_pitch_deg)
        self.min_volume = min_volume
        self.max_volume = max_volume

        self.fx, self.fy, self.cx, self.cy = get_camera_intrinsics(
            image_width, image_height, camera_hfov
        )

        self.objects: List[Object3D] = []
        self._next_id = 0

    # ------------------------------------------------------------------
    #  核心更新
    # ------------------------------------------------------------------

    def update(self, rgb, depth, agent_nav_x, agent_nav_y, agent_yaw):
        """
        每帧调用: 分割 → 3D 定位 → 匹配去重 → 地图更新

        Args:
            rgb:   (H, W, 3) uint8 BGR 前视图
            depth: (H, W) float32 前视深度 (米, 与 rgb 像素对齐)
            agent_nav_x, agent_nav_y: 导航系位置
            agent_yaw: 航向角 (rad)
        """
        segments = self.segmentor.segment(rgb, depth)

        new_objects = []
        for seg in segments:
            result = self._mask_to_global_bbox(
                seg.mask, depth, agent_nav_x, agent_nav_y, agent_yaw
            )
            if result is None:
                continue
            center, size = result

            vol = float(size[0] * size[1] * size[2])
            if vol < self.min_volume or vol > self.max_volume:
                continue

            new_objects.append(Object3D(
                id=-1, label=seg.label,
                center=center, size=size,
                color=(0, 0, 0), confidence=seg.confidence,
            ))

        self._match_and_add(new_objects)

    # ------------------------------------------------------------------
    #  掩码 → 全局 3D AABB
    # ------------------------------------------------------------------

    def _mask_to_global_bbox(self, mask, depth, nav_x, nav_y, yaw):
        """
        掩码区域的深度点 → 相机系 3D → (考虑 pitch) → 全局导航系 AABB

        返回: (center[3], size[3]) 或 None
        """
        ys, xs = np.where(mask)
        if len(xs) < 10:
            return None

        d = depth[ys, xs]
        valid = (d > 0.1) & (d < 10.0)
        if valid.sum() < 10:
            return None

        xs = xs[valid].astype(np.float64)
        ys = ys[valid].astype(np.float64)
        d = d[valid].astype(np.float64)

        # 相机坐标系: x=右, y=下, z=前
        cam_x = (xs - self.cx) * d / self.fx
        cam_y = (ys - self.cy) * d / self.fy
        cam_z = d

        # 相机 → 机体 (考虑 pitch 旋转)
        # pitch < 0 表示下看, undo 旋转: R_x(-pitch) @ cam
        cp = math.cos(-self.camera_pitch)
        sp = math.sin(-self.camera_pitch)

        unpitched_y = cp * cam_y - sp * cam_z
        unpitched_z = sp * cam_y + cp * cam_z

        # 机体(Habitat): x=右, y=上, z=后
        body_x = cam_x
        body_y = -unpitched_y   # 上 = -下
        body_z = -unpitched_z   # 后 = -前

        # 机体 → 全局导航系
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # nav_x = -world_z, nav_y = -world_x
        # world_x = agent_hab_x + cos(yaw)*body_x + sin(yaw)*body_z
        # world_z = agent_hab_z - sin(yaw)*body_x + cos(yaw)*body_z
        # 展开后等价于:
        global_x = nav_x + (-body_z) * cos_yaw + body_x * sin_yaw
        global_y = nav_y + (-body_z) * sin_yaw - body_x * cos_yaw
        global_z = self.camera_height + body_y  # 绝对高度

        # 轴对齐包围盒
        mins = np.array([global_x.min(), global_y.min(), global_z.min()])
        maxs = np.array([global_x.max(), global_y.max(), global_z.max()])

        center = (mins + maxs) / 2
        size = np.maximum(maxs - mins, 0.05)  # 最小 5cm 避免零尺寸

        return center, size

    # ------------------------------------------------------------------
    #  匹配去重
    # ------------------------------------------------------------------

    def _match_and_add(self, new_objects):
        """新物体与已有物体匹配: XY投影重合时保留置信度更高的"""
        for obj in new_objects:
            overlapping_idx = None
            for i, existing in enumerate(self.objects):
                if self._overlap_xy(obj, existing):
                    overlapping_idx = i
                    break
            if overlapping_idx is not None:
                existing = self.objects[overlapping_idx]
                if obj.confidence > existing.confidence:
                    # 新物体置信度更高，替换已有物体（保留其 id 和颜色）
                    obj.id = existing.id
                    obj.color = existing.color
                    self.objects[overlapping_idx] = obj
            else:
                obj.id = self._next_id
                obj.color = PALETTE[self._next_id % len(PALETTE)]
                self._next_id += 1
                self.objects.append(obj)

    @staticmethod
    def _overlap_xy(a, b):
        """检查两个物体在XY平面投影是否有重合"""
        a_min, a_max = a.min_corner, a.max_corner
        b_min, b_max = b.min_corner, b.max_corner
        # X轴和Y轴都有交集才算重合
        return (a_min[0] < b_max[0] and a_max[0] > b_min[0] and
                a_min[1] < b_max[1] and a_max[1] > b_min[1])

    # ------------------------------------------------------------------
    #  俯视图渲染
    # ------------------------------------------------------------------

    def render_topdown(self, agent_x=None, agent_y=None, agent_yaw=None,
                       map_size=800, scale=40, trajectory=None):
        """
        渲染俯视 2D 语义地图。

        Args:
            agent_x, agent_y: 导航系坐标 (图像以 agent 为中心)
            agent_yaw: 航向角
            map_size: 图像尺寸 (像素)
            scale:    像素/米
            trajectory: [(x,y), ...] 历史轨迹点 (可选)

        图像约定: 右=导航x(前), 上=导航y(左)

        Returns:
            (map_size, map_size, 3) uint8 BGR
        """
        img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 245

        cx, cy = map_size // 2, map_size // 2
        ox = agent_x if agent_x is not None else 0.0
        oy = agent_y if agent_y is not None else 0.0

        # 1m 网格
        step = max(1, int(scale))
        for i in range(0, map_size, step):
            cv2.line(img, (i, 0), (i, map_size), (225, 225, 225), 1)
            cv2.line(img, (0, i), (map_size, i), (225, 225, 225), 1)

        # 轨迹
        if trajectory and len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                p1 = self._to_px(trajectory[i - 1][0], trajectory[i - 1][1],
                                 ox, oy, cx, cy, scale, map_size)
                p2 = self._to_px(trajectory[i][0], trajectory[i][1],
                                 ox, oy, cx, cy, scale, map_size)
                if p1 and p2:
                    cv2.line(img, p1, p2, (180, 220, 180), 2)

        # 物体矩形
        for obj in self.objects:
            self._draw_object(img, obj, ox, oy, cx, cy, scale, map_size)

        # Agent
        if agent_x is not None:
            cv2.circle(img, (cx, cy), 7, (0, 180, 0), -1)
            cv2.circle(img, (cx, cy), 7, (0, 0, 0), 1)
            if agent_yaw is not None:
                al = 22
                ax = int(cx + al * math.cos(agent_yaw))
                ay = int(cy - al * math.sin(agent_yaw))
                cv2.arrowedLine(img, (cx, cy), (ax, ay), (0, 180, 0), 2,
                                tipLength=0.4)

        # 信息
        cv2.putText(img, f"Objects: {len(self.objects)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img, f"IoU thresh: {self.overlap_threshold}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        cv2.putText(img, "Semantic Map (top-down)", (10, map_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        return img

    def _draw_object(self, img, obj, ox, oy, cx, cy, scale, map_size):
        """画单个物体: 填色矩形 + 标签"""
        dx = obj.center[0] - ox
        dy = obj.center[1] - oy
        hw = obj.size[0] / 2 * scale
        hh = obj.size[1] / 2 * scale

        px = cx + dx * scale
        py = cy - dy * scale

        x1, y1 = int(px - hw), int(py - hh)
        x2, y2 = int(px + hw), int(py + hh)

        # 可见性检查
        if x2 < 0 or x1 >= map_size or y2 < 0 or y1 >= map_size:
            return

        # 半透明填充
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), obj.color, -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        # 边框
        cv2.rectangle(img, (x1, y1), (x2, y2), obj.color, 2)

        # 标签
        text = f"{obj.label}"
        ty = max(y1 - 8, 15)
        cv2.putText(img, text, (x1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)

    @staticmethod
    def _to_px(nx, ny, ox, oy, cx, cy, scale, sz):
        px = int(cx + (nx - ox) * scale)
        py = int(cy - (ny - oy) * scale)
        if 0 <= px < sz and 0 <= py < sz:
            return (px, py)
        return None
