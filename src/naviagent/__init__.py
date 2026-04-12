"""
NaviAgent 核心模块
==================
按功能分为四个子包:
  - perception/  感知模块 (输入侧): 目标检测、深度处理、语义建图、观测读取
  - decision/    决策模块 (输出侧): DWA 规划、编排器、导航引擎
  - vlm/         VLM 模块: System 1 快速 VLM + System 2 慢思考规划
  - common/      公共模块: 导航状态、坐标变换、视图常量、可视化
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

# 从子包统一导出, 保持向后兼容
from .common import (
    NavState,
    VIEW_ORDER, VIEW_ABBR, ABBR_TO_VIEW, VIEW_LABELS, wrap_angle,
    habitat_pos_to_nav2d, habitat_quat_to_yaw, nav2d_to_habitat_pos,
    camera_points_to_robot2d, camera_point_to_robot2d,
    camera_points_to_nav2d, camera_point_to_nav2d,
    draw_debug_frame, build_panel_info,
)
from .perception import (
    get_camera_intrinsics, pixel_to_camera_3d, depth_to_pointcloud,
    YOLOESegmentor, Segment,
    SemanticMapper,
    ObsBundle, HabitatObsReader, SimClientObsReader,
)
from .decision import (
    VLMNavigator, VLMAsyncWorker,
    System2Planner, PlanDecision,
    TaskOrchestrator,
    DWAPlanner,
    TurnController,
    NavigationEngine, NavEngineConfig, StepResult,
)
