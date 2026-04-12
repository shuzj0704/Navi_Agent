"""
视图常量与几何工具
==================
VLM Navigator 和 Planner 共享的视角定义、缩写映射和角度工具函数。
"""

import math

VIEW_ORDER = ["front", "left", "right", "back"]

VIEW_ABBR = {"front": "f", "left": "l", "right": "r", "back": "b"}
ABBR_TO_VIEW = {v: k for k, v in VIEW_ABBR.items()}

VIEW_LABELS = {
    "front": "[f] 前视图:",
    "left":  "[l] 左视图:",
    "right": "[r] 右视图:",
    "back":  "[b] 后视图:",
}


def wrap_angle(a):
    """将角度差归一到 [-pi, pi]。"""
    return math.atan2(math.sin(a), math.cos(a))
