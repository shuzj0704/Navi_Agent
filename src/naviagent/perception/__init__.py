"""
感知模块 (输入侧)
==================
传感器数据处理、目标检测、语义建图、观测读取。
"""

from .pixel_to_3d import get_camera_intrinsics, pixel_to_camera_3d, depth_to_pointcloud
from .yoloe_segmentor import YOLOESegmentor, Segment
from .semantic_mapper import SemanticMapper
from .obs_reader import ObsBundle, HabitatObsReader, SimClientObsReader


def __getattr__(name):
    """Lazy import — Sam3Segmentor pulls torch/sam3; don't force it on YOLOE-only users."""
    if name == "Sam3Segmentor":
        from .sam3_segmentor import Sam3Segmentor
        return Sam3Segmentor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
