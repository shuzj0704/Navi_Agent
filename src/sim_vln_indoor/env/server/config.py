"""
仿真服务器配置加载
==================
读取 YAML 配置文件, 合并 camera_defaults 到每个传感器。
"""

import getpass
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SensorConfig:
    type: str           # "COLOR" | "DEPTH"
    position: List[float] = field(default_factory=lambda: [0.0, 0.5, 0.0])
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    hfov: int = 120
    width: int = 640
    height: int = 640


@dataclass
class EncodingConfig:
    rgb_format: str = "jpeg"
    rgb_jpeg_quality: int = 90
    depth_format: str = "raw_f32"


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 5100
    gpu_device_id: int = 0
    scenes_base_dir: str = "data/scene_data/mp3d"
    default_scene: Optional[str] = None
    enable_physics: bool = False
    sensors: Dict[str, SensorConfig] = field(default_factory=dict)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)


def load_config(path: str) -> ServerConfig:
    """加载 YAML 配置, 合并 camera_defaults 到每个传感器."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    server_raw = raw.get("server", {})
    scenes_raw = raw.get("scenes", {})
    physics_raw = raw.get("physics", {})
    defaults = raw.get("camera_defaults", {})
    sensors_raw = raw.get("sensors", {})
    encoding_raw = raw.get("encoding", {})

    sensors = {}
    for name, overrides in sensors_raw.items():
        merged = {**defaults, **overrides}
        sensors[name] = SensorConfig(
            type=merged["type"],
            position=merged.get("position", [0.0, 0.5, 0.0]),
            pitch=merged.get("pitch", 0.0),
            yaw=merged.get("yaw", 0.0),
            roll=merged.get("roll", 0.0),
            hfov=merged.get("hfov", 120),
            width=merged.get("width", 640),
            height=merged.get("height", 640),
        )

    # base_dir 支持按当前系统用户名选择 (便于 nuc / ps 共用同一份配置)
    by_user = scenes_raw.get("base_dir_by_user", {}) or {}
    scenes_base_dir = by_user.get(getpass.getuser()) or scenes_raw.get(
        "base_dir", "data/scene_data/mp3d"
    )

    gpu_by_user = server_raw.get("gpu_device_id_by_user", {}) or {}
    user = getpass.getuser()
    gpu_device_id = gpu_by_user.get(user)
    if gpu_device_id is None:
        gpu_device_id = server_raw.get("gpu_device_id", 0)

    return ServerConfig(
        host=server_raw.get("host", "0.0.0.0"),
        port=server_raw.get("port", 5100),
        gpu_device_id=gpu_device_id,
        scenes_base_dir=scenes_base_dir,
        default_scene=scenes_raw.get("default_scene"),
        enable_physics=physics_raw.get("enable", False),
        sensors=sensors,
        encoding=EncodingConfig(
            rgb_format=encoding_raw.get("rgb_format", "jpeg"),
            rgb_jpeg_quality=encoding_raw.get("rgb_jpeg_quality", 90),
            depth_format=encoding_raw.get("depth_format", "raw_f32"),
        ),
    )
