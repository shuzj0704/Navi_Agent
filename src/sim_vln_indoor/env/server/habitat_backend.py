"""
Habitat 仿真后端
================
线程安全的 Habitat-sim 封装, 供 FastAPI 端点调用。
"""

import os
import math
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import habitat_sim

from .config import ServerConfig, SensorConfig


@dataclass
class AgentStateData:
    position: List[float]
    rotation: List[float]  # [qx, qy, qz, qw]


class HabitatBackend:
    """线程安全的 Habitat 仿真器封装。"""

    def __init__(self, config: ServerConfig):
        self.config = config
        self._sim: Optional[habitat_sim.Simulator] = None
        self._lock = threading.Lock()
        self._scene_name: Optional[str] = None
        self._scene_path: Optional[str] = None
        self._navmesh_loaded: bool = False

    def load_scene(self, scene_name: str = None,
                   scene_path: str = None) -> dict:
        """加载场景, 自动加载 navmesh。"""
        with self._lock:
            if self._sim is not None:
                self._sim.close()
                self._sim = None

            if scene_path is None and scene_name is not None:
                scene_path = os.path.join(
                    self.config.scenes_base_dir,
                    scene_name, f"{scene_name}.glb"
                )

            if not os.path.exists(scene_path):
                raise FileNotFoundError(f"Scene not found: {scene_path}")

            # 构建仿真器
            sim_cfg = habitat_sim.SimulatorConfiguration()
            sim_cfg.scene_id = scene_path
            sim_cfg.enable_physics = self.config.enable_physics
            sim_cfg.gpu_device_id = self.config.gpu_device_id

            sensors = [
                self._make_sensor_spec(name, scfg)
                for name, scfg in self.config.sensors.items()
            ]
            agent_cfg = habitat_sim.agent.AgentConfiguration()
            agent_cfg.sensor_specifications = sensors

            self._sim = habitat_sim.Simulator(
                habitat_sim.Configuration(sim_cfg, [agent_cfg])
            )

            # 自动加载 navmesh
            navmesh_path = scene_path.replace(".glb", ".navmesh")
            self._navmesh_loaded = False
            if os.path.exists(navmesh_path):
                self._sim.pathfinder.load_nav_mesh(navmesh_path)
                self._navmesh_loaded = True

            self._scene_name = scene_name or os.path.basename(
                os.path.dirname(scene_path)
            )
            self._scene_path = scene_path

            return {
                "scene_name": self._scene_name,
                "scene_path": self._scene_path,
                "navmesh_loaded": self._navmesh_loaded,
            }

    def get_observations(self) -> Tuple[Dict[str, np.ndarray], AgentStateData]:
        """获取所有传感器观测 + agent 状态。"""
        with self._lock:
            self._ensure_sim()
            obs = self._sim.get_sensor_observations()
            agent_state = self._sim.get_agent(0).get_state()
            state_data = AgentStateData(
                position=[float(v) for v in agent_state.position],
                rotation=[
                    float(agent_state.rotation.x),
                    float(agent_state.rotation.y),
                    float(agent_state.rotation.z),
                    float(agent_state.rotation.w),
                ],
            )
            return obs, state_data

    def get_single_observation(self, sensor_name: str) -> np.ndarray:
        """获取单个传感器观测。"""
        with self._lock:
            self._ensure_sim()
            obs = self._sim.get_sensor_observations()
            if sensor_name not in obs:
                raise KeyError(f"Unknown sensor: {sensor_name}")
            return obs[sensor_name]

    def execute_action(self, action: str) -> AgentStateData:
        """执行单个离散动作, 返回动作后状态。"""
        with self._lock:
            self._ensure_sim()
            self._sim.get_agent(0).act(action)
            return self._get_agent_state_data()

    def execute_actions(self, actions: List[str]) -> AgentStateData:
        """执行多个离散动作, 返回最终状态。"""
        with self._lock:
            self._ensure_sim()
            agent = self._sim.get_agent(0)
            for action in actions:
                agent.act(action)
            return self._get_agent_state_data()

    def get_agent_state(self) -> AgentStateData:
        """获取 agent 状态 (不渲染传感器)。"""
        with self._lock:
            self._ensure_sim()
            return self._get_agent_state_data()

    def set_agent_state(self, position: List[float],
                        rotation: Optional[List[float]] = None) -> AgentStateData:
        """传送 agent 到指定位姿。rotation: [qx, qy, qz, qw]"""
        with self._lock:
            self._ensure_sim()
            agent = self._sim.get_agent(0)
            state = agent.get_state()
            state.position = np.array(position, dtype=np.float32)
            if rotation is not None:
                qx, qy, qz, qw = rotation
                state.rotation = np.quaternion(qw, qx, qy, qz)
            agent.set_state(state)
            return self._get_agent_state_data()

    def list_scenes(self) -> List[dict]:
        """列出可用场景。"""
        scenes = []
        base = self.config.scenes_base_dir
        if not os.path.isdir(base):
            return scenes
        for name in sorted(os.listdir(base)):
            glb = os.path.join(base, name, f"{name}.glb")
            if os.path.exists(glb):
                navmesh = glb.replace(".glb", ".navmesh")
                scenes.append({
                    "name": name,
                    "has_navmesh": os.path.exists(navmesh),
                })
        return scenes

    def get_scene_info(self) -> Optional[dict]:
        """返回当前场景信息, 未加载时返回 None。"""
        if self._sim is None:
            return None
        return {
            "scene_name": self._scene_name,
            "scene_path": self._scene_path,
            "navmesh_loaded": self._navmesh_loaded,
            "sensors": list(self.config.sensors.keys()),
        }

    def close(self):
        with self._lock:
            if self._sim is not None:
                self._sim.close()
                self._sim = None

    # ------------------------------------------------------------------

    def _ensure_sim(self):
        if self._sim is None:
            raise RuntimeError("No scene loaded. POST /scene first.")

    def _get_agent_state_data(self) -> AgentStateData:
        state = self._sim.get_agent(0).get_state()
        return AgentStateData(
            position=[float(v) for v in state.position],
            rotation=[
                float(state.rotation.x),
                float(state.rotation.y),
                float(state.rotation.z),
                float(state.rotation.w),
            ],
        )

    @staticmethod
    def _make_sensor_spec(uuid: str, cfg: SensorConfig):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = getattr(habitat_sim.SensorType, cfg.type)
        spec.resolution = [cfg.height, cfg.width]
        spec.hfov = cfg.hfov
        spec.position = cfg.position
        spec.orientation = [
            math.radians(cfg.pitch),
            math.radians(cfg.yaw),
            math.radians(cfg.roll),
        ]
        return spec
