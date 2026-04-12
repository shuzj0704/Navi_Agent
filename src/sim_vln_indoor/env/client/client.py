"""
仿真客户端
==========
HTTP 客户端封装, 对接 sim_server。
返回 numpy 数组和兼容的 AgentState, 可直接传给 coordinate_transform 函数。
"""

import json
import time
import numpy as np
import cv2
import httpx
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class AgentState:
    """兼容 habitat_sim agent state 的数据类。

    coordinate_transform 中的 habitat_pos_to_nav2d(state.position) 和
    habitat_quat_to_yaw(state.rotation) 可直接使用。
    """
    position: np.ndarray          # [x, y, z] float64
    rotation_xyzw: np.ndarray     # [qx, qy, qz, qw] float64

    @property
    def rotation(self):
        """返回与 habitat quaternion 兼容的对象 (有 x, y, z, w 属性)。"""
        return _QuatCompat(self.rotation_xyzw)


class _QuatCompat:
    """轻量四元数包装, 提供 .x .y .z .w 属性供 habitat_quat_to_yaw() 使用。"""
    __slots__ = ("x", "y", "z", "w")

    def __init__(self, xyzw):
        self.x = float(xyzw[0])
        self.y = float(xyzw[1])
        self.z = float(xyzw[2])
        self.w = float(xyzw[3])


class SimClient:
    """仿真服务器 HTTP 客户端。"""

    def __init__(self, base_url: str = "http://localhost:5100",
                 timeout: float = 30.0):
        self._url = base_url.rstrip("/")
        self._session = httpx.Client(timeout=timeout)

    # ---- 场景管理 ----

    def load_scene(self, scene_name: str = None,
                   scene_path: str = None) -> dict:
        body = {}
        if scene_name is not None:
            body["scene_name"] = scene_name
        if scene_path is not None:
            body["scene_path"] = scene_path
        resp = self._session.post(f"{self._url}/scene", json=body)
        resp.raise_for_status()
        return resp.json()

    def get_scene(self) -> dict:
        resp = self._session.get(f"{self._url}/scene")
        resp.raise_for_status()
        return resp.json()

    def list_scenes(self) -> List[dict]:
        resp = self._session.get(f"{self._url}/scenes")
        resp.raise_for_status()
        return resp.json()["scenes"]

    # ---- 观测 ----

    def get_observations(self) -> Tuple[Dict[str, np.ndarray], AgentState]:
        """获取所有传感器观测 + agent 状态。

        Returns:
            (obs_dict, agent_state)
            obs_dict: sensor_name -> numpy array
              RGB: (H, W, 3) uint8 BGR (已从 RGB 转换, 与 cv2 约定一致)
              Depth: (H, W) float32
            agent_state: AgentState, 兼容 habitat_pos_to_nav2d / habitat_quat_to_yaw
        """
        resp = self._session.get(f"{self._url}/obs")
        resp.raise_for_status()

        # 解析 multipart/mixed
        content_type = resp.headers.get("content-type", "")
        boundary = self._extract_boundary(content_type)
        parts = self._parse_multipart(resp.content, boundary)

        if not parts:
            raise RuntimeError("Empty multipart response from /obs")

        # Part 0: JSON metadata
        meta_name, meta_data, meta_ct = parts[0]
        metadata = json.loads(meta_data)
        agent_data = metadata["agent_state"]
        sensors_meta = metadata["sensors"]

        # 构建 AgentState
        agent_state = AgentState(
            position=np.array(agent_data["position"], dtype=np.float64),
            rotation_xyzw=np.array(agent_data["rotation"], dtype=np.float64),
        )

        # Part 1..N: 传感器数据
        obs_dict = {}
        for part_name, part_data, part_ct in parts[1:]:
            if part_name not in sensors_meta:
                continue
            smeta = sensors_meta[part_name]
            if smeta["dtype"] == "uint8":
                # JPEG/PNG → BGR via cv2.imdecode
                buf = np.frombuffer(part_data, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                obs_dict[part_name] = img  # BGR
            else:
                # Raw depth
                np_dtype = np.float16 if smeta["dtype"] == "float16" else np.float32
                arr = np.frombuffer(part_data, dtype=np_dtype)
                obs_dict[part_name] = arr.reshape(
                    smeta["height"], smeta["width"]
                ).astype(np.float32)

        return obs_dict, agent_state

    # ---- 动作 ----

    def act(self, action: str) -> AgentState:
        resp = self._session.post(
            f"{self._url}/action", json={"action": action}
        )
        resp.raise_for_status()
        data = resp.json()["agent_state"]
        return self._to_agent_state(data)

    def act_many(self, actions: List[str]) -> AgentState:
        resp = self._session.post(
            f"{self._url}/actions", json={"actions": actions}
        )
        resp.raise_for_status()
        data = resp.json()["agent_state"]
        return self._to_agent_state(data)

    # ---- Agent 状态 ----

    def get_agent_state(self) -> AgentState:
        resp = self._session.get(f"{self._url}/agent")
        resp.raise_for_status()
        return self._to_agent_state(resp.json())

    def set_agent_state(self, position, rotation=None) -> AgentState:
        body = {"position": list(position)}
        if rotation is not None:
            body["rotation"] = list(rotation)
        resp = self._session.put(f"{self._url}/agent", json=body)
        resp.raise_for_status()
        return self._to_agent_state(resp.json())

    # ---- 生命周期 ----

    def close(self):
        self._session.close()

    def wait_ready(self, timeout: float = 30.0):
        """轮询 /health 直到服务器就绪。"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                resp = self._session.get(f"{self._url}/health")
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise TimeoutError(f"SimServer not ready after {timeout}s")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ---- 内部方法 ----

    @staticmethod
    def _to_agent_state(data: dict) -> AgentState:
        return AgentState(
            position=np.array(data["position"], dtype=np.float64),
            rotation_xyzw=np.array(data["rotation"], dtype=np.float64),
        )

    @staticmethod
    def _extract_boundary(content_type: str) -> str:
        for part in content_type.split(";"):
            part = part.strip()
            if part.startswith("boundary="):
                return part.split("=", 1)[1].strip('"')
        raise ValueError(f"No boundary in content-type: {content_type}")

    @staticmethod
    def _parse_multipart(body: bytes, boundary: str) -> list:
        """解析 multipart/mixed body, 返回 [(name, data, content_type), ...]"""
        sep = f"--{boundary}".encode()
        end = f"--{boundary}--".encode()

        parts = []
        chunks = body.split(sep)

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk or chunk == b"--" or chunk.startswith(end.split(sep[-1:])[0] if len(sep) > 0 else b""):
                continue

            # 跳过结束标记
            if chunk == b"--":
                continue

            # 分离头和体
            header_end = chunk.find(b"\r\n\r\n")
            if header_end < 0:
                header_end = chunk.find(b"\n\n")
                if header_end < 0:
                    continue
                header_bytes = chunk[:header_end]
                data = chunk[header_end + 2:]
            else:
                header_bytes = chunk[:header_end]
                data = chunk[header_end + 4:]

            # 去掉尾部的 \r\n
            if data.endswith(b"\r\n"):
                data = data[:-2]

            # 解析头
            headers = {}
            for line in header_bytes.decode("utf-8", errors="replace").split("\n"):
                line = line.strip()
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            ct = headers.get("content-type", "application/octet-stream")
            name = ""
            disp = headers.get("content-disposition", "")
            if 'name="' in disp:
                name = disp.split('name="')[1].split('"')[0]

            parts.append((name, data, ct))

        return parts
