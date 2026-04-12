"""
FastAPI 仿真服务端
==================
HTTP REST API 封装 Habitat 仿真器。
"""

import os
import json
import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

from .config import load_config, ServerConfig
from .habitat_backend import HabitatBackend
from .encoding import encode_rgb, encode_depth


# ========== 请求模型 ==========

class SceneRequest(BaseModel):
    scene_name: Optional[str] = None
    scene_path: Optional[str] = None

class ActionRequest(BaseModel):
    action: str

class ActionsRequest(BaseModel):
    actions: list

class AgentStateRequest(BaseModel):
    position: list
    rotation: Optional[list] = None


# ========== 应用工厂 ==========

def create_app(config_path: str = "config/sim_server.yaml") -> FastAPI:
    """创建 FastAPI 应用, 加载配置, 初始化后端。"""
    config = load_config(config_path)
    app = FastAPI(title="VLN Sim Server")
    app.state.config = config
    app.state.backend = HabitatBackend(config)

    @app.on_event("startup")
    async def startup():
        if config.default_scene:
            app.state.backend.load_scene(scene_name=config.default_scene)
            print(f"[SimServer] Default scene loaded: {config.default_scene}")

    @app.on_event("shutdown")
    async def shutdown():
        app.state.backend.close()

    # ========== 健康检查 ==========

    @app.get("/health")
    async def health():
        info = app.state.backend.get_scene_info()
        return {
            "status": "ok",
            "scene_loaded": info is not None,
            "scene_name": info["scene_name"] if info else None,
        }

    # ========== 场景管理 ==========

    @app.get("/scenes")
    async def list_scenes():
        scenes = app.state.backend.list_scenes()
        return {"scenes": scenes}

    @app.get("/scene")
    async def get_scene():
        info = app.state.backend.get_scene_info()
        if info is None:
            raise HTTPException(404, "No scene loaded")
        return info

    @app.post("/scene")
    async def load_scene(req: SceneRequest):
        if req.scene_name is None and req.scene_path is None:
            raise HTTPException(400, "Provide scene_name or scene_path")
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: app.state.backend.load_scene(
                    scene_name=req.scene_name,
                    scene_path=req.scene_path,
                ),
            )
            return result
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except Exception as e:
            raise HTTPException(500, str(e))

    # ========== 观测 ==========

    @app.get("/obs")
    async def get_observations():
        """获取所有传感器观测 + agent 状态。

        返回 multipart/mixed: 第一部分 JSON 元数据, 后续部分为二进制传感器数据。
        """
        loop = asyncio.get_event_loop()
        try:
            obs, agent_state = await loop.run_in_executor(
                None, app.state.backend.get_observations
            )
        except RuntimeError as e:
            raise HTTPException(400, str(e))

        enc = config.encoding
        boundary = "frame"

        # 构建元数据
        sensors_meta = {}
        parts = []

        for name, scfg in config.sensors.items():
            data = obs[name]
            if scfg.type == "COLOR":
                encoded, ct = encode_rgb(data, enc.rgb_format, enc.rgb_jpeg_quality)
                sensors_meta[name] = {
                    "width": scfg.width, "height": scfg.height,
                    "channels": 3, "dtype": "uint8",
                    "encoding": enc.rgb_format,
                }
            else:  # DEPTH
                raw = data
                if raw.ndim == 3:
                    raw = raw[:, :, 0]
                encoded, ct = encode_depth(raw, enc.depth_format)
                dtype_str = "float16" if enc.depth_format == "raw_f16" else "float32"
                sensors_meta[name] = {
                    "width": scfg.width, "height": scfg.height,
                    "channels": 1, "dtype": dtype_str,
                    "encoding": enc.depth_format,
                }
            parts.append((name, encoded, ct))

        metadata = {
            "agent_state": {
                "position": agent_state.position,
                "rotation": agent_state.rotation,
            },
            "sensors": sensors_meta,
        }

        # 构建 multipart body
        body_parts = []
        # Part 0: JSON metadata
        meta_bytes = json.dumps(metadata).encode()
        body_parts.append(
            f"--{boundary}\r\n"
            f"Content-Type: application/json\r\n"
            f"\r\n".encode() + meta_bytes + b"\r\n"
        )
        # Part 1..N: sensor data
        for name, data_bytes, ct in parts:
            header = (
                f"--{boundary}\r\n"
                f"Content-Disposition: attachment; name=\"{name}\"\r\n"
                f"Content-Type: {ct}\r\n"
                f"\r\n"
            ).encode()
            body_parts.append(header + data_bytes + b"\r\n")

        body_parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(body_parts)

        return Response(
            content=body,
            media_type=f"multipart/mixed; boundary={boundary}",
        )

    @app.get("/obs/{sensor_name}")
    async def get_single_obs(sensor_name: str):
        """获取单个传感器观测。"""
        loop = asyncio.get_event_loop()
        try:
            data = await loop.run_in_executor(
                None,
                lambda: app.state.backend.get_single_observation(sensor_name),
            )
        except (RuntimeError, KeyError) as e:
            raise HTTPException(400, str(e))

        scfg = config.sensors.get(sensor_name)
        if scfg is None:
            raise HTTPException(404, f"Unknown sensor: {sensor_name}")

        enc = config.encoding
        if scfg.type == "COLOR":
            encoded, ct = encode_rgb(data, enc.rgb_format, enc.rgb_jpeg_quality)
        else:
            raw = data
            if raw.ndim == 3:
                raw = raw[:, :, 0]
            encoded, ct = encode_depth(raw, enc.depth_format)

        return Response(
            content=encoded,
            media_type=ct,
            headers={
                "X-Width": str(scfg.width),
                "X-Height": str(scfg.height),
                "X-Dtype": "uint8" if scfg.type == "COLOR" else "float32",
                "X-Channels": "3" if scfg.type == "COLOR" else "1",
            },
        )

    # ========== 动作 ==========

    @app.post("/action")
    async def execute_action(req: ActionRequest):
        loop = asyncio.get_event_loop()
        try:
            state = await loop.run_in_executor(
                None, lambda: app.state.backend.execute_action(req.action)
            )
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return {
            "ok": True,
            "agent_state": {
                "position": state.position,
                "rotation": state.rotation,
            },
        }

    @app.post("/actions")
    async def execute_actions(req: ActionsRequest):
        loop = asyncio.get_event_loop()
        try:
            state = await loop.run_in_executor(
                None, lambda: app.state.backend.execute_actions(req.actions)
            )
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return {
            "ok": True,
            "n_executed": len(req.actions),
            "agent_state": {
                "position": state.position,
                "rotation": state.rotation,
            },
        }

    # ========== Agent 状态 ==========

    @app.get("/agent")
    async def get_agent():
        loop = asyncio.get_event_loop()
        try:
            state = await loop.run_in_executor(
                None, app.state.backend.get_agent_state
            )
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return {
            "position": state.position,
            "rotation": state.rotation,
        }

    @app.put("/agent")
    async def set_agent(req: AgentStateRequest):
        loop = asyncio.get_event_loop()
        try:
            state = await loop.run_in_executor(
                None,
                lambda: app.state.backend.set_agent_state(
                    req.position, req.rotation
                ),
            )
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return {
            "ok": True,
            "position": state.position,
            "rotation": state.rotation,
        }

    return app
