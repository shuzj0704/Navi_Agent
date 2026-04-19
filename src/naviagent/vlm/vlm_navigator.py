"""
VLM Navigation Module
=====================
Send N-view images + task instruction to a Qwen-VL endpoint, parse output.

默认行为等价重构前 (3 视角 / 方向输出 / 语义文字 / 图片记忆长 8 / 决策+位姿记忆长 20)。
通过 AblationConfig 可切换到: 单视角 / 四视角 / pixel-goal 输出 / 语义图片输入 / 不同长度记忆。
"""

import numpy as np
import cv2
import base64
import math
import time
import os
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

from ..common.view_constants import (
    VIEW_ORDER, VIEW_ABBR, ABBR_TO_VIEW, VIEW_LABELS, wrap_angle,
)


# ---- 前视历史记忆采样阈值 (长度由 AblationConfig.image_memory_len 控制) ----
FRONT_MEMORY_MIN_DIST = 0.5            # m
FRONT_MEMORY_MIN_ANGLE = math.radians(20.0)


@dataclass
class AblationConfig:
    """System1 消融实验配置 (默认 = 当前主干行为)。"""

    # #3-5: 输入视角 (front/left/right/back 的子集)
    views: Tuple[str, ...] = ("front", "left", "right")

    # #1-2: 输出模式
    #   "direction" -> F / L / R / STOP
    #   "pixel"     -> v,vx,vy 像素目标
    output_mode: str = "direction"

    # #6-7: 语义图模态
    #   "none"  -> 不传
    #   "text"  -> 结构化文字 (机器人坐标系)
    #   "image" -> 俯视渲染图
    semantic_mode: str = "text"

    # #8 / #10: 历史图片记忆条数 (0 表示禁用)
    image_memory_len: int = 8

    # #8 / #9 / #11: 决策动作记忆 / 位姿记忆长度 (0 表示禁用)
    action_history_len: int = 20
    pose_history_len: int = 20

    def label(self) -> str:
        """生成简短 tag, 用于评测输出目录命名。"""
        return (
            f"v{len(self.views)}_{self.output_mode}_"
            f"sem-{self.semantic_mode}_"
            f"img{self.image_memory_len}_"
            f"act{self.action_history_len}_"
            f"pose{self.pose_history_len}"
        )


# ---- Prompt 模板 ----

_VIEW_DESC = {
    1: "上方一张图像是机器人前(f)相机拍摄",
    3: "上方三张图像分别由机器人的前(f)、左(l)、右(r)三个相机拍摄",
    4: "上方四张图像分别由机器人的前(f)、左(l)、右(r)、后(b)四个相机拍摄",
}


def _direction_prompt(n_views: int) -> str:
    desc = _VIEW_DESC.get(n_views, "上方图像是机器人相机拍摄")
    return (
        "\n\n你是一名专业的导航向导。" + desc +
        "，每张图像分辨率为 640x640 像素，原点 (0,0) 位于左上角。\n\n"
        "你的导航任务是: {instruction}\n\n"
        "请输出以下三种动作之一(且仅一种):\n"
        "  - F     : 沿当前朝向直行一步 (前视图中的可通行区域足够前进时)\n"
        "  - L     : 原地左转 (左侧视图更可能通往目标, 或前方受阻需要换方向)\n"
        "  - R     : 原地右转 (右侧视图更可能通往目标, 或前方受阻需要换方向)\n\n"
        "如果你认为已经完成当前任务要求(例如已抵达目标物前 1-2 米), 请输出: STOP\n\n"
        "严格按上述格式输出, 仅输出 F / L / R / STOP 中的一个, 不要任何其他内容、解释、引号或标点。"
    )


def _pixel_prompt(views: Tuple[str, ...]) -> str:
    abbrs = "/".join(VIEW_ABBR[v] for v in views)
    desc = _VIEW_DESC.get(len(views), "上方图像是机器人相机拍摄")
    return (
        "\n\n你是一名专业的导航向导。" + desc +
        "，每张图像分辨率为 640x640 像素，原点 (0,0) 位于左上角。\n\n"
        "你的导航任务是: {instruction}\n\n"
        "请选择前进方向，并在该方向视图中可通行区域(地面/门口/走廊)上挑选一个目标像素。\n\n"
        "严格按以下格式输出: v,vx,vy\n\n"
        f"其中 v 是 {abbrs} 之一，vx (0-639) 为水平像素坐标，vy (0-639) 为垂直像素坐标。\n\n"
        "如果你认为已经完成当前任务要求，请输出: STOP\n\n"
        "除此之外不要输出任何其他内容。"
    )


def _format_action_text(act):
    """把 (view, vx, vy) 渲染成 'L' / 'R' / 'F' / 'STOP' 或 'f,vx,vy'。"""
    if act is None:
        return "action=N/A"
    if not isinstance(act, tuple) or len(act) < 1:
        return f"action={act}"
    v = act[0]
    if v == "stop":
        return "action=STOP"
    if v == "left":
        return "action=L"
    if v == "right":
        return "action=R"
    if v == "front":
        if len(act) >= 3 and (act[1] != 0 or act[2] != 0):
            return f"action=f,{act[1]},{act[2]}"
        return "action=F"
    if len(act) >= 3:
        return f"action={VIEW_ABBR.get(v, v)},{act[1]},{act[2]}"
    return f"action={act}"


DEFAULT_VLM_API_URL = "http://10.100.0.1:8000/v1"
DEFAULT_VLM_API_KEY = "none"
DEFAULT_VLM_MODEL = "qwen3-vl"


class VLMNavigator:
    def __init__(self, api_url=None, api_key=None, model=None,
                 temperature=1.0, max_tokens=100,
                 config=None, ablation: Optional[AblationConfig] = None):
        """System 1 反应式 VLM。

        优先级: config > 显式参数 > 环境变量 > 模块默认值。

        Args:
            config: VLMEndpointConfig, 从 YAML 加载的配置 (api_url/api_key/model 等)。
            ablation: AblationConfig — 消融实验开关 (None 则用默认当前行为)。
        """
        if config is not None:
            api_url = config.api_url
            api_key = config.api_key
            model = config.model
            temperature = config.temperature
            max_tokens = config.max_tokens
            self._extra_body = config.extra_body
        else:
            api_url = api_url or os.environ.get(
                "VLM1_API_URL", DEFAULT_VLM_API_URL)
            api_key = api_key or os.environ.get(
                "VLM1_API_KEY", DEFAULT_VLM_API_KEY)
            model = model or os.environ.get(
                "VLM1_MODEL", DEFAULT_VLM_MODEL)
            self._extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        self.client = OpenAI(base_url=api_url, api_key=api_key)
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_latency = 0.0

        self.ablation = ablation or AblationConfig()

        # 合并决策+位姿进同一份列表, 渲染时按各自长度截取
        self._history_cap = max(
            self.ablation.action_history_len, self.ablation.pose_history_len
        )
        self.history: List[dict] = []      # [{step, view, pose}, ...]
        self.front_memory: List[dict] = []  # [{bgr, x, y, yaw, step, action}, ...]

    @property
    def output_mode(self) -> str:
        return self.ablation.output_mode

    def reset_history(self):
        """新 episode 时清空历史"""
        self.history = []
        self.front_memory = []

    def get_viz_state(self) -> dict:
        """返回可视化所需的状态, 不暴露内部对象。"""
        return {"history": list(self.history)}

    # ------------------------------------------------------------------
    #  主推理
    # ------------------------------------------------------------------

    def predict(self, images_dict, instruction, step=None, pose=None,
                semantic_objects=None, semantic_map=None,
                subtask_start_pose=None):
        """
        Args:
            images_dict: {"front": (H,W,3), ... } BGR uint8
            instruction: navigation task instruction
            pose: 可选 (nav_x, nav_y, nav_yaw)
            semantic_objects: 可选 list[Object3D] (semantic_mode="text" 时使用)
            semantic_map: 可选 俯视 BGR 图 (semantic_mode="image" 时使用)
            subtask_start_pose: 保留占位, 当前未使用
        Returns:
            (view, vx, vy) or None (parse failed)
        """
        abl = self.ablation
        views_used = tuple(abl.views)

        content = []

        # ---- 0. 历史前视图 ----
        if abl.image_memory_len > 0 and self.front_memory:
            shown = self.front_memory[-abl.image_memory_len:]
            content.append({
                "type": "text",
                "text": (
                    "这是你当前的任务执行进度,请自行判断执行到哪一步任务了。"
                    f"以下是过去 {len(shown)} 个位置的前视图"
                    "(按时间从早到晚排列, 每帧附带当时的模型输出与位姿), "
                    "仅供空间记忆与进度参考; 决策请以下面的当前视角为准。"
                ),
            })
            for i, entry in enumerate(shown):
                t_back = len(shown) - i
                act = entry.get("action")
                act_txt = _format_action_text(act)
                ex, ey, eyaw = entry["x"], entry["y"], entry["yaw"]
                meta = (
                    f"[历史前视 t-{t_back}] step={entry.get('step','?')} | "
                    f"pose=({ex:.2f},{ey:.2f}) yaw={math.degrees(eyaw):.0f}° | "
                    f"{act_txt}"
                )
                content.append({"type": "text", "text": meta})
                _, buf = cv2.imencode(
                    ".jpg", entry["bgr"], [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                b64 = base64.b64encode(buf).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

        # ---- 1. 当前视角 ----
        for vname in views_used:
            content.append({"type": "text", "text": VIEW_LABELS[vname]})
            img = images_dict[vname]
            if img.shape[0] != 640 or img.shape[1] != 640:
                img = cv2.resize(img, (640, 640))
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        # ---- 2. 语义地图 ----
        if abl.semantic_mode == "text" and semantic_objects and pose is not None:
            sem_text = self._format_semantic_objects(semantic_objects, pose)
            if sem_text:
                content.append({"type": "text", "text": sem_text})
        elif abl.semantic_mode == "image" and semantic_map is not None and semantic_map.size > 0:
            ok, buf = cv2.imencode(".jpg", semantic_map, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                b64 = base64.b64encode(buf).decode()
                content.append({
                    "type": "text",
                    "text": ("[语义地图，俯视视角] 绿点=机器人, 浅绿线=轨迹, "
                             "彩色框=检出物体; 用于参考全局布局和探索进度。"),
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

        # ---- 3. 动作 / 位姿历史 ----
        if self.history and (abl.action_history_len > 0 or abl.pose_history_len > 0):
            n = max(abl.action_history_len, abl.pose_history_len)
            shown = self.history[-n:] if n > 0 else []
            lines = [f"[最近 {len(shown)} 次决策 (旧→新)]"]
            # 最新 k 条包含什么
            act_start = len(shown) - abl.action_history_len if abl.action_history_len > 0 else len(shown)
            pose_start = len(shown) - abl.pose_history_len if abl.pose_history_len > 0 else len(shown)
            for i, h in enumerate(shown):
                parts = [f"- step={h.get('step','?')}"]
                if i >= pose_start:
                    hx, hy, hyaw = h.get("pose", (None, None, None))
                    if hx is not None:
                        parts.append(
                            f"pos=({hx:.2f},{hy:.2f}) yaw={math.degrees(hyaw):.0f}°"
                        )
                if i >= act_start:
                    act_txt = _format_action_text((h.get("view"), h.get("vx", 0), h.get("vy", 0)))
                    parts.append(act_txt)
                lines.append(" | ".join(parts))
            if pose is not None and abl.pose_history_len > 0:
                cx, cy, cyaw = pose
                lines.append(
                    f"[当前位姿] pos=({cx:.2f},{cy:.2f}) yaw={math.degrees(cyaw):.0f}°"
                )
            content.append({"type": "text", "text": "\n".join(lines)})

        # ---- 4. 任务指令 ----
        if abl.output_mode == "pixel":
            prompt_tpl = _pixel_prompt(views_used)
        else:
            prompt_tpl = _direction_prompt(len(views_used))
        current_step = step if step is not None else len(self.history)
        content.append({
            "type": "text",
            "text": f"[Step {current_step}] " + prompt_tpl.format(instruction=instruction),
        })

        messages = [{"role": "user", "content": content}]

        # ---- 5. 调用 API ----
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_body=self._extra_body,
            )
            raw = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[VLM] API call failed: {e}")
            return None
        finally:
            self.last_latency = time.time() - t0

        print(f"[VLM raw] {raw}")

        parsed = self._parse_response(raw, views_used)

        # 记录决策+位姿
        if parsed is not None:
            self._record_history(parsed, step=step, pose=pose)

        # 更新前视图记忆
        if pose is not None and abl.image_memory_len > 0 and "front" in images_dict:
            self._maybe_push_front_memory(images_dict["front"], pose, step, parsed)

        return parsed

    # ------------------------------------------------------------------
    #  辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _format_semantic_objects(objects, pose):
        """全局物体列表 → 机器人坐标系 (x=前 y=左) 文字。"""
        rx, ry, ryaw = pose
        cy, sy = math.cos(ryaw), math.sin(ryaw)
        lines = []
        for obj in objects:
            try:
                gx, gy = float(obj.center[0]), float(obj.center[1])
                label = obj.label
            except (AttributeError, IndexError, TypeError):
                continue
            dx, dy = gx - rx, gy - ry
            fwd = cy * dx + sy * dy
            left = -sy * dx + cy * dy
            lines.append(f"- {label}: (前={fwd:+.2f}m, 左={left:+.2f}m)")
        if not lines:
            return ""
        header = (
            f"[语义物体 — 机器人坐标系, x=前 y=左, 共 {len(lines)} 个] "
            "用于参考全局布局和已探索物体位置:"
        )
        return header + "\n" + "\n".join(lines)

    def _maybe_push_front_memory(self, front_bgr, pose, step, action=None):
        x, y, yaw = pose
        if self.front_memory:
            last = self.front_memory[-1]
            dx = x - last["x"]
            dy = y - last["y"]
            dist = math.hypot(dx, dy)
            ang = abs(wrap_angle(yaw - last["yaw"]))
            if dist < FRONT_MEMORY_MIN_DIST and ang < FRONT_MEMORY_MIN_ANGLE:
                return

        img = front_bgr
        if img.shape[0] != 640 or img.shape[1] != 640:
            img = cv2.resize(img, (640, 640))
        self.front_memory.append({
            "bgr": img.copy(),
            "x": float(x), "y": float(y), "yaw": float(yaw),
            "step": step,
            "action": action,
        })
        cap = self.ablation.image_memory_len
        if cap > 0 and len(self.front_memory) > cap:
            self.front_memory = self.front_memory[-cap:]

    def _parse_response(self, raw, views_used):
        """解析模型输出, 返回 (view, vx, vy) 或 None。

        direction 模式:  F/L/R/STOP -> ("front"|"left"|"right"|"stop", 0, 0)
        pixel 模式:     v,vx,vy     -> (view, vx, vy);  STOP / 0,0,0 -> ("stop", 0, 0)
        """
        import re

        token = raw.strip().strip("`'\"")

        # 先识别 STOP (两种模式共用)
        if re.search(r'\b(stop|done|finish(ed)?)\b', token, re.IGNORECASE):
            print("[VLM] Model decided: STOP")
            return "stop", 0, 0

        if self.ablation.output_mode == "pixel":
            m = re.search(r'([flrb])\s*,\s*(\d+)\s*,\s*(\d+)', token, re.IGNORECASE)
            if not m:
                print(f"[VLM] Parse failed (pixel): {raw}")
                return None
            abbr = m.group(1).lower()
            view = ABBR_TO_VIEW.get(abbr)
            if view is None or view not in views_used:
                print(f"[VLM] view '{abbr}' 不在可用视角 {views_used}: {raw}")
                return None
            try:
                vx = int(m.group(2))
                vy = int(m.group(3))
            except (ValueError, TypeError):
                print(f"[VLM] Invalid coords: {raw}")
                return None
            if vx == 0 and vy == 0:
                return "stop", 0, 0
            vx = max(0, min(vx, 639))
            vy = max(0, min(vy, 639))
            return view, vx, vy

        # direction 模式
        m = re.search(r'[a-zA-Z]+', token)
        if not m:
            print(f"[VLM] Parse failed: {raw}")
            return None
        word = m.group(0).lower()
        if word in ("f", "forward", "front", "go"):
            return "front", 0, 0
        if word in ("l", "left"):
            return "left", 0, 0
        if word in ("r", "right"):
            return "right", 0, 0
        print(f"[VLM] Parse failed: {raw}")
        return None

    def _record_history(self, parsed, step=None, pose=None):
        view, vx, vy = parsed
        entry = {
            "step": step if step is not None else len(self.history),
            "view": view,
            "vx": vx,
            "vy": vy,
        }
        if pose is not None:
            entry["pose"] = (float(pose[0]), float(pose[1]), float(pose[2]))
        self.history.append(entry)
        if self._history_cap > 0 and len(self.history) > self._history_cap:
            self.history = self.history[-self._history_cap:]


# ----------------------------------------------------------------------
#  异步 VLM 工作器
# ----------------------------------------------------------------------

class VLMAsyncWorker:
    """把 VLMNavigator.predict 挂到单工作线程后台。"""

    def __init__(self, vlm):
        self.vlm = vlm
        self._exec = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vlm1"
        )

    def submit(self, views_bgr, instruction, step, pose=None,
               semantic_objects=None, semantic_map=None,
               subtask_start_pose=None):
        snapshot = {k: v.copy() for k, v in views_bgr.items()}
        sem_copy = list(semantic_objects) if semantic_objects else None
        sem_map_copy = semantic_map.copy() if semantic_map is not None else None
        return self._exec.submit(
            self._call, snapshot, instruction, step, pose, sem_copy,
            sem_map_copy, subtask_start_pose,
        )

    def _call(self, views_bgr, instruction, step, pose, semantic_objects,
              semantic_map, subtask_start_pose):
        if self.vlm is None:
            return ("front", 320, 240)
        return self.vlm.predict(
            views_bgr, instruction, step=step, pose=pose,
            semantic_objects=semantic_objects,
            semantic_map=semantic_map,
            subtask_start_pose=subtask_start_pose,
        )

    def shutdown(self):
        self._exec.shutdown(wait=False)
