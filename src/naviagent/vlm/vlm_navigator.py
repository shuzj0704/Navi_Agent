"""
VLM Navigation Module
=====================
Send 4-view images + task instruction to Qwen3.0-VL, parse output (view, vx, vy).

Usage:
    vlm = VLMNavigator(api_url="http://10.100.0.1:8000/v1")
    result = vlm.predict(images_dict, "Go to the sofa in the living room")
    # result = ("front", 320, 200) or None
"""

import numpy as np
import cv2
import base64
import math
import time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI


import os

from ..common.view_constants import VIEW_ORDER, VIEW_ABBR, ABBR_TO_VIEW, VIEW_LABELS, wrap_angle


# ---- 前视历史记忆阈值 ----
FRONT_MEMORY_MAX_LEN = 4
FRONT_MEMORY_MIN_DIST = 0.5            # m
FRONT_MEMORY_MIN_ANGLE = math.radians(20.0)

TASK_PROMPT = """

你是一名专业的导航向导。上方三张图像分别由机器人的前(f)、左(l)、右(r)三个相机拍摄，每张图像分辨率为 640x480 像素，原点 (0,0) 位于左上角。

你的导航任务是: {instruction}

请选择前进方向，并在该方向视图中可通行区域(地面/门口/走廊)上挑选一个目标像素。

严格按以下格式输出: v,vx,vy

其中 v 是 f/l/r 之一，vx (0-639) 为水平像素坐标，vy (0-479) 为垂直像素坐标。

如果你认为已经完成当前任务要求，请输出: v,0,0

除此之外不要输出任何其他内容。"""


DEFAULT_VLM_API_URL = "http://10.100.0.1:8000/v1"
DEFAULT_VLM_API_KEY = "none"
DEFAULT_VLM_MODEL = "qwen3-vl"


class VLMNavigator:
    def __init__(self, api_url=None, api_key=None, model=None,
                 temperature=1.0, max_tokens=100, history_len=1,
                 config=None):
        """System 1 反应式 VLM。

        优先级: config > 显式参数 > 环境变量 > 模块默认值。

        Args:
            config: VLMEndpointConfig, 从 YAML 加载的配置。
                    传入后 api_url/api_key/model/temperature/max_tokens 从中取值。
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
        self.history_len = history_len
        self.history = []  # [{step, view, vx, vy}, ...]
        # 前视历史记忆: 最近 FRONT_MEMORY_MAX_LEN 帧, 相邻帧需满足位移或角度阈值。
        # 每项: {"bgr": HxWx3 uint8, "x": float, "y": float, "yaw": float, "step": int}
        self.front_memory = []

    def reset_history(self):
        """新 episode 时清空历史"""
        self.history = []
        self.front_memory = []

    def get_viz_state(self) -> dict:
        """返回可视化所需的状态, 不暴露内部对象。"""
        return {"history": list(self.history)}

    def predict(self, images_dict, instruction, step=None, pose=None,
                semantic_objects=None, subtask_start_pose=None):
        """
        Args:
            images_dict: {"front": (H,W,3), "left": ..., "right": ...} BGR uint8
            instruction: navigation task instruction
            pose: 可选 (nav_x, nav_y, nav_yaw) — 用于前视历史记忆的增量判定
            semantic_objects: 可选 list[Object3D] 全局物体列表; 将按当前 pose
                转换为机器人坐标系下的结构化文字传入 (物体类型 + 位置)
            subtask_start_pose: 可选 (x, y, yaw) 当前子任务下发时的起始位姿
        Returns:
            (view, vx, vy) or None (parse failed)
        """
        content = []

        # 0. 前视历史记忆 (旧 → 新), 放在当前四视角之前
        if self.front_memory:
            content.append({
                "type": "text",
                "text": (
                    "这是你当前的任务执行进度,请自行判断执行到哪一步任务了。"
                    f"以下是过去 {len(self.front_memory)} 个位置的前视图"
                    "(按时间从早到晚排列, 每帧附带当时的模型输出与位姿), "
                    "仅供空间记忆与进度参考; 决策请以下面的当前四视角为准。"
                ),
            })
            for i, entry in enumerate(self.front_memory):
                t_back = len(self.front_memory) - i
                act = entry.get("action")
                if act is None:
                    act_txt = "action=N/A"
                elif isinstance(act, tuple) and act[0] == "stop":
                    act_txt = "action=stop"
                else:
                    v, vx, vy = act
                    act_txt = f"action={VIEW_ABBR[v]},{vx},{vy}"
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

        # 1. 当前四视角
        for vname in VIEW_ORDER:
            content.append({"type": "text", "text": VIEW_LABELS[vname]})
            img = images_dict[vname]
            if img.shape[0] != 480 or img.shape[1] != 640:
                img = cv2.resize(img, (640, 480))
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        # 2. 语义物体列表 (以机器人坐标系位置的结构化文字传入)
        if semantic_objects and pose is not None:
            sem_text = self._format_semantic_objects(semantic_objects, pose)
            if sem_text:
                content.append({"type": "text", "text": sem_text})

        # 3. 位姿上下文: 子任务起始位姿 vs 当前位姿
        if subtask_start_pose is not None and pose is not None:
            sx, sy, syaw = subtask_start_pose
            cx, cy, cyaw = pose
            delta_dist = math.hypot(cx - sx, cy - sy)
            delta_yaw = math.degrees(wrap_angle(cyaw - syaw))
            content.append({"type": "text", "text": (
                f"当前子任务起始位姿: pos=({sx:.2f},{sy:.2f}) yaw={math.degrees(syaw):.0f}°\n"
                f"当前位姿: pos=({cx:.2f},{cy:.2f}) yaw={math.degrees(cyaw):.0f}°\n"
                f"本子任务内变化: 移动{delta_dist:.2f}m, 转向{delta_yaw:+.0f}°"
            )})

        # 4. 任务指令 (加入步数打破 KV cache 前缀)
        current_step = step if step is not None else len(self.history)
        content.append({"type": "text", "text": f"[Step {current_step}] " + TASK_PROMPT.format(instruction=instruction)})

        messages = [{"role": "user", "content": content}]

        # 4. 调用 API
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

        parsed = self._parse_response(raw)

        # 5. 若提供 pose 且本帧相对上一记忆帧满足阈值, 把当前前视+动作存进 front_memory
        if pose is not None:
            self._maybe_push_front_memory(images_dict["front"], pose, step, parsed)

        return parsed

    @staticmethod
    def _format_semantic_objects(objects, pose):
        """把全局物体列表转成机器人坐标系下的结构化文字。

        机器人坐标系: x=前, y=左 (单位 m)。
        """
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
                return  # 太近也没转, 跳过

        img = front_bgr
        if img.shape[0] != 480 or img.shape[1] != 640:
            img = cv2.resize(img, (640, 480))
        self.front_memory.append({
            "bgr": img.copy(),
            "x": float(x), "y": float(y), "yaw": float(yaw),
            "step": step,
            "action": action,
        })
        if len(self.front_memory) > FRONT_MEMORY_MAX_LEN:
            self.front_memory = self.front_memory[-FRONT_MEMORY_MAX_LEN:]

    def _parse_response(self, raw):
        """Parse VLM output 'v,vx,vy' to (view, vx, vy), or 'stop' to ("stop", 0, 0)"""
        import re

        token = raw.strip().lower()
        if token == "stop":
            print("[VLM] Model decided: STOP (target reached)")
            return "stop", 0, 0

        m = re.search(r'([flrb])\s*,\s*(\d+)\s*,\s*(\d+)', raw)
        if not m:
            print(f"[VLM] Parse failed: {raw}")
            return None

        abbr = m.group(1)
        view = ABBR_TO_VIEW[abbr]

        try:
            vx = int(m.group(2))
            vy = int(m.group(3))
        except (ValueError, TypeError):
            print(f"[VLM] Invalid coordinates: {raw}")
            return None

        # v,0,0 表示到达目标
        if vx == 0 and vy == 0:
            print(f"[VLM] Model decided: STOP ({abbr},0,0)")
            return "stop", 0, 0

        # clamp to valid range
        vx = max(0, min(vx, 639))
        vy = max(0, min(vy, 479))

        # record history
        self.history.append({
            "step": len(self.history),
            "view": view, "vx": vx, "vy": vy,
        })
        if len(self.history) > self.history_len:
            self.history = self.history[-self.history_len:]

        return view, vx, vy

    # ------------------------------------------------------------------
    #  诊断模式
    # ------------------------------------------------------------------

    DIAG_PROMPT = """上方图像是导航机器人三个方向的相机视图。

导航任务: {instruction}

请完成以下工作:
1. 简要描述你在每张图像中看到的内容(前/左/右)，每个 1-2 句
2. 分析哪个方向最有可能通往其他房间或任务目标，并说明原因
3. 给出你的导航决策

按以下格式输出:
=== 场景分析 ===
front: ...
left: ...
right: ...

=== 决策推理 ===
...

=== 决策 ===
{{"view": "...", "vx": ..., "vy": ...}}"""

    def diagnose(self, images_dict, instruction, step, save_dir):
        """
        诊断调用：发送相同图片，让模型输出详细分析 + 决策。
        结果保存到 save_dir/step_XXXX_diag.txt + 四张原图。

        Args:
            images_dict: {"front": ..., "left": ..., "right": ...} BGR
            instruction: 任务指令
            step: 当前步数
            save_dir: 保存目录
        Returns:
            (raw_analysis, parsed_result) — parsed_result 可能为 None
        """
        os.makedirs(save_dir, exist_ok=True)

        # 构造带标签的图片内容 (和 predict 一样)
        content = []
        for vname in VIEW_ORDER:
            content.append({"type": "text", "text": VIEW_LABELS[vname]})
            img = images_dict[vname]
            if img.shape[0] != 480 or img.shape[1] != 640:
                img = cv2.resize(img, (640, 480))
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
            # 保存原图
            cv2.imwrite(os.path.join(save_dir, f"step_{step:04d}_{vname}.jpg"), img)

        # 诊断 prompt（要求详细分析）
        content.append({
            "type": "text",
            "text": self.DIAG_PROMPT.format(instruction=instruction)
        })

        messages = [{"role": "user", "content": content}]

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=self.temperature,
                extra_body=self._extra_body,
            )
            raw = resp.choices[0].message.content.strip()
        except Exception as e:
            raw = f"[API Error] {e}"

        # 保存诊断文本
        diag_path = os.path.join(save_dir, f"step_{step:04d}_diag.txt")
        with open(diag_path, "w", encoding="utf-8") as f:
            f.write(f"Step: {step}\n")
            f.write(f"Instruction: {instruction}\n")
            f.write(f"{'='*60}\n")
            f.write(raw)
            f.write("\n")

        # 尝试从诊断结果中解析决策
        parsed = self._parse_response(raw)

        print(f"[DIAG step {step}] Saved → {diag_path}")

        return raw, parsed


# ----------------------------------------------------------------------
#  异步 VLM 工作器
# ----------------------------------------------------------------------

class VLMAsyncWorker:
    """把 VLMNavigator.predict 挂到单工作线程后台, 让主循环非阻塞地跑 DWA/sim。

    用法:
        worker = VLMAsyncWorker(vlm)
        fut = worker.submit(views_bgr, instruction, step)
        ...
        if fut.done():
            prediction = fut.result()  # (view, vx, vy) 或 None
    """

    def __init__(self, vlm):
        self.vlm = vlm
        self._exec = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vlm1"
        )

    def submit(self, views_bgr, instruction, step, pose=None,
               semantic_objects=None, subtask_start_pose=None):
        snapshot = {k: v.copy() for k, v in views_bgr.items()}
        sem_copy = list(semantic_objects) if semantic_objects else None
        return self._exec.submit(
            self._call, snapshot, instruction, step, pose, sem_copy,
            subtask_start_pose,
        )

    def _call(self, views_bgr, instruction, step, pose, semantic_objects,
              subtask_start_pose):
        if self.vlm is None:
            return ("front", 320, 240)
        return self.vlm.predict(
            views_bgr, instruction, step=step, pose=pose,
            semantic_objects=semantic_objects,
            subtask_start_pose=subtask_start_pose,
        )

    def shutdown(self):
        self._exec.shutdown(wait=False)
