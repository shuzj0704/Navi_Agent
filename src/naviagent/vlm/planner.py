"""
System 2 Planner
================
慢思考规划器。把当前任务、已完成子任务、当前子任务、最新语义地图图片一起
发给 Qwen3.0-VL(启用 thinking),返回一个 PlanDecision:
  status = continue | advance | done
  next_subtask = "..."
  reason = "..."

运行在独立线程池(max_workers=1),主循环通过 submit() 得到 Future,
非阻塞地 poll future.done() 或阻塞地 future.result() 取结果。
"""

import base64
import json
import math
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from openai import OpenAI

from ..common.view_constants import VIEW_LABELS, VIEW_ORDER, wrap_angle


# ---- System 2 前视历史记忆阈值 ----
PLANNER_FRONT_MEMORY_MAX_LEN = 4
PLANNER_FRONT_MEMORY_MIN_DIST = 0.5            # m
PLANNER_FRONT_MEMORY_MIN_ANGLE = math.radians(30.0)


PLANNER_SYSTEM_PROMPT = """你是室内导航机器人的高层规划器。把自然语言指令拆成简短、具体、可视觉验证的子任务,每次只输出一条,并判断当前子任务应当 continue / advance / done。只输出 JSON,不要任何其他文字。"""


PLANNER_USER_PROMPT_TEMPLATE = """整体任务: {full_instruction}
已完成: {completed}
当前子任务: {current}{hint}

图像: 前(f)/左(l)/右(r)/后(b) 四张相机视图 + 俯视语义地图(绿点=机器人, 浅绿线=轨迹, 彩色框=检出物体)。
{mapped_classes_note}

规则:
- 当前子任务完成 → advance, 给出下一条单一动作。
- 当前子任务仍是原始整句、或包含多个动作("然后"/逗号分隔) → 必须 advance, 只返回第一步。
- 整体目标物体在任一视图中清晰可见且距离约 1-2 米 或者 机器人在语义地图中和目标物很接近 → done。
- 因 STOP 触发(见 NOTE): 看历史帧前视图和当前帧前视图或者语义图, 看到目标且距离近 → done。
- 其余情况, 若 System 1 在有效探索 → continue。
- 当前子任务无法完成或不合理，就切换至新的子任务。

输出(单行 JSON, 无代码块, 无多余文字):
{{"status": "continue"|"advance"|"done", "next_subtask": "...", "reason": "..."}}
- next_subtask 和 reason 一律中文。
- advance 时 next_subtask 必须非空, 其它情况留空字符串。
- next_subtask 是具体的导航指令(例如"找到通往厨房的门口"、"环视寻找电视")。"""


@dataclass
class PlanDecision:
    status: str                       # "continue" | "advance" | "done"
    next_subtask: str = ""
    reason: str = ""
    raw: str = ""
    latency_sec: float = 0.0
    error: Optional[str] = None

    @classmethod
    def safe_default(cls, reason: str, raw: str = "", latency: float = 0.0,
                     error: Optional[str] = None) -> "PlanDecision":
        return cls(status="continue", reason=reason, raw=raw,
                   latency_sec=latency, error=error)


class System2Planner:
    """后台线程池封装的 Qwen3.0-VL 规划调用。"""

    def __init__(self,
                 api_url: str = "http://192.168.1.137:8000/v1",
                 model: str = "qwen3-vl",
                 temperature: float = 0.3,
                 max_tokens: int = 2048,
                 enable_thinking: bool = True,
                 jpeg_quality: int = 90,
                 mapped_classes: Optional[List[str]] = None):
        self.client = OpenAI(base_url=api_url, api_key="none")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.jpeg_quality = jpeg_quality
        self.mapped_classes = list(mapped_classes) if mapped_classes else None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="system2-planner"
        )
        # 前视历史记忆: 最近 PLANNER_FRONT_MEMORY_MAX_LEN 帧, 增量阈值同 System 1
        # 每项: {"bgr": HxWx3 uint8, "x": float, "y": float, "yaw": float, "step": int}
        self.front_memory: List[dict] = []

    def submit(self,
               full_instruction: str,
               current_subtask: str,
               completed_subtasks: List[str],
               semantic_map_bgr: np.ndarray,
               views_bgr: Optional[dict] = None,
               hint: Optional[str] = None,
               pose: Optional[Tuple[float, float, float]] = None,
               ) -> "Future[PlanDecision]":
        """非阻塞提交一次规划调用,返回 Future。

        views_bgr: {"front": BGR(H,W,3), "left": ..., "right": ..., "back": ...}
        pose:      (nav_x, nav_y, nav_yaw), 用于前视历史记忆的增量判定。
        所有图像在派发时复制一份,避免主线程后续覆盖影响后台调用。
        """
        copied_views = (
            {k: v.copy() for k, v in views_bgr.items() if v is not None}
            if views_bgr else None
        )
        return self._executor.submit(
            self._call,
            full_instruction, current_subtask,
            list(completed_subtasks),
            semantic_map_bgr.copy() if semantic_map_bgr is not None else None,
            copied_views,
            hint,
            pose,
        )

    def shutdown(self):
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    #  后台执行
    # ------------------------------------------------------------------

    def _call(self,
              full_instruction: str,
              current_subtask: str,
              completed_subtasks: List[str],
              semantic_map_bgr: Optional[np.ndarray],
              views_bgr: Optional[dict],
              hint: Optional[str],
              pose: Optional[Tuple[float, float, float]]) -> PlanDecision:
        t0 = time.time()

        if semantic_map_bgr is None or semantic_map_bgr.size == 0:
            return PlanDecision.safe_default(
                reason="no_semantic_map", latency=time.time() - t0,
                error="semantic_map image is empty",
            )

        try:
            content = self._build_content(
                full_instruction, current_subtask,
                completed_subtasks, views_bgr, semantic_map_bgr, hint,
            )
        except Exception as e:
            return PlanDecision.safe_default(
                reason="build_content_failed", latency=time.time() - t0,
                error=str(e),
            )

        # 本帧 front 喂进 memory (按阈值增量)
        if pose is not None and views_bgr is not None:
            front_bgr = views_bgr.get("front")
            if front_bgr is not None:
                self._maybe_push_front_memory(front_bgr, pose)

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": self.enable_thinking
                    }
                },
            )
            raw = resp.choices[0].message.content.strip()
        except Exception as e:
            return PlanDecision.safe_default(
                reason="api_error", latency=time.time() - t0, error=str(e),
            )

        latency = time.time() - t0
        return self._parse_response(raw, latency)

    # ------------------------------------------------------------------
    #  消息构造
    # ------------------------------------------------------------------

    def _build_content(self,
                       full_instruction: str,
                       current_subtask: str,
                       completed_subtasks: List[str],
                       views_bgr: Optional[dict],
                       semantic_map_bgr: np.ndarray,
                       hint: Optional[str]) -> list:
        content: list = []

        # 0. 前视历史记忆 (旧 → 新), 仅供空间记忆参考
        if self.front_memory:
            content.append({
                "type": "text",
                "text": (
                    f"以下是过去 {len(self.front_memory)} 个位置的前视图"
                    f"(按时间从早到晚排列), 仅供空间记忆参考; 决策请以下面的"
                    f"当前四视角和语义地图为准。"
                ),
            })
            for i, entry in enumerate(self.front_memory):
                t_back = len(self.front_memory) - i
                content.append({
                    "type": "text",
                    "text": f"[历史前视 t-{t_back}] step={entry.get('step', '?')}",
                })
                ok, buf = cv2.imencode(
                    ".jpg", entry["bgr"], [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if not ok:
                    continue
                b64 = base64.b64encode(buf).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

        # 1. 四视角 RGB (front/left/right/back) — 缺失会跳过
        if views_bgr:
            for vname in VIEW_ORDER:
                img = views_bgr.get(vname)
                if img is None:
                    continue
                if img.shape[0] != 480 or img.shape[1] != 640:
                    img = cv2.resize(img, (640, 480))
                ok, buf = cv2.imencode(
                    ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if not ok:
                    continue
                b64 = base64.b64encode(buf).decode()
                content.append({"type": "text", "text": VIEW_LABELS[vname]})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

        # 2. 语义地图 (俯视)
        ok, buf = cv2.imencode(
            ".jpg", semantic_map_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed on semantic map")
        b64 = base64.b64encode(buf).decode()
        content.append({"type": "text", "text": "[语义地图，俯视视角]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

        # 3. 文本提示
        completed_block = (
            "\n".join(f"  {i+1}. {s}" for i, s in enumerate(completed_subtasks))
            if completed_subtasks else "  (尚无)"
        )
        current_block = (
            current_subtask.strip()
            if current_subtask and current_subtask.strip()
            else "(尚无当前子任务 —— 请产出第一条)"
        )
        hint_block = f"\n\n注意: {hint}" if hint else ""

        if self.mapped_classes:
            mapped_classes_note = (
                f"语义地图检测器只认 {len(self.mapped_classes)} 类: "
                f"{', '.join(self.mapped_classes)}; 其它物体即使可见也不会进入地图,"
                f"判断物体存在请只看相机视图。"
            )
        else:
            mapped_classes_note = (
                "语义地图检测器词表有限, 许多物体即使可见也不会出现在地图里, "
                "判断物体存在请只看相机视图。"
            )

        text = PLANNER_USER_PROMPT_TEMPLATE.format(
            full_instruction=full_instruction.strip(),
            completed=completed_block,
            current=current_block,
            hint=hint_block,
            mapped_classes_note=mapped_classes_note,
        )
        content.append({"type": "text", "text": text})

        return content

    # ------------------------------------------------------------------
    #  前视历史记忆
    # ------------------------------------------------------------------

    def _maybe_push_front_memory(self,
                                 front_bgr: np.ndarray,
                                 pose: Tuple[float, float, float]):
        x, y, yaw = pose
        if self.front_memory:
            last = self.front_memory[-1]
            dx = x - last["x"]
            dy = y - last["y"]
            dist = math.hypot(dx, dy)
            ang = abs(wrap_angle(yaw - last["yaw"]))
            if (dist < PLANNER_FRONT_MEMORY_MIN_DIST
                    and ang < PLANNER_FRONT_MEMORY_MIN_ANGLE):
                return

        img = front_bgr
        if img.shape[0] != 480 or img.shape[1] != 640:
            img = cv2.resize(img, (640, 480))
        self.front_memory.append({
            "bgr": img.copy(),
            "x": float(x), "y": float(y), "yaw": float(yaw),
            "step": len(self.front_memory),
        })
        if len(self.front_memory) > PLANNER_FRONT_MEMORY_MAX_LEN:
            self.front_memory = self.front_memory[-PLANNER_FRONT_MEMORY_MAX_LEN:]

    def reset_memory(self):
        """每个 episode 切换时可以调一次, 清空前视记忆。"""
        self.front_memory = []

    # ------------------------------------------------------------------
    #  响应解析
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_thinking(raw: str) -> str:
        # 移除 <think>...</think> 段落(可能多段、可能不闭合)
        cleaned = re.sub(r"<think>.*?</think>", "", raw,
                         flags=re.DOTALL | re.IGNORECASE)
        # 若模型只开了 <think> 没闭合,截掉到最后一个闭合标记之后的内容
        cleaned = re.sub(r"<think>[\s\S]*$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _extract_json_object(text: str) -> Optional[dict]:
        # 优先找 ```json ... ``` 代码块
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass

        # 退化: 扫描文本找到第一个平衡的 {...}
        start = text.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
            start = text.find("{", start + 1)
        return None

    def _parse_response(self, raw: str, latency: float) -> PlanDecision:
        clean = self._strip_thinking(raw)

        data = self._extract_json_object(clean) or self._extract_json_object(raw)
        if data is not None:
            status = str(data.get("status", "continue")).strip().lower()
            if status not in ("continue", "advance", "done"):
                status = "continue"
            next_subtask = str(data.get("next_subtask", "") or "").strip()
            reason = str(data.get("reason", "") or "").strip()

            if status == "advance" and not next_subtask:
                return PlanDecision.safe_default(
                    reason="advance_without_subtask",
                    raw=raw, latency=latency,
                )

            return PlanDecision(
                status=status,
                next_subtask=next_subtask,
                reason=reason,
                raw=raw,
                latency_sec=latency,
            )

        # 最后兜底: 正则抓 status 关键字
        status_match = re.search(
            r"status[\"'\s:]+(continue|advance|done)", clean, re.IGNORECASE
        )
        if status_match:
            return PlanDecision(
                status=status_match.group(1).lower(),
                reason="loose_parse",
                raw=raw,
                latency_sec=latency,
            )

        return PlanDecision.safe_default(
            reason="parse_failed", raw=raw, latency=latency,
            error="no valid JSON in response",
        )
