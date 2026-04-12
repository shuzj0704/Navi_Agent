"""
导航引擎
========
核心导航状态机, 从 nav_main.py 提取。
管理 VLM 异步调用管线、目标生命周期、DWA 内循环、编排器交互。

nav_main.py 和 batch_eval.py 共享同一个引擎, 消除代码重复。

用法:
    engine = NavigationEngine(vlm, dwa, turn_ctrl, front_intrinsics, instruction, ...)
    while not done:
        obs = reader.read()
        result = engine.step(obs, step_num)
        for action in result.actions:
            platform.execute(action)
        if result.done:
            break
    engine.shutdown()
"""

import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..common.nav_state import NavState
from .dwa_planner import DWAPlanner
from .turn_controller import TurnController
from ..vlm.vlm_navigator import VLMNavigator, VLMAsyncWorker
from ..perception.pixel_to_3d import pixel_to_camera_3d
from ..common.coordinate_transform import camera_point_to_nav2d
from ..perception.obs_reader import ObsBundle


def velocity_to_action(v, omega, v_thresh=0.05, w_thresh=0.25):
    """DWA (v, omega) → 离散动作列表 ["move_forward"] / ["turn_left"] / ...
    坐标系 X=前 Y=右: omega>0 → 右转, omega<0 → 左转"""
    actions = []
    if abs(omega) > w_thresh:
        actions.append("turn_right" if omega > 0 else "turn_left")
    if v > v_thresh:
        actions.append("move_forward")
    if not actions:
        actions.append("move_forward")
    return actions


@dataclass
class NavEngineConfig:
    """导航引擎可调参数"""
    goal_reached_threshold: float = 0.3   # m
    v_eps: float = 0.01                    # m/s, DWA 输出 v < 此值视为停下
    idle_cap_multiplier: int = 200         # loop_guard 上限 = max_steps * 此值


@dataclass
class StepResult:
    """引擎每步的输出, 告诉调用方该做什么。"""
    actions: List[str] = field(default_factory=list)
    done: bool = False
    done_reason: str = ""
    idle: bool = False          # 没有活动目标, 等待 VLM 结果
    step_counted: bool = True   # 本次是否算一步 (idle 时为 False)

    # 可视化数据
    dwa_debug: Optional[dict] = None
    vlm_view: Optional[str] = None
    target_vx: Optional[float] = None
    target_vy: Optional[float] = None
    action_type: Optional[str] = None
    cam_goal: Optional[object] = None


class NavigationEngine:
    """共享导航状态机。

    不直接操作仿真器 — 通过 ObsBundle 接收观测, 通过 StepResult 输出动作。
    """

    def __init__(self,
                 vlm: Optional[VLMNavigator],
                 dwa: DWAPlanner,
                 turn_ctrl: TurnController,
                 front_intrinsics: Tuple[float, float, float, float],
                 instruction: str,
                 orchestrator=None,
                 mapper=None,
                 config: Optional[NavEngineConfig] = None):
        self.vlm = vlm
        self.dwa = dwa
        self.turn_ctrl = turn_ctrl
        self.front_fx, self.front_fy, self.front_cx, self.front_cy = front_intrinsics
        self.instruction = instruction
        self.orchestrator = orchestrator
        self.mapper = mapper
        self.config = config or NavEngineConfig()

        # 公开状态
        self.nav = NavState()
        self.trajectory: List[Tuple[float, float]] = []
        self.vlm_times: List[float] = []

        # 异步 VLM 管线
        self._vlm_worker = VLMAsyncWorker(vlm)
        self._goal_nav: Optional[Tuple[float, float]] = None
        self._goal_meta: Optional[dict] = None
        self._pending_future = None
        self._pending_snapshot: Optional[dict] = None
        self._staged_prediction = None
        self._last_instruction: Optional[str] = None

    def step(self, obs: ObsBundle, step_num: int) -> StepResult:
        """执行一次导航状态机迭代。

        Args:
            obs: 一帧处理后的观测
            step_num: 当前步数 (供 VLM / orchestrator 使用)
        Returns:
            StepResult 告诉调用方该执行哪些动作
        """
        result = StepResult()

        # ---- 1. 从观测更新导航状态 ----
        self.nav.x, self.nav.y, self.nav.yaw = obs.nav_x, obs.nav_y, obs.nav_yaw
        self.nav.obstacles = obs.obstacles_global

        # ---- 2. Orchestrator tick ----
        current_instruction = self._tick_orchestrator(step_num, obs)
        if self.orchestrator and self.orchestrator.is_done:
            result.done = True
            result.done_reason = "orchestrator_done"
            return result

        # ---- 3. 指令变化检测 ----
        if current_instruction != self._last_instruction:
            if self._last_instruction is not None:
                print(f"[Engine] 指令变: {self._last_instruction!r} → {current_instruction!r}")
            self._last_instruction = current_instruction
            self._goal_nav = None
            self._goal_meta = None
            self._staged_prediction = None

        # ---- 4. 吸收 pending Future (不阻塞) ----
        self._absorb_future(current_instruction)

        # ---- 5. 消费 staged_prediction (仅在当前没有活动目标时) ----
        if self._goal_nav is None and self._staged_prediction is not None:
            consumed = self._consume_staged(obs, step_num, result, current_instruction)
            if consumed:
                return result

        # ---- 6. 提交新 VLM 调用 ----
        # 关键: goal_nav is None 防止 spin-in-place bug
        if (self._pending_future is None
                and self._staged_prediction is None
                and self._goal_nav is None):
            self._submit_vlm(obs, step_num, current_instruction)

        # ---- 7. DWA 内循环 ----
        if self._goal_nav is not None:
            return self._dwa_step(obs, step_num, result)

        # 没有活动目标, 等 VLM 结果
        result.idle = True
        result.step_counted = False
        return result

    def shutdown(self):
        """清理资源"""
        self._vlm_worker.shutdown()
        if self.orchestrator:
            self.orchestrator.shutdown()

    def reset(self):
        """重置为新 episode (batch_eval 用)"""
        self._goal_nav = None
        self._goal_meta = None
        self._pending_future = None
        self._pending_snapshot = None
        self._staged_prediction = None
        self._last_instruction = None
        self.nav = NavState()
        self.trajectory = []
        if self.vlm:
            self.vlm.reset_history()

    # ------------------------------------------------------------------
    #  私有方法
    # ------------------------------------------------------------------

    def _tick_orchestrator(self, step_num, obs):
        """调用 orchestrator.tick(), 返回当前指令"""
        if self.orchestrator is not None:
            self.orchestrator.tick(
                step_num, self.mapper, self.nav, self.trajectory,
                views_bgr=obs.views_bgr,
            )
            return self.orchestrator.current_instruction
        return self.instruction

    def _absorb_future(self, current_instruction):
        """非阻塞地检查 VLM future 是否完成"""
        if self._pending_future is None or not self._pending_future.done():
            return

        try:
            prediction = self._pending_future.result()
        except Exception as e:
            print(f"[VLM1] future raised: {e}")
            prediction = None

        if self.vlm is not None:
            self.vlm_times.append(self.vlm.last_latency)
            avg_vlm = sum(self.vlm_times) / len(self.vlm_times)
            print(f"[VLM1] {self.vlm.last_latency*1000:.0f}ms "
                  f"(avg {avg_vlm*1000:.0f}ms, n={len(self.vlm_times)})")

        snap = self._pending_snapshot
        self._pending_future = None
        self._pending_snapshot = None

        if snap is None or snap["instruction"] != current_instruction:
            print("[VLM1] 结果已过期 (指令变化), 丢弃")
        elif prediction is None:
            print("[VLM1] parse failed, 丢弃")
        else:
            self._staged_prediction = (prediction, snap)

    def _consume_staged(self, obs, step_num, result, current_instruction):
        """消费 staged_prediction。返回 True 表示本步已完成 (调用方应 return result)。"""
        prediction, snap = self._staged_prediction
        self._staged_prediction = None
        vv, vvx, vvy = prediction

        # STOP
        if vv == "stop":
            print(f"[VLM1] STOP (subtask={current_instruction!r})")
            should_end = self._on_stop(obs, snap, step_num)
            if should_end:
                result.done = True
                result.done_reason = "vlm_stop"
                return True
            result.action_type = "stop"
            result.vlm_view = vv
            return True

        # 转向 / 前进决策
        action_type, tvx, tvy = self.turn_ctrl.decide(vv, vvx, vvy)

        if action_type in ("turn_left", "turn_right"):
            print(f"[VLM1] {vv} → {action_type}")
            self.nav.goal_valid = False
            self.nav.cmd_v = 0.0
            self.nav.cmd_omega = -0.5 if action_type == "turn_left" else 0.5
            self._update_mapper(obs)
            self.trajectory.append((self.nav.x, self.nav.y))
            result.actions = [action_type]
            result.action_type = action_type
            result.vlm_view = vv
            return True

        # forward: 用 snapshot 里的 front_depth / 位姿反投影得到全局目标
        cam_goal = pixel_to_camera_3d(
            tvx, tvy, snap["front_depth"],
            self.front_fx, self.front_fy, self.front_cx, self.front_cy,
        )
        if cam_goal is None:
            print("[VLM1] cam_goal 无效 (深度空洞), 兜底 move_forward")
            self._fallback_forward(obs, result, vv, tvx, tvy)
            return True

        gx, gy = camera_point_to_nav2d(
            cam_goal, snap["nav_x"], snap["nav_y"], snap["nav_yaw"],
        )
        init_dist = math.hypot(gx - self.nav.x, gy - self.nav.y)
        if init_dist < self.config.goal_reached_threshold:
            print(f"[VLM1] 目标太近 ({init_dist:.2f}m < "
                  f"{self.config.goal_reached_threshold}m), 兜底 move_forward")
            self._fallback_forward(obs, result, vv, tvx, tvy)
            return True

        # 设置新的全局目标
        self._goal_nav = (gx, gy)
        self._goal_meta = {
            "cam_goal": cam_goal,
            "vlm_view": vv,
            "target_vx": tvx,
            "target_vy": tvy,
        }
        self.nav.goal_x, self.nav.goal_y = gx, gy
        self.nav.goal_valid = True
        print(f"[VLM1] 新目标 (全局 nav): ({gx:.2f}, {gy:.2f}), dist={init_dist:.2f}m")
        return False  # 目标已设, 交给 DWA 步骤处理

    def _dwa_step(self, obs, step_num, result):
        """DWA 追踪当前目标走一步"""
        gx, gy = self._goal_nav
        dx = gx - self.nav.x
        dy = gy - self.nav.y
        dist = math.hypot(dx, dy)

        if dist < self.config.goal_reached_threshold:
            print(f"[DWA] 到达目标 dist={dist:.2f}m")
            self._goal_nav = None
            self._goal_meta = None
            result.idle = True
            result.step_counted = False
            return result

        # 全局目标 → 机器人局部坐标
        cos_y = math.cos(self.nav.yaw)
        sin_y = math.sin(self.nav.yaw)
        goal_local = np.array(
            [dx * cos_y + dy * sin_y, dx * sin_y - dy * cos_y],
            dtype=np.float32,
        )
        robot_state = np.array([0.0, 0.0, 0.0, self.nav.v, self.nav.omega])
        v, omega, dwa_debug = self.dwa.plan_debug(
            robot_state, goal_local, obs.obstacles_local
        )
        if dwa_debug is not None:
            dwa_debug["obstacles_local"] = obs.obstacles_local
        self.nav.cmd_v, self.nav.cmd_omega = v, omega
        self.nav.goal_x, self.nav.goal_y = gx, gy
        self.nav.goal_valid = True

        if abs(v) < self.config.v_eps:
            print(f"[DWA] v≈0 ({v:+.3f}, w={omega:+.3f}), 丢弃目标")
            result.dwa_debug = dwa_debug
            result.vlm_view = self._goal_meta["vlm_view"] if self._goal_meta else "front"
            result.action_type = "forward"
            self._goal_nav = None
            self._goal_meta = None
            return result

        actions = velocity_to_action(v, omega)
        if not isinstance(actions, list):
            actions = [actions]

        self._update_mapper(obs)
        self.trajectory.append((self.nav.x, self.nav.y))

        result.actions = actions
        result.dwa_debug = dwa_debug
        result.vlm_view = self._goal_meta["vlm_view"]
        result.target_vx = self._goal_meta["target_vx"]
        result.target_vy = self._goal_meta["target_vy"]
        result.action_type = "forward"
        result.cam_goal = self._goal_meta["cam_goal"]
        return result

    def _on_stop(self, obs, snap, step_num):
        """VLM STOP 信号处理。返回 True 表示导航应当结束。"""
        if self.orchestrator is None:
            print("[VLM1] STOP — end")
            return True

        self._update_mapper(obs)
        self.trajectory.append((self.nav.x, self.nav.y))
        print("[Orchestrator] blocking planner for stop re-evaluation...")
        self.orchestrator.on_system1_stop(
            step_num, self.mapper, self.nav, self.trajectory,
            views_bgr=obs.views_bgr,
        )
        if self.orchestrator.is_done:
            print("[Orchestrator] planner 确认任务完成 — 结束导航")
            return True

        print(f"[Orchestrator] next instruction → "
              f"{self.orchestrator.current_instruction!r}")
        self._goal_nav = None
        self._goal_meta = None
        self._last_instruction = self.orchestrator.current_instruction
        return False

    def _submit_vlm(self, obs, step_num, current_instruction):
        """提交一次新的后台 VLM 调用"""
        sem_for_vlm = None
        if self.mapper is not None:
            sem_for_vlm = self.mapper.render_topdown(
                agent_x=self.nav.x, agent_y=self.nav.y, agent_yaw=self.nav.yaw,
                map_size=480, scale=40, trajectory=self.trajectory,
            )
        self._pending_snapshot = {
            "views": {k: v.copy() for k, v in obs.views_bgr.items()},
            "front_depth": obs.front_depth.copy(),
            "nav_x": self.nav.x, "nav_y": self.nav.y, "nav_yaw": self.nav.yaw,
            "instruction": current_instruction,
        }
        self._pending_future = self._vlm_worker.submit(
            self._pending_snapshot["views"], current_instruction, step_num,
            pose=(
                self._pending_snapshot["nav_x"],
                self._pending_snapshot["nav_y"],
                self._pending_snapshot["nav_yaw"],
            ),
            semantic_map=sem_for_vlm,
            subtask_start_pose=(
                self.orchestrator.subtask_start_pose
                if self.orchestrator is not None else None
            ),
        )

    def _fallback_forward(self, obs, result, vlm_view, tvx, tvy):
        """深度空洞或目标太近时的兜底前进"""
        self.nav.goal_valid = False
        self.nav.cmd_v, self.nav.cmd_omega = 0.1, 0.0
        self._update_mapper(obs)
        self.trajectory.append((self.nav.x, self.nav.y))
        result.actions = ["move_forward"]
        result.action_type = "forward"
        result.vlm_view = vlm_view
        result.target_vx = tvx
        result.target_vy = tvy

    def _update_mapper(self, obs):
        """如果有 mapper, 更新语义地图"""
        if self.mapper is not None:
            self.mapper.update(
                obs.views_bgr["front"], obs.front_depth,
                self.nav.x, self.nav.y, self.nav.yaw,
            )
