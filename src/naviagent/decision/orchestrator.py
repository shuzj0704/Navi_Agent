"""
Task Orchestrator (双系统胶水)
==============================
把 System2Planner 挂到主循环里:
  - 每步 tick() 非阻塞地探一次 planner future,到了就吸收结果;
  - 冷却期(heartbeat_steps)过后派发下一次 planner 调用;
  - VLM 反应式导航输出 stop 时,on_system1_stop() 阻塞等 planner 判决;
  - 子任务切换时自动 vlm.reset_history() 清空像素决策历史。

主循环只需要:
    orch = TaskOrchestrator(full_instruction, vlm, ...)
    for step in range(max_steps):
        orch.tick(step, mapper, nav, trajectory)
        if orch.is_done: break
        instruction = orch.current_subtask
        pred = vlm.predict(views, instruction, step=step)
        if pred is stop:
            orch.on_system1_stop(step, mapper, nav, trajectory)
            if orch.is_done: break
            # 本步停一拍,下一步 vlm 会用新 subtask 继续
    orch.shutdown()
"""

import re
from concurrent.futures import Future
from typing import Callable, List, Optional, Tuple

from ..common.nav_state import NavState
from ..vlm.planner import PlanDecision, System2Planner
from ..perception.semantic_mapper import SemanticMapper


_ANNOTATION_RE = re.compile(r"\s*\[规划器拒绝之前的 STOP[^\]]*\]\s*")


INITIAL_DECOMPOSITION_HINT = (
    "首次调用: 这是本 episode 规划器的第一次调用。上面的"
    "『当前子任务』就是整体任务原样复制, 尚未做过分解。你必须返回 "
    "'advance' 并给出一个窄、具体的第一步子任务(单一导航动作)。"
    "不要在首次调用返回 'continue' —— 下游的反应式导航器无法自行"
    "分解多步指令, 如果你现在不分解这个 episode 就会失败。如果整体"
    "任务本身已经是单一的原子动作(例如『找一台电视』), 请 advance "
    "并给出一个略具体一点的表述, 例如『环视当前房间寻找电视』。"
)


class TaskOrchestrator:
    def __init__(self,
                 full_instruction: str,
                 api_url: str = "http://localhost:8004/v1",
                 model: str = "qwen3-vl",
                 heartbeat_steps: int = 15,
                 map_size: int = 640,
                 map_scale: int = 25,
                 enable_thinking: bool = True,
                 blocking_timeout: float = 120.0,
                 mapped_classes: Optional[List[str]] = None,
                 max_stop_overrides: int = 2,
                 override_decay_steps: int = 8,
                 on_subtask_change: Optional[Callable[[str], None]] = None,
                 verbose: bool = True,
                 vlm_config=None):
        self.full_instruction = full_instruction.strip()
        self._on_subtask_change = on_subtask_change or (lambda _subtask: None)
        self.heartbeat_steps = heartbeat_steps
        self.map_size = map_size
        self.map_scale = map_scale
        self.blocking_timeout = blocking_timeout
        self.max_stop_overrides = max_stop_overrides
        self.override_decay_steps = override_decay_steps
        self.verbose = verbose

        # 子任务状态
        self.current_subtask: str = self.full_instruction  # planner 给出首个 subtask 前的兜底
        self.completed_subtasks: List[str] = []
        self.is_done: bool = False
        # 当前子任务下发时刻的机器人位姿; System 1 用它计算"本子任务内走了多远/转了多少"
        self.subtask_start_pose: Optional[Tuple[float, float, float]] = None
        self._nav_pose: Optional[Tuple[float, float, float]] = None

        # 规划器
        if vlm_config is not None:
            self.planner = System2Planner(
                config=vlm_config,
                mapped_classes=mapped_classes,
            )
        else:
            self.planner = System2Planner(
                api_url=api_url, model=model,
                enable_thinking=enable_thinking,
                mapped_classes=mapped_classes,
            )

        # 异步调度
        self._in_flight: Optional[Future] = None
        self._last_plan_step: int = -10**9   # 首次 tick 立即触发
        self._using_default: bool = True     # current_subtask 仍是原始整句指令
        self._plan_count: int = 0

        # Stop 死锁守卫:当 planner 在 stop 后说 continue 而 subtask 没变,
        # 给传给 System 1 的指令追加一段 override 后缀,强制改变 prompt
        # 打破死循环。planner 自己看不到这个后缀(它仍然只看 current_subtask)。
        self._stop_override_text: Optional[str] = None
        self._stop_override_count: int = 0
        self._stop_override_step: int = -10**9
        self._last_reason: str = ""

        # System 1 最近一次输出的动作 (view, vx, vy)，用于喂给 planner 历史
        self._last_action: Optional[tuple] = None

    # ------------------------------------------------------------------
    #  对外属性
    # ------------------------------------------------------------------

    @property
    def current_instruction(self) -> str:
        """传给 System 1 反应式 VLM 的真实指令字符串。

        正常情况下 == current_subtask;
        当 stop 死锁守卫被触发时,会附加一个 [OVERRIDE ...] 后缀以
        强制 prompt 发生变化、打破 System 1 的输入-输出死循环。
        planner 永远只看 current_subtask,看不到这个后缀。
        """
        if self._stop_override_text:
            return f"{self.current_subtask} {self._stop_override_text}"
        return self.current_subtask

    def notify_action(self, action: Optional[tuple]):
        """由 NavEngine 在 System 1 产出新动作后调用, 供 planner 历史记录使用。"""
        self._last_action = action

    # ------------------------------------------------------------------
    #  主循环入口
    # ------------------------------------------------------------------

    def tick(self,
             step: int,
             mapper: SemanticMapper,
             nav: NavState,
             trajectory: List[Tuple[float, float]],
             views_bgr: Optional[dict] = None):
        """每步主循环调用一次(非阻塞)。

        views_bgr: 当前帧三视角 BGR 图像 dict {"front": ..., "left": ..., "right": ...},
                   会和语义地图一起喂给 planner。
        """
        if self.is_done:
            return

        # 记录最新位姿; 首次 tick 时初始化 subtask_start_pose
        self._nav_pose = (nav.x, nav.y, nav.yaw)
        if self.subtask_start_pose is None:
            self.subtask_start_pose = self._nav_pose

        # 0. 衰减 stop override(N 步无再触发即清掉)
        if (self._stop_override_text is not None
                and (step - self._stop_override_step) > self.override_decay_steps):
            self._log(
                f"clearing stop override (no re-trigger for "
                f"{self.override_decay_steps} steps)"
            )
            self._stop_override_text = None
            self._stop_override_count = 0

        # 1. 有结果就吸收
        if self._in_flight is not None and self._in_flight.done():
            self._absorb(step)

        # 2. 冷却期满且当前没有 in-flight → 派发新的规划
        if self._in_flight is None and \
                (step - self._last_plan_step) >= self.heartbeat_steps:
            self._dispatch(step, mapper, nav, trajectory,
                           views_bgr=views_bgr, hint=None)

    def on_system1_stop(self,
                        step: int,
                        mapper: SemanticMapper,
                        nav: NavState,
                        trajectory: List[Tuple[float, float]],
                        views_bgr: Optional[dict] = None):
        """
        反应式 VLM 输出 stop 时调用(阻塞)。
        若已经有 in-flight 调用就等它;否则立刻派发一次带 hint 的同步调用。

        关键守卫:如果 planner 在 stop 之后依然返回 'continue'(子任务不变),
        会触发 stop-override:给传给 System 1 的 prompt 追加一段强制后缀,
        从根上改变 prompt,打破 System 1 的输入-输出死循环。
        """
        if self.is_done:
            return

        prev_subtask = self.current_subtask

        if self._in_flight is None:
            verify_attempt = self._stop_override_count + 1
            self._dispatch(
                step, mapper, nav, trajectory,
                views_bgr=views_bgr,
                hint=(
                    f"系统 1(反应式导航器)刚刚发出 STOP "
                    f"(第 {verify_attempt} 次验证)。系统 1 会在它认为目标物体"
                    f"已经出现在某个相机视图中且距离很近时发出 STOP。"
                    f"你现在的任务是从相机视图核实这一判断: "
                    f"  - 首先仔细查看前视图，然后再看其它三个方向。"
                    f"  - 如果你在任意视图中能清楚看到目标，请返回 DONE。"
                    f"    目标不需要同时出现在语义地图中 —— 检测器只认识一小"
                    f"    部分固定词表的类别，经常漏检。"
                    f"  - 只有当你确信四个视图中都看不到目标时，才返回 "
                    f"    ADVANCE 或 CONTINUE。"
                    f"  - 不要仅仅因为子任务文字本身还合理就选择 'continue' "
                    f"    —— 如果你一直 continue 而系统 1 一直 STOP，机器人"
                    f"    会陷入死锁。"
                ),
            )

        self._log(f"blocking on planner (stop signal at step {step})")
        try:
            self._in_flight.result(timeout=self.blocking_timeout)
        except Exception as e:
            self._log(f"blocking planner call failed: {e}")

        self._absorb(step)

        if self.is_done:
            return

        # Stop 死锁守卫:planner 已经看过证据但 subtask 没变 → continue
        if self.current_subtask == prev_subtask:
            self._handle_stop_deadlock()
        # 否则 _absorb 已经清掉 override 并切换到了新 subtask

    def shutdown(self):
        self.planner.shutdown()

    # ------------------------------------------------------------------
    #  派发 / 吸收
    # ------------------------------------------------------------------

    def _dispatch(self,
                  step: int,
                  mapper: SemanticMapper,
                  nav: NavState,
                  trajectory: List[Tuple[float, float]],
                  views_bgr: Optional[dict],
                  hint: Optional[str]):
        try:
            semantic_map = mapper.render_topdown(
                agent_x=nav.x, agent_y=nav.y, agent_yaw=nav.yaw,
                map_size=self.map_size, scale=self.map_scale,
                trajectory=trajectory,
            )
        except Exception as e:
            self._log(f"render_topdown failed, skip dispatch: {e}")
            self._last_plan_step = step
            return

        # 如果调用方没传 hint 且当前还在 default 模式(子任务就是原指令),
        # 自动注入 INITIAL_DECOMPOSITION_HINT,告诉 planner 必须 advance 出第一步
        effective_hint = hint
        if effective_hint is None and self._using_default:
            effective_hint = INITIAL_DECOMPOSITION_HINT

        self._in_flight = self.planner.submit(
            full_instruction=self.full_instruction,
            current_subtask=self.current_subtask,
            completed_subtasks=self.completed_subtasks,
            semantic_map_bgr=semantic_map,
            views_bgr=views_bgr,
            hint=effective_hint,
            pose=(nav.x, nav.y, nav.yaw),
            step=step,
            last_action=self._last_action,
        )
        self._last_plan_step = step
        self._plan_count += 1
        n_views = len(views_bgr) if views_bgr else 0
        self._log(
            f"dispatch #{self._plan_count} @ step {step} "
            f"views={n_views} hint={'Y' if effective_hint else 'N'} "
            f"default={self._using_default} "
            f"subtask={self._short(self.current_subtask)}"
        )

    def _absorb(self, step: int):
        fut = self._in_flight
        if fut is None:
            return

        try:
            decision: PlanDecision = fut.result()
        except Exception as e:
            self._log(f"planner future raised: {e}")
            self._in_flight = None
            self._last_plan_step = step
            return

        self._in_flight = None
        self._last_plan_step = step
        self._last_reason = (decision.reason or "").strip()

        if decision.error:
            self._log(
                f"[{decision.status}] error={decision.error} "
                f"latency={decision.latency_sec:.1f}s"
            )
        else:
            self._log(
                f"[{decision.status}] latency={decision.latency_sec:.1f}s "
                f"reason={decision.reason!r}"
            )

        if decision.status == "done":
            self.is_done = True
            self._stop_override_text = None
            self._stop_override_count = 0
            self._log("TASK COMPLETE")
            return

        if decision.status == "advance":
            next_subtask = decision.next_subtask.strip()
            if not next_subtask:
                self._log("advance without next_subtask — keeping current")
                return
            if not self._using_default:
                self.completed_subtasks.append(self.current_subtask)
            self.current_subtask = next_subtask
            self._using_default = False
            self._stop_override_text = None
            self._stop_override_count = 0
            self._on_subtask_change(next_subtask)
            # 新子任务起点 = 当前位姿
            self.subtask_start_pose = self._nav_pose
            self._log(f"→ new sub-task: {self._short(next_subtask)}")
            return

        # status == "continue": subtask 不变 (override 由 on_system1_stop 决定是否注入)

    # ------------------------------------------------------------------
    #  Stop 死锁守卫
    # ------------------------------------------------------------------

    def _handle_stop_deadlock(self):
        """planner 在 stop 后说 continue 但 subtask 不变 → 死锁。
        前 max_stop_overrides 次:注入 override 后缀强制改 prompt;
        超过阈值:auto-advance,把 reason 直接写进新 subtask 当作硬约束。
        """
        self._stop_override_count += 1
        reason = self._last_reason or "规划器拒绝了刚才的 STOP"

        if self._stop_override_count > self.max_stop_overrides:
            # 多次 override 都没救活 → 把 reason 嵌进 subtask,正式 advance
            base = _ANNOTATION_RE.sub(" ", self.current_subtask).strip()
            new_subtask = (
                f"{base} [规划器拒绝之前的 STOP: {reason}。"
                f"请看向完全不同的方向或区域, 不要回到之前的位置。]"
            )
            self._log(
                f"max stop overrides ({self.max_stop_overrides}) exhausted "
                f"— auto-advancing"
            )
            if not self._using_default:
                self.completed_subtasks.append(self.current_subtask)
            self.current_subtask = new_subtask
            self._using_default = False
            self._stop_override_text = None
            self._stop_override_count = 0
            self._on_subtask_change(new_subtask)
            self._log(f"auto-advanced subtask: {self._short(new_subtask)}")
            return

        # 否则:注入(或更新)override 后缀
        self._stop_override_text = (
            f"[高层规划器 OVERRIDE #{self._stop_override_count}: "
            f"\"{reason}\"。你必须继续移动, 本步不要输出 v,0,0 (STOP)。"
            f"请重新审视四个视角, 在 front/left/right/back 中挑一个"
            f"可通行像素作为移动目标。]"
        )
        self._stop_override_step = self._last_plan_step
        self._on_subtask_change(self.current_subtask)
        self._log(
            f"injected STOP override #{self._stop_override_count}: "
            f"reason={reason!r}"
        )

    # ------------------------------------------------------------------
    #  公开属性 / 可视化状态
    # ------------------------------------------------------------------

    @property
    def last_reason(self) -> str:
        return self._last_reason

    @property
    def stop_override_text(self) -> Optional[str]:
        return self._stop_override_text

    def get_viz_state(self) -> dict:
        """返回可视化所需的全部状态, 不暴露内部对象。"""
        return {
            "current_subtask": self.current_subtask,
            "current_instruction": self.current_instruction,
            "completed_subtasks": list(self.completed_subtasks),
            "last_reason": self._last_reason,
            "stop_override_text": self._stop_override_text,
        }

    # ------------------------------------------------------------------
    #  工具
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Orchestrator] {msg}")

    @staticmethod
    def _short(s: str, n: int = 80) -> str:
        s = s.strip().replace("\n", " ")
        return s if len(s) <= n else s[: n - 3] + "..."
