"""GPS-guided VLM navigation controller.

Each step the controller:
  1. Updates progress along a pre-loaded dense trajectory (closest-point search,
     monotonic — never decreases progress).
  2. Computes a lookahead segment in ego frame (+x = forward, +y = left).
  3. Detects the next sharp turn within the lookahead window and converts it to
     a human-readable hint ("turn right in 1.4m" / "continue straight").
  4. Builds a structured text prompt carrying current pose, route progress,
     ego-frame waypoints, and the next-turn hint.
  5. Sends the FPV RGB image + prompt to a vLLM-served VLM (Qwen3-VL).
  6. Parses the keyword reply (FORWARD / TURN_LEFT / TURN_RIGHT / STOP)
     into an Action.

Reaching `goal_tol` of the final waypoint emits Action(done=True).

Designed to be instantiated and stepped from
`src/sim_vln_outdoor/scripts/vlm_gps_nav.py`, but is also usable from
`nav_eval.py --controller "nav.gps_vlm_controller:GPSVLMNavController"
--controller-kwargs '{"trajectory_path":"..."}'`.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .controller import Action, NavController, Observation


def _derive_start_yaw(dense_path: np.ndarray) -> float:
    """Return the yaw (deg) pointing from dense_path[0] toward dense_path[1].

    A navigation route file only carries positions, not headings. We recover
    the initial heading from the first edge of the route the same way a real
    navigation app would (the agent starts already facing forward).
    """
    if len(dense_path) < 2:
        return 0.0
    dx = float(dense_path[1, 0] - dense_path[0, 0])
    dy = float(dense_path[1, 1] - dense_path[0, 1])
    return math.degrees(math.atan2(dy, dx))


# Make src/vlm_serve importable from this file's location
# (this file lives at src/sim_vln_outdoor/nav/gps_vlm_controller.py)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


class GPSVLMNavController(NavController):
    """VLM controller that follows a pre-recorded GPS trajectory."""

    def __init__(
        self,
        trajectory_path: str,
        base_url: str = "http://localhost:8004/v1",
        model: str = "qwen3-vl",
        instruction: str = "Follow the navigation route to the destination.",
        forward_step: float = 0.5,
        yaw_step: float = 15.0,
        lookahead: int = 5,
        goal_tol: float = 2.0,
        turn_threshold_deg: float = 30.0,
        max_tokens: int = 96,
        enable_thinking: bool = False,
        output_dir: str | None = None,
    ):
        from vlm_serve.client import VLMClient

        # ── load trajectory ────────────────────────────────────────────
        with open(trajectory_path, "r") as f:
            traj = json.load(f)
        self.dense_path = np.asarray(traj["path"], dtype=np.float64)  # (N, 3)
        self.waypoints = traj["waypoints"]
        self.total_length = float(traj["total_length_m"])
        self.goal_pos = self.dense_path[-1]

        # Convenience for the entry script: starting pose.
        # The route file only carries positions — the initial yaw is derived
        # from the first edge of the route so the agent starts already facing
        # forward, matching how a real navigation app behaves.
        self.start_pos = list(traj["waypoints"][0]["pos"])
        self.start_yaw = _derive_start_yaw(self.dense_path)

        # ── vlm client ─────────────────────────────────────────────────
        self.client = VLMClient(base_url=base_url, model=model)

        self.instruction = instruction
        self.forward_step = forward_step
        self.yaw_step = yaw_step
        self.lookahead = int(lookahead)
        self.goal_tol = float(goal_tol)
        self.turn_threshold = float(turn_threshold_deg)
        self.max_tokens = int(max_tokens)
        self.enable_thinking = bool(enable_thinking)

        # ── output dir (frames + io log) ───────────────────────────────
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = str(
                _REPO_ROOT / "data" / "urbanverse" / "vlm_gps_nav" / timestamp
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        self.io_log_path = self.output_dir / "vlm_io.jsonl"

        # ── runtime state ──────────────────────────────────────────────
        self.progress_idx = 0

        print(
            f"[GPSVLM] trajectory={trajectory_path}\n"
            f"[GPSVLM] dense points={len(self.dense_path)} "
            f"length={self.total_length:.1f}m goal={self.goal_pos[:2].tolist()}\n"
            f"[GPSVLM] vlm={base_url} ({model}) "
            f"lookahead={self.lookahead} goal_tol={self.goal_tol}m\n"
            f"[GPSVLM] output -> {self.output_dir}"
        )

    # ── public NavController interface ──────────────────────────────────

    def reset(self) -> None:
        self.progress_idx = 0

    def act(self, obs: Observation) -> Action:
        cur_pos = np.asarray(obs.pose[:3], dtype=np.float64)
        cur_yaw = float(obs.pose[5])  # degrees

        # 1. progress update (monotonic)
        self.progress_idx = self._update_progress(cur_pos)

        # 2. distance to goal -> early termination
        dist_to_goal = float(
            np.linalg.norm(self.goal_pos[:2] - cur_pos[:2])
        )
        if dist_to_goal < self.goal_tol:
            print(f"[GPSVLM] reached goal! dist={dist_to_goal:.2f}m")
            self._log_io(
                obs.step, "<terminal>", "<reached_goal>", Action(done=True),
                cur_pos, cur_yaw, dist_to_goal,
            )
            return Action(done=True)

        # 3. lookahead segment in world coordinates (skip current point)
        end_idx = min(self.progress_idx + self.lookahead, len(self.dense_path) - 1)
        start = self.progress_idx + 1
        stop = end_idx + 1
        lookahead_world = self.dense_path[start:stop]

        # 4. transform to ego frame
        lookahead_ego = self._world_to_ego(lookahead_world, cur_pos, cur_yaw)

        # 5. detect next turn within lookahead
        next_turn = self._detect_next_turn(lookahead_ego)

        # 6. save FPV frame
        frame_path = self.frames_dir / f"frame_{obs.step:06d}.png"
        Image.fromarray(obs.rgb).save(frame_path)

        # 7. build prompt
        prompt = self._build_prompt(
            obs=obs,
            cur_pos=cur_pos,
            cur_yaw=cur_yaw,
            dist_to_goal=dist_to_goal,
            lookahead_ego=lookahead_ego,
            next_turn=next_turn,
        )

        # 8. call VLM
        try:
            reply = self.client.chat_with_image(
                prompt=prompt,
                image_path=frame_path,
                max_tokens=self.max_tokens,
                enable_thinking=self.enable_thinking,
            )
        except Exception as e:
            print(f"[GPSVLM] VLM call failed: {e} -- no-op")
            self._log_io(
                obs.step, prompt, f"<error: {e}>", Action(),
                cur_pos, cur_yaw, dist_to_goal,
            )
            return Action()

        action = self._parse_action(reply)
        print(
            f"[GPSVLM] step={obs.step} prog={self.progress_idx}/{len(self.dense_path)-1} "
            f"d2g={dist_to_goal:.1f}m turn={next_turn} reply={reply!r} -> {action}"
        )
        self._log_io(obs.step, prompt, reply, action, cur_pos, cur_yaw, dist_to_goal)
        return action

    def on_episode_end(self, trajectory: list) -> None:
        print(f"[GPSVLM] episode end: {len(trajectory)} controller steps logged")

    # ── helpers ──────────────────────────────────────────────────────────

    def _update_progress(self, cur_pos: np.ndarray) -> int:
        """Find the closest dense_path point in [progress_idx, end), XY plane only.

        Monotonic — never decreases progress_idx, prevents getting stuck in loops
        if the agent passes the same area twice.
        """
        search = self.dense_path[self.progress_idx:, :2]
        if len(search) == 0:
            return self.progress_idx
        d = np.linalg.norm(search - cur_pos[:2], axis=1)
        local_min = int(np.argmin(d))
        return self.progress_idx + local_min

    @staticmethod
    def _world_to_ego(
        points_world: np.ndarray, origin: np.ndarray, yaw_deg: float
    ) -> np.ndarray:
        """Rotate XY world points into ego frame (+x = forward, +y = left).

        With yaw=0 the agent faces world +x. Rotating world points by -yaw aligns
        the world +x direction with the ego +x axis.
        """
        if len(points_world) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        rel = points_world[:, :2] - origin[:2]
        yaw = math.radians(yaw_deg)
        c, s = math.cos(-yaw), math.sin(-yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        return (R @ rel.T).T  # (N, 2)

    def _detect_next_turn(self, lookahead_ego: np.ndarray) -> str:
        """Find the first sharp bearing change in the lookahead window.

        Returns a human-readable hint:
            "continue straight" if no sharp turn within window
            "turn left in 0.0m" / "turn right in 1.4m" otherwise

        Detection covers two cases:
          1. The very first segment already deviates from ego +x by > threshold
             (e.g. the agent is standing right on a corner) -> "turn ... in 0.0m".
          2. Two consecutive segments differ in bearing by > threshold
             (e.g. an upcoming corner ahead) -> "turn ... in <arc>m".
        """
        if len(lookahead_ego) < 1:
            return "continue straight"

        # First-segment bearing relative to ego +x direction
        first = lookahead_ego[0]
        first_bearing = math.degrees(math.atan2(first[1], first[0]))
        if abs(first_bearing) > self.turn_threshold:
            direction = "left" if first_bearing > 0 else "right"
            return f"turn {direction} in 0.0m"

        if len(lookahead_ego) < 2:
            return "continue straight"

        # Bearings of consecutive segments along the path
        bearings = []
        for i in range(len(lookahead_ego) - 1):
            dx = lookahead_ego[i + 1, 0] - lookahead_ego[i, 0]
            dy = lookahead_ego[i + 1, 1] - lookahead_ego[i, 1]
            bearings.append(math.degrees(math.atan2(dy, dx)))

        for i in range(1, len(bearings)):
            delta = ((bearings[i] - bearings[i - 1] + 180) % 360) - 180
            if abs(delta) > self.turn_threshold:
                # arc length from origin (current pose) to lookahead_ego[i]
                head = lookahead_ego[:i + 1]
                arc_pts = np.vstack([np.zeros((1, 2)), head])
                arc = float(np.sum(
                    np.linalg.norm(np.diff(arc_pts, axis=0), axis=1)
                ))
                direction = "left" if delta > 0 else "right"
                return f"turn {direction} in {arc:.1f}m"
        return "continue straight"

    def _build_prompt(
        self, obs, cur_pos, cur_yaw, dist_to_goal, lookahead_ego, next_turn
    ) -> str:
        progress_pct = int(round(
            100.0 * (1.0 - dist_to_goal / max(self.total_length, 1e-6))
        ))
        progress_pct = max(0, min(100, progress_pct))

        if len(lookahead_ego) > 0:
            wp_lines = "\n".join(
                f"  {i+1}. ({p[0]:+.1f}, {p[1]:+.1f})  "
                f"dist={float(np.linalg.norm(p)):.1f}m"
                for i, p in enumerate(lookahead_ego)
            )
        else:
            wp_lines = "  (none — at end of route)"

        turn_steps = max(1, int(math.ceil(90.0 / max(self.yaw_step, 1e-6))))

        return (
            f"You are an outdoor navigation robot following a pre-planned GPS route.\n"
            f"Task: {self.instruction}\n"
            f"\n"
            f"CAMERA VIEW: the attached first-person RGB image.\n"
            f"IMPORTANT: the image alone is NOT enough to decide. You MUST follow\n"
            f"the NEXT_TURN hint below, even when the road in the image looks\n"
            f"straight and open. The route planner already knows where to turn.\n"
            f"\n"
            f"ROUTE STATE:\n"
            f"  position: ({cur_pos[0]:.1f}, {cur_pos[1]:.1f})   "
            f"heading: {cur_yaw:.0f} deg\n"
            f"  goal: ({self.goal_pos[0]:.1f}, {self.goal_pos[1]:.1f})\n"
            f"  remaining: {dist_to_goal:.1f} m   ({progress_pct}% complete)\n"
            f"  step: {obs.step}\n"
            f"\n"
            f"NEXT {len(lookahead_ego)} WAYPOINTS "
            f"(ego frame, +x=front, +y=left, meters):\n"
            f"{wp_lines}\n"
            f"  (negative y means the waypoint is to your RIGHT;\n"
            f"   positive y means it is to your LEFT)\n"
            f"\n"
            f"=== DECISION RULE — apply NEXT_TURN first, override the image if needed ===\n"
            f"\n"
            f'NEXT_TURN = "{next_turn}"\n'
            f"\n"
            f"Map NEXT_TURN to an action:\n"
            f'  - "continue straight"                         -> FORWARD\n'
            f'  - "turn left  in X m"  with X < 0.5           -> TURN_LEFT\n'
            f'  - "turn right in X m"  with X < 0.5           -> TURN_RIGHT\n'
            f'  - "turn left/right in X m"  with X >= 0.5     -> FORWARD (approach the corner first)\n'
            f"\n"
            f"Notes:\n"
            f"  - Each FORWARD advances {self.forward_step} m. "
            f"Each TURN rotates only {self.yaw_step:.0f} deg.\n"
            f"  - A sharp 90 deg corner takes about {turn_steps} consecutive TURN actions\n"
            f'    to complete. This is normal — keep turning until NEXT_TURN becomes\n'
            f'    "continue straight" again. Do not resume FORWARD after a single TURN.\n'
            f"  - If remaining distance < {self.goal_tol} m -> STOP.\n"
            f"\n"
            f"=== EXAMPLES ===\n"
            f"\n"
            f'Example 1 — NEXT_TURN = "continue straight"\n'
            f"REASON: NEXT_TURN is straight, no corner in sight.\n"
            f"ACTION: FORWARD\n"
            f"\n"
            f'Example 2 — NEXT_TURN = "turn right in 1.4m"\n'
            f"REASON: corner is 1.4m away, I must approach it first before turning.\n"
            f"ACTION: FORWARD\n"
            f"\n"
            f'Example 3 — NEXT_TURN = "turn right in 0.0m"\n'
            f"REASON: I am standing at the corner, must turn right even though the road ahead looks open.\n"
            f"ACTION: TURN_RIGHT\n"
            f"\n"
            f"=== YOUR TURN ===\n"
            f"\n"
            f"Reply format — EXACTLY two lines, no extra text:\n"
            f"  REASON: <one short sentence stating which rule matched>\n"
            f"  ACTION: <FORWARD | TURN_LEFT | TURN_RIGHT | STOP>\n"
        )

    def _parse_action(self, reply: str) -> Action:
        """Parse a two-line REASON/ACTION reply.

        Primary path: find the line starting with "ACTION" and match keywords
        on its right-hand side only, so REASON text does not leak in.
        Fallback: if no ACTION line is found (bare-keyword reply), match
        against the last non-empty line.
        """
        lines = [l.strip() for l in (reply or "").splitlines() if l.strip()]
        action_line = next(
            (l for l in lines if l.upper().startswith("ACTION")),
            lines[-1] if lines else "",
        )
        # Strip "ACTION:" prefix if present, keep only the right-hand side
        if ":" in action_line:
            action_line = action_line.split(":", 1)[1]
        text = action_line.strip().upper()

        # Check compound keywords before bare ones
        if "TURN_LEFT" in text or "TURN LEFT" in text:
            return Action(yaw=self.yaw_step)
        if "TURN_RIGHT" in text or "TURN RIGHT" in text:
            return Action(yaw=-self.yaw_step)
        if "FORWARD" in text:
            return Action(forward=self.forward_step)
        if "STOP" in text:
            return Action(done=True)
        if text == "LEFT":
            return Action(yaw=self.yaw_step)
        if text == "RIGHT":
            return Action(yaw=-self.yaw_step)
        print(f"[GPSVLM] unparseable reply: {reply!r} -- no-op")
        return Action()

    def _log_io(self, step, prompt, reply, action, cur_pos, cur_yaw, dist_to_goal):
        record = {
            "step": int(step),
            "pose": {
                "x": float(cur_pos[0]),
                "y": float(cur_pos[1]),
                "z": float(cur_pos[2]),
                "yaw": float(cur_yaw),
            },
            "progress_idx": int(self.progress_idx),
            "dist_to_goal_m": float(dist_to_goal),
            "prompt": prompt,
            "reply": reply,
            "action": {
                "forward": float(action.forward),
                "yaw": float(action.yaw),
                "done": bool(action.done),
            },
        }
        with open(self.io_log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
