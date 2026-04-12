"""Interactive replay viewer for vlm_gps_nav runs.

Loads a run directory produced by src/sim_vln_outdoor/scripts/vlm_gps_nav.py
and lets you step through the FPV frames, the top-down trajectory, and the
per-step VLM prompt/reply with keyboard navigation.

Usage:
    python scripts/utils/replay_vlm_gps_nav.py \
        data/urbanverse/vlm_gps_nav/20260411_180711

Keys (focus the matplotlib window first):
    Right / Left   : next / previous step
    Shift+Right/Left : jump 10 steps
    Home / End     : first / last step
    Space          : play / pause auto-advance
    +/-            : speed up / slow down playback
    1..9           : jump to 10%, 20%, ... 90%
    0              : jump to 0%
    g              : prompt for absolute step index in terminal
    q              : quit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider
from PIL import Image


def load_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_next_turn(prompt: str) -> str:
    for line in prompt.splitlines():
        if "NEXT TURN" in line:
            return line.split("NEXT TURN:", 1)[-1].strip()
    return ""


def extract_lookahead(prompt: str) -> list[str]:
    out, in_block = [], False
    for line in prompt.splitlines():
        if "NEXT 5 WAYPOINTS" in line:
            in_block = True
            continue
        if in_block:
            stripped = line.strip()
            if not stripped:
                break
            if stripped.startswith(("1.", "2.", "3.", "4.", "5.")):
                out.append(stripped)
            else:
                break
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=str, help="Run output dir, e.g. data/urbanverse/vlm_gps_nav/<timestamp>")
    p.add_argument("--trajectory", type=str, default=None,
                   help="Override dense_trajectory.json path (default: read from summary.json).")
    p.add_argument("--start-step", type=int, default=0)
    return p.parse_args()


class Replay:
    def __init__(self, run_dir: Path, trajectory_path: Path | None, start_step: int):
        self.run_dir = run_dir
        self.frames_dir = run_dir / "frames"

        with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
            self.summary = json.load(f)
        self.traj = load_jsonl(run_dir / "trajectory.jsonl")
        self.vlm = load_jsonl(run_dir / "vlm_io.jsonl")
        assert len(self.traj) == len(self.vlm), \
            f"trajectory ({len(self.traj)}) and vlm_io ({len(self.vlm)}) length mismatch"
        self.n_steps = len(self.traj)

        # Resolve dense trajectory: --trajectory > summary > project default
        if trajectory_path is None:
            project_root = Path(__file__).resolve().parents[2]
            tp = self.summary.get("trajectory_path", "")
            candidate = project_root / tp if tp else None
            if candidate and candidate.exists():
                trajectory_path = candidate
            else:
                # fall back to scene_09 default
                trajectory_path = project_root / "data/urbanverse/trajectory/scene_09/dense_trajectory.json"
        with open(trajectory_path, "r", encoding="utf-8") as f:
            dense = json.load(f)
        self.path_xy = np.array([[p[0], p[1]] for p in dense["path"]])
        self.waypoints_xy = np.array([[w["pos"][0], w["pos"][1]] for w in dense["waypoints"]])
        self.goal_xy = self.path_xy[-1]
        self.actual_xy = np.array([[t["pose"]["x"], t["pose"]["y"]] for t in self.traj])

        # Pre-extract reply / action / pose / next_turn for fast access
        self.replies = [v.get("reply", "") for v in self.vlm]
        self.next_turns = [extract_next_turn(v.get("prompt", "")) for v in self.vlm]
        self.lookaheads = [extract_lookahead(v.get("prompt", "")) for v in self.vlm]

        # State
        self.cur = max(0, min(start_step, self.n_steps - 1))
        self.playing = False
        self.play_dt_ms = 200  # 5 fps default

        self._build_figure()
        self.update()

    # ────────────────────────────────────────────────────────────────────
    def _build_figure(self):
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.canvas.manager.set_window_title(f"replay: {self.run_dir.name}")
        gs = GridSpec(3, 2, figure=self.fig, width_ratios=[1.4, 1.0],
                      height_ratios=[6, 4, 0.4], hspace=0.25, wspace=0.15)

        # Left: FPV image
        self.ax_img = self.fig.add_subplot(gs[:2, 0])
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])
        self.im_artist = self.ax_img.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        self.img_title = self.ax_img.set_title("")

        # Right top: top-down map
        self.ax_map = self.fig.add_subplot(gs[0, 1])
        self.ax_map.set_aspect("equal")
        self.ax_map.set_title("Top-down trajectory")
        self.ax_map.plot(self.path_xy[:, 0], self.path_xy[:, 1], "-",
                         color="#888", lw=2, label="planned (dense)")
        self.ax_map.scatter(self.waypoints_xy[:, 0], self.waypoints_xy[:, 1],
                            marker="s", s=70, c="#444", zorder=5, label="waypoints")
        self.ax_map.scatter(self.path_xy[0, 0], self.path_xy[0, 1],
                            marker="o", s=120, c="green", zorder=6, label="start")
        self.ax_map.scatter(self.goal_xy[0], self.goal_xy[1],
                            marker="*", s=220, c="red", zorder=6, label="goal")
        # actual full path (will be partially shown via separate artist)
        (self.actual_line,) = self.ax_map.plot([], [], "-", color="dodgerblue", lw=2, label="actual")
        self.cur_marker = self.ax_map.scatter([], [], marker="o", s=110,
                                              c="orange", edgecolors="black", zorder=7)
        # heading arrow (quiver)
        self.heading_arrow = self.ax_map.quiver(
            [self.actual_xy[0, 0]], [self.actual_xy[0, 1]], [1.0], [0.0],
            angles="xy", scale_units="xy", scale=0.5, color="orange", width=0.012,
            zorder=8,
        )
        self.ax_map.legend(loc="lower right", fontsize=8)
        # nice padding
        all_x = np.concatenate([self.path_xy[:, 0], self.actual_xy[:, 0]])
        all_y = np.concatenate([self.path_xy[:, 1], self.actual_xy[:, 1]])
        pad = 5.0
        self.ax_map.set_xlim(all_x.min() - pad, all_x.max() + pad)
        self.ax_map.set_ylim(all_y.min() - pad, all_y.max() + pad)
        self.ax_map.grid(True, alpha=0.3)

        # Right middle: text panel
        self.ax_txt = self.fig.add_subplot(gs[1, 1])
        self.ax_txt.set_axis_off()
        self.txt_artist = self.ax_txt.text(
            0.0, 1.0, "", fontsize=9, family="monospace",
            va="top", ha="left", transform=self.ax_txt.transAxes,
        )

        # Bottom: slider spanning full width
        ax_slider = self.fig.add_subplot(gs[2, :])
        self.slider = Slider(ax_slider, "step", 0, self.n_steps - 1,
                             valinit=self.cur, valstep=1, valfmt="%d")
        self.slider.on_changed(self._on_slider)

        # Connect events
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.timer = self.fig.canvas.new_timer(interval=self.play_dt_ms)
        self.timer.add_callback(self._auto_advance)

    # ────────────────────────────────────────────────────────────────────
    def _load_frame(self, idx: int) -> np.ndarray:
        path = self.frames_dir / f"frame_{idx:06d}.png"
        if not path.exists():
            return np.zeros((480, 640, 3), dtype=np.uint8)
        with Image.open(path) as im:
            return np.array(im.convert("RGB"))

    def update(self):
        i = self.cur
        traj = self.traj[i]
        vlm = self.vlm[i]
        pose = traj["pose"]
        action = traj["action"]
        d2g = traj["dist_to_goal_m"]
        reply = self.replies[i]
        next_turn = self.next_turns[i] or "(none)"
        lookahead = self.lookaheads[i]
        progress_idx = vlm.get("progress_idx", -1)

        # Image
        rgb = self._load_frame(i)
        self.im_artist.set_data(rgb)
        self.img_title.set_text(
            f"FPV  step {i}/{self.n_steps - 1}   reply={reply}"
        )

        # Map: actual path up to step i
        self.actual_line.set_data(self.actual_xy[: i + 1, 0], self.actual_xy[: i + 1, 1])
        self.cur_marker.set_offsets([[pose["x"], pose["y"]]])
        yaw_rad = np.radians(pose["yaw"])
        ux, uy = np.cos(yaw_rad), np.sin(yaw_rad)
        self.heading_arrow.set_offsets([[pose["x"], pose["y"]]])
        self.heading_arrow.set_UVC([ux * 3.0], [uy * 3.0])

        # Text panel
        lookahead_str = "\n          ".join(lookahead) if lookahead else "(none)"
        text = (
            f"STEP {i}   progress_idx={progress_idx}   d2g={d2g:6.2f} m\n"
            f"pose:    x={pose['x']:9.2f}  y={pose['y']:8.2f}  yaw={pose['yaw']:6.1f} deg\n"
            f"action:  forward={action.get('forward', 0):.2f}  yaw={action.get('yaw', 0):.2f}  done={action.get('done', False)}\n"
            f"\n"
            f"NEXT TURN: {next_turn}\n"
            f"LOOKAHEAD:\n          {lookahead_str}\n"
            f"\n"
            f"VLM REPLY: {reply}\n"
        )
        self.txt_artist.set_text(text)

        # Sync slider without recursive callback
        if int(self.slider.val) != i:
            self.slider.eventson = False
            self.slider.set_val(i)
            self.slider.eventson = True

        self.fig.canvas.draw_idle()

    # ────────────────────────────────────────────────────────────────────
    def _on_slider(self, val):
        new = int(val)
        if new != self.cur:
            self.cur = new
            self.update()

    def _on_key(self, event):
        key = event.key
        if key == "right":
            self.cur = min(self.cur + 1, self.n_steps - 1)
        elif key == "left":
            self.cur = max(self.cur - 1, 0)
        elif key == "shift+right":
            self.cur = min(self.cur + 10, self.n_steps - 1)
        elif key == "shift+left":
            self.cur = max(self.cur - 10, 0)
        elif key == "home":
            self.cur = 0
        elif key == "end":
            self.cur = self.n_steps - 1
        elif key == " ":
            self.playing = not self.playing
            if self.playing:
                self.timer.start()
            else:
                self.timer.stop()
            return
        elif key in ("+", "="):
            self.play_dt_ms = max(40, int(self.play_dt_ms * 0.7))
            self.timer.interval = self.play_dt_ms
            print(f"[replay] playback interval -> {self.play_dt_ms}ms ({1000/self.play_dt_ms:.1f} fps)")
            return
        elif key == "-":
            self.play_dt_ms = min(2000, int(self.play_dt_ms * 1.4))
            self.timer.interval = self.play_dt_ms
            print(f"[replay] playback interval -> {self.play_dt_ms}ms ({1000/self.play_dt_ms:.1f} fps)")
            return
        elif key in "0123456789":
            frac = int(key) / 10.0
            self.cur = int(round(frac * (self.n_steps - 1)))
        elif key == "g":
            try:
                target = int(input(f"jump to step [0..{self.n_steps - 1}]: "))
                self.cur = max(0, min(target, self.n_steps - 1))
            except (ValueError, EOFError):
                return
        elif key == "q":
            plt.close(self.fig)
            return
        else:
            return
        self.update()

    def _auto_advance(self):
        if self.cur >= self.n_steps - 1:
            self.playing = False
            self.timer.stop()
            return
        self.cur += 1
        self.update()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"[error] run dir does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)
    for f in ("summary.json", "trajectory.jsonl", "vlm_io.jsonl", "frames"):
        if not (run_dir / f).exists():
            print(f"[error] missing {f} in {run_dir}", file=sys.stderr)
            sys.exit(1)

    trajectory_path = Path(args.trajectory).resolve() if args.trajectory else None
    replay = Replay(run_dir, trajectory_path, args.start_step)

    print(f"[replay] {run_dir.name}: {replay.n_steps} steps")
    print(f"[replay] success={replay.summary.get('success')}  "
          f"final_dist={replay.summary.get('final_dist_to_goal_m'):.2f}m  "
          f"route={replay.summary.get('route_length_m'):.2f}m")
    print("[replay] keys: ←/→ step  ⇧+←/→ ±10  Home/End  Space play  +/- speed  0..9 jump  g goto  q quit")
    plt.show()


if __name__ == "__main__":
    main()
