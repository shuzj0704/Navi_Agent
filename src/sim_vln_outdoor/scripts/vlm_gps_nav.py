"""Closed-loop GPS-guided VLM navigation in Isaac Sim.

Loads a dense trajectory (produced by scripts/utils/interpolate_trajectory.py),
spawns the D435i camera at the first waypoint, and lets a Qwen3-VL backed
controller (GPSVLMNavController) drive the camera toward the goal step by step.

Each step the VLM receives:
  - the FPV RGB frame
  - a structured text prompt with current pose, route progress, lookahead
    waypoints in ego frame, and a "next turn" hint
The VLM replies with one of FORWARD / TURN_LEFT / TURN_RIGHT / STOP, which
is converted into a pose delta via the same apply_action() used by nav_eval.

Usage:
    # 0. (One-time per route) Densify the human-authored waypoint file
    #    blog_point.txt -> dense_trajectory.json. Skip if already done.
    python scripts/utils/interpolate_trajectory.py \\
        --input data/urbanverse/trajectory/scene_09/blog_point.txt \\
        --step 0.5 --visualize

    # 1. Start the Qwen3-VL vLLM server (in another terminal)
    conda activate lwy_swift
    python scripts/serve/start_qwen3vl.py        # GPU 2, port 8004

    # 2. Run the closed loop (in the project root)
    cd ~/navigation/Navi_Agent
    xvfb-run -a ./python.sh src/sim_vln_outdoor/scripts/vlm_gps_nav.py \\
        --headless --max-steps 200 --controller-freq 1.0 \\
        --trajectory data/urbanverse/trajectory/scene_09/dense_trajectory.json
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_SCRIPT_DIR)            # sim_vln_outdoor/
sys.path.insert(0, _PACKAGE_ROOT)
sys.path.insert(0, _SCRIPT_DIR)                          # so we can import nav_eval

_PROJECT_ROOT = os.path.dirname(os.path.dirname(_PACKAGE_ROOT))  # Navi_Agent/

_CRAFTBENCH_ROOT = os.environ.get(
    "CRAFTBENCH_ROOT",
    os.path.expanduser("~/navigation/urban_verse/CraftBench"),
)
DEFAULT_USD = os.path.join(
    _CRAFTBENCH_ROOT,
    "scene_09_cbd_t_intersection_construction_sites",
    "Collected_export_version",
    "export_version.usd",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPS-guided VLM closed-loop navigation in Isaac Sim",
    )
    parser.add_argument(
        "--trajectory", type=str, required=True,
        help="Path to dense_trajectory.json (produced by interpolate_trajectory.py)",
    )
    parser.add_argument(
        "--instruction", type=str,
        default="Follow the navigation route to the destination.",
        help="Natural-language task instruction passed to the VLM prompt.",
    )
    parser.add_argument(
        "--usd-path", type=str, default=DEFAULT_USD,
        help="Path to the USD scene file.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--max-steps", type=int, default=200,
        help="Maximum number of controller steps.",
    )
    parser.add_argument(
        "--controller-freq", type=float, default=1.0,
        help="Controller frequency in Hz (camera runs at 20Hz).",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index for rendering (default: 0).",
    )
    parser.add_argument(
        "--base-url", type=str, default="http://localhost:8004/v1",
        help="vLLM server base URL for the VLM.",
    )
    parser.add_argument(
        "--model", type=str, default="qwen3-vl",
        help="Served model name on the vLLM server.",
    )
    parser.add_argument(
        "--lookahead", type=int, default=5,
        help="Number of dense path points ahead to expose to the VLM.",
    )
    parser.add_argument(
        "--goal-tol", type=float, default=2.0,
        help="Distance to goal (m) below which we declare success.",
    )
    parser.add_argument(
        "--forward-step", type=float, default=0.5,
        help="Meters per FORWARD action.",
    )
    parser.add_argument(
        "--yaw-step", type=float, default=15.0,
        help="Degrees per TURN action.",
    )
    parser.add_argument(
        "--start-yaw", type=float, default=None,
        help="Override the initial camera yaw (degrees). "
             "By default it is derived from the first edge of the route, "
             "so the agent starts already facing forward along the path.",
    )
    parser.add_argument(
        "--start-pos", type=float, nargs=3, default=None,
        help="Override the initial camera position [x, y, z] in meters. "
             "By default it is taken from the first waypoint of the trajectory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load trajectory to get the start pose (must happen before SimulationApp).
    with open(args.trajectory, "r") as f:
        traj = json.load(f)
    start_pos = list(args.start_pos) if args.start_pos else list(traj["waypoints"][0]["pos"])
    goal_pos = list(traj["waypoints"][-1]["pos"])

    # The route file is position-only (like a real nav-app polyline). The
    # initial yaw is derived from the first edge of the path so the agent
    # starts already facing forward, unless --start-yaw overrides it.
    dense_path = np.asarray(traj["path"], dtype=np.float64)
    if args.start_yaw is not None:
        start_yaw = float(args.start_yaw)
    else:
        dx = float(dense_path[1, 0] - dense_path[0, 0])
        dy = float(dense_path[1, 1] - dense_path[0, 1])
        start_yaw = np.degrees(np.arctan2(dy, dx))
    start_rot = [0.0, 0.0, start_yaw]

    print(f"[Info] trajectory: {args.trajectory}")
    print(f"[Info] start (P1): pos={start_pos} yaw={start_yaw:.1f}deg")
    print(f"[Info] goal  (P{len(traj['waypoints'])}): pos={goal_pos}")
    print(f"[Info] route length: {traj['total_length_m']:.1f}m, "
          f"{len(traj['path'])} dense points")

    # 2. Create the IsaacSimEnv (must precede any omni.* import)
    from env import IsaacSimEnv
    env = IsaacSimEnv(usd_path=args.usd_path, headless=args.headless,
                      gpu_id=args.gpu)

    # 3. Reuse camera + pose + apply_action helpers from nav_eval (no duplication)
    from nav_eval import CameraPose, apply_action, create_d435i_camera

    camera = create_d435i_camera(start_pos, start_rot, headless=args.headless)
    pose = CameraPose(start_pos, start_rot)

    # 4. Output directory: data/urbanverse/vlm_gps_nav/<timestamp>/
    output_dir = Path(_PROJECT_ROOT) / "data" / "urbanverse" / "vlm_gps_nav" / \
        datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] output -> {output_dir}")

    # 5. Instantiate the GPS controller (it owns frames/ + vlm_io.jsonl)
    from nav.gps_vlm_controller import GPSVLMNavController
    controller = GPSVLMNavController(
        trajectory_path=args.trajectory,
        base_url=args.base_url,
        model=args.model,
        instruction=args.instruction,
        forward_step=args.forward_step,
        yaw_step=args.yaw_step,
        lookahead=args.lookahead,
        goal_tol=args.goal_tol,
        output_dir=str(output_dir),
    )
    controller.reset()

    # 6. Controller decimation
    render_hz = 20.0
    ctrl_decimation = max(1, int(round(render_hz / args.controller_freq)))
    print(f"[Info] controller freq: {args.controller_freq} Hz "
          f"(decimation={ctrl_decimation})")

    # 7. Warm-up so the first frame is not blank
    for _ in range(5):
        env.step()

    # 8. Closed loop
    traj_log = output_dir / "trajectory.jsonl"
    trajectory_records = []
    step_count = 0
    ctrl_step = 0
    success = False
    final_dist = None

    print(f"[Info] Running GPS-VLM loop "
          f"(max {args.max_steps} steps, Ctrl+C to stop)...")
    try:
        while env.is_running and ctrl_step < args.max_steps:
            env.step()
            step_count += 1

            # Update camera pose if changed
            new_pos, new_quat, dirty = pose.consume_if_dirty()
            if dirty:
                camera.set_world_pose(position=new_pos, orientation=new_quat)

            # Skip until controller decimation step
            if step_count % ctrl_decimation != 0:
                continue

            # Build observation
            rgba = camera.get_rgba()
            if rgba is None or rgba.size == 0:
                continue
            rgb = rgba[:, :, :3]  # (480, 640, 3) uint8

            from nav import Observation
            pos_list, rot_list = pose.get_pos_rot()
            obs = Observation(
                rgb=rgb,
                pose=(*pos_list, *rot_list),
                step=ctrl_step,
            )

            # Controller step
            action = controller.act(obs)

            # Record trajectory step
            cur_pos = np.asarray(pos_list)
            cur_dist = float(np.linalg.norm(
                np.asarray(goal_pos[:2]) - cur_pos[:2]
            ))
            record = {
                "step": ctrl_step,
                "pose": {
                    "x": pose.x, "y": pose.y, "z": pose.z,
                    "roll": pose.roll, "pitch": pose.pitch, "yaw": pose.yaw,
                },
                "action": {
                    "forward": action.forward, "yaw": action.yaw,
                    "done": action.done,
                },
                "dist_to_goal_m": cur_dist,
            }
            trajectory_records.append(record)
            with open(traj_log, "a") as f:
                f.write(json.dumps(record) + "\n")

            # Apply action -> update pose (will be flushed to camera next loop)
            apply_action(pose, action)
            ctrl_step += 1

            if action.done:
                success = (cur_dist < args.goal_tol)
                final_dist = cur_dist
                print(f"[Info] controller signaled done at step {ctrl_step}, "
                      f"dist_to_goal={cur_dist:.2f}m, success={success}")
                break

            if ctrl_step % 20 == 0:
                p, r = pose.get_pos_rot()
                print(f"[Step {ctrl_step}] pos=({p[0]:.1f}, {p[1]:.1f})  "
                      f"yaw={r[2]:.0f}  dist_to_goal={cur_dist:.1f}m")

    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")

    # Determine final outcome
    if final_dist is None:
        # loop ended without explicit done
        last_pos = np.asarray([pose.x, pose.y])
        final_dist = float(np.linalg.norm(np.asarray(goal_pos[:2]) - last_pos))
        success = (final_dist < args.goal_tol)

    controller.on_episode_end(trajectory_records)

    # 9. summary.json
    summary = {
        "success": bool(success),
        "total_steps": ctrl_step,
        "final_dist_to_goal_m": final_dist,
        "route_length_m": float(traj["total_length_m"]),
        "goal_tol_m": args.goal_tol,
        "controller_freq_hz": args.controller_freq,
        "start_pos": start_pos,
        "start_yaw": start_yaw,
        "instruction": args.instruction,
        "trajectory_path": args.trajectory,
        "usd_path": args.usd_path,
        "model": args.model,
        "base_url": args.base_url,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Done] {ctrl_step} steps, success={success}, "
          f"final_dist={final_dist:.2f}m")
    print(f"[Done] summary -> {output_dir / 'summary.json'}")

    # 10. Stitch frames into 30fps video
    frames = sorted(glob.glob(str(output_dir / "frames" / "frame_*.png")))
    if frames:
        video_path = output_dir / "nav.mp4"
        input_pattern = str(output_dir / "frames" / "frame_%06d.png")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", "30",
            "-i", input_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(video_path),
        ]
        print(f"[Info] stitching {len(frames)} frames -> {video_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[Done] video -> {video_path}")
        else:
            print(f"[Warn] ffmpeg failed: {result.stderr[:200]}")

    env.close()


if __name__ == "__main__":
    main()
