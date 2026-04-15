"""Closed-loop navigation controller evaluation in Isaac Sim.

Renders D435i camera images, feeds them to a NavController, applies the
returned actions to update camera pose, and logs the trajectory.

Usage:
    cd ~/navigation/Navi_Agent

    # Default forward-only controller
    ./python.sh src/sim_vln_outdoor/scripts/nav_eval.py \
        --camera-pos -730.0 490.0 1.5 --max-steps 500

    # Random walk, headless, save frames
    ./python.sh src/sim_vln_outdoor/scripts/nav_eval.py \
        --headless --save-frames --max-steps 200 \
        --controller "nav.demo_controllers:RandomWalkController"

    # Custom VLM controller at 2Hz
    ./python.sh src/sim_vln_outdoor/scripts/nav_eval.py \
        --headless --controller "my_nav.vlm_ctrl:VLMNavController" \
        --controller-freq 2.0 --max-steps 100 --save-frames
"""

import argparse
import glob
import importlib
import json
import os
import subprocess
import sys
import threading
from datetime import datetime

import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_SCRIPT_DIR)  # sim_vln_outdoor/
sys.path.insert(0, _PACKAGE_ROOT)

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

# ── D435i intrinsics (duplicated from load_scene_view.py) ─────────────────
D435I_WIDTH = 640
D435I_HEIGHT = 480
D435I_FX = 615.0
D435I_FY = 615.0
D435I_CX = 320.0
D435I_CY = 240.0
D435I_FOCAL_LENGTH = 1.88  # mm
D435I_H_APERTURE = D435I_FOCAL_LENGTH * D435I_WIDTH / D435I_FX
D435I_V_APERTURE = D435I_FOCAL_LENGTH * D435I_HEIGHT / D435I_FY


# ── Helpers (duplicated from load_scene_view.py) ──────────────────────────

def euler_to_quat_wxyz(roll_deg, pitch_deg, yaw_deg):
    """Convert roll/pitch/yaw (degrees) to quaternion (w, x, y, z)."""
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_ = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y_, z], dtype=np.float64)


class CameraPose:
    """Thread-safe mutable camera pose."""

    def __init__(self, pos, rot):
        self.x, self.y, self.z = pos
        self.roll, self.pitch, self.yaw = rot  # degrees
        self._lock = threading.Lock()
        self._dirty = True

    def set_pose(self, x, y, z, roll, pitch, yaw):
        with self._lock:
            self.x, self.y, self.z = x, y, z
            self.roll, self.pitch, self.yaw = roll, pitch, yaw
            self._dirty = True

    def consume_if_dirty(self):
        """Return (pos, quat, dirty_flag). Resets dirty flag."""
        with self._lock:
            dirty = self._dirty
            self._dirty = False
            pos = np.array([self.x, self.y, self.z], dtype=np.float64)
            quat = euler_to_quat_wxyz(self.roll, self.pitch, self.yaw)
            return pos, quat, dirty

    def get_pos_rot(self):
        with self._lock:
            return (
                [self.x, self.y, self.z],
                [self.roll, self.pitch, self.yaw],
            )


# ── Camera creation (duplicated from load_scene_view.py) ──────────────────

def create_d435i_camera(camera_pos, camera_rot, headless=False):
    """Create a camera prim with D435i intrinsics."""
    from omni.isaac.sensor import Camera
    from pxr import UsdGeom

    quat = euler_to_quat_wxyz(*camera_rot)
    camera = Camera(
        prim_path="/World/D435i_Camera",
        position=np.array(camera_pos, dtype=np.float64),
        orientation=quat,
        frequency=10,
        resolution=(D435I_WIDTH, D435I_HEIGHT),
    )

    cam_api = UsdGeom.Camera(camera.prim)
    cam_api.GetFocalLengthAttr().Set(D435I_FOCAL_LENGTH)
    cam_api.GetHorizontalApertureAttr().Set(D435I_H_APERTURE)
    cam_api.GetVerticalApertureAttr().Set(D435I_V_APERTURE)
    cam_api.GetClippingRangeAttr().Set((0.1, 100.0))

    camera.initialize()

    if not headless:
        try:
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            viewport.camera_path = "/World/D435i_Camera"
            print("[Info] Viewport active camera set to /World/D435i_Camera")
        except Exception:
            pass

    print(f"[Info] D435i camera created at pos={camera_pos}, rot={camera_rot}")
    return camera


# ── Action application ─────────────────────────────────────────────────────

def apply_action(pose, action):
    """Apply a navigation Action to update camera pose.

    Movement logic mirrors the keyboard controls in load_scene_view.py.
    """
    yaw_rad = np.radians(pose.yaw)
    dx_fwd = np.cos(yaw_rad)
    dy_fwd = np.sin(yaw_rad)
    dx_right = np.cos(yaw_rad + np.pi / 2)
    dy_right = np.sin(yaw_rad + np.pi / 2)

    with pose._lock:
        # Forward / backward
        pose.x += dx_fwd * action.forward
        pose.y += dy_fwd * action.forward
        # Strafe (positive = left, matching A key)
        pose.x += dx_right * action.strafe
        pose.y += dy_right * action.strafe
        # Vertical
        pose.z += action.vertical
        # Rotation (positive pitch = tilt up = decrease pitch angle)
        pose.pitch -= action.pitch
        # Positive yaw = turn left = increase yaw angle
        pose.yaw += action.yaw
        pose._dirty = True


# ── Controller loading ─────────────────────────────────────────────────────

def load_controller(spec, kwargs=None):
    """Load a NavController from a 'module.path:ClassName' string.

    Args:
        spec: Import spec like "nav.demo_controllers:ForwardOnlyController".
        kwargs: Optional dict of keyword arguments passed to the constructor.
    """
    module_path, class_name = spec.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**(kwargs or {}))


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Closed-loop navigation controller evaluation in Isaac Sim.",
    )
    parser.add_argument(
        "--usd-path", type=str, default=DEFAULT_USD,
        help="Path to the USD scene file.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--camera-pos", type=float, nargs=3,
        default=[-693.5133185566787, 496.8050450996168, 2.0],
        help="Initial camera position [x, y, z] in meters.",
    )
    parser.add_argument(
        "--camera-rot", type=float, nargs=3, default=[0.0, 0.0, 360.0],
        help="Initial camera orientation [roll, pitch, yaw] in degrees.",
    )
    parser.add_argument(
        "--controller", type=str,
        default="nav.demo_controllers:ForwardOnlyController",
        help="Controller class as 'module.path:ClassName'.",
    )
    parser.add_argument(
        "--controller-kwargs", type=str, default=None,
        help='JSON string of kwargs for the controller __init__, e.g. '
             '\'{"instruction":"find the construction site"}\'.',
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000,
        help="Maximum number of controller steps.",
    )
    parser.add_argument(
        "--controller-freq", type=float, default=20.0,
        help="Controller frequency in Hz (camera runs at 20Hz).",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save trajectory and frames.",
    )
    parser.add_argument(
        "--save-frames", action="store_true",
        help="Save RGB frame at each controller step.",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index for rendering (default: 0).",
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 1. Create env (must happen before any omni imports)
    from env import IsaacSimEnv
    env = IsaacSimEnv(usd_path=args.usd_path, headless=args.headless,
                      gpu_id=args.gpu)

    # 2. Create camera and pose
    camera = create_d435i_camera(args.camera_pos, args.camera_rot,
                                 headless=args.headless)
    pose = CameraPose(args.camera_pos, args.camera_rot)

    # 3. Load controller
    ctrl_kwargs = (
        json.loads(args.controller_kwargs) if args.controller_kwargs else None
    )
    controller = load_controller(args.controller, ctrl_kwargs)
    controller.reset()
    print(f"[Info] Controller: {args.controller}")
    if ctrl_kwargs:
        print(f"[Info] Controller kwargs: {ctrl_kwargs}")

    # 4. Output directory
    if args.save_dir is None:
        base = os.path.join(_PROJECT_ROOT, "data", "urbanverse", "nav_eval")
        args.save_dir = os.path.join(
            base, datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    os.makedirs(args.save_dir, exist_ok=True)
    traj_path = os.path.join(args.save_dir, "trajectory.jsonl")
    print(f"[Info] Output: {args.save_dir}")

    # 5. Controller decimation
    render_hz = 20.0
    ctrl_decimation = max(1, int(round(render_hz / args.controller_freq)))
    print(f"[Info] Controller freq: {args.controller_freq} Hz "
          f"(decimation={ctrl_decimation})")

    # 6. Warm-up (first few frames may be blank)
    for _ in range(5):
        env.step()

    # 7. Closed loop
    trajectory = []
    step_count = 0
    ctrl_step = 0

    print(f"[Info] Running eval loop (max {args.max_steps} steps, Ctrl+C to stop)...")
    try:
        while env.is_running and ctrl_step < args.max_steps:
            env.step()
            step_count += 1

            # Update camera pose if changed
            new_pos, new_quat, dirty = pose.consume_if_dirty()
            if dirty:
                camera.set_world_pose(position=new_pos, orientation=new_quat)

            # Skip until controller step
            if step_count % ctrl_decimation != 0:
                continue

            # Get observation
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

            # Controller produces action
            action = controller.act(obs)

            # Record trajectory (s_t, a_t)
            record = {
                "step": ctrl_step,
                "pose": {
                    "x": pose.x, "y": pose.y, "z": pose.z,
                    "roll": pose.roll, "pitch": pose.pitch, "yaw": pose.yaw,
                },
                "action": {
                    "forward": action.forward, "strafe": action.strafe,
                    "vertical": action.vertical,
                    "pitch": action.pitch, "yaw": action.yaw,
                },
            }

            # Save frame if requested
            if args.save_frames:
                from PIL import Image
                fname = f"frame_{ctrl_step:06d}.png"
                fpath = os.path.join(args.save_dir, fname)
                Image.fromarray(rgb).save(fpath)
                record["frame"] = fname

            trajectory.append(record)
            with open(traj_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            # Apply action to update pose
            apply_action(pose, action)

            ctrl_step += 1

            if action.done:
                print(f"[Info] Controller signaled done at step {ctrl_step}.")
                break

            if ctrl_step % 100 == 0:
                p, r = pose.get_pos_rot()
                print(f"[Step {ctrl_step}] pos=({p[0]:.1f}, {p[1]:.1f}, "
                      f"{p[2]:.1f})  rot=({r[0]:.0f}, {r[1]:.0f}, {r[2]:.0f})")

    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")

    # Finalize
    controller.on_episode_end(trajectory)

    summary = {
        "total_steps": ctrl_step,
        "controller": args.controller,
        "initial_pos": args.camera_pos,
        "initial_rot": args.camera_rot,
        "controller_freq_hz": args.controller_freq,
        "save_dir": args.save_dir,
    }
    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Done] {ctrl_step} steps -> {traj_path}")
    print(f"[Done] Summary -> {summary_path}")

    # 8. Stitch frames into 30fps video
    if args.save_frames and ctrl_step > 0:
        frames = sorted(glob.glob(os.path.join(args.save_dir, "frame_*.png")))
        if frames:
            video_path = os.path.join(args.save_dir, "nav_eval.mp4")
            input_pattern = os.path.join(args.save_dir, "frame_%06d.png")
            cmd = [
                "ffmpeg", "-y",
                "-framerate", "30",
                "-i", input_pattern,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                video_path,
            ]
            print(f"[Info] Stitching {len(frames)} frames -> {video_path}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[Done] Video -> {video_path}")
            else:
                print(f"[Warn] ffmpeg failed (is it installed?): {result.stderr[:200]}")

    env.close()


if __name__ == "__main__":
    main()
