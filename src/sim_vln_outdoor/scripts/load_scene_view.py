"""Launch Isaac Sim, load a USD scene, and view from a D435i camera.

The Isaac Sim viewport is set to the D435i camera prim, so you see
exactly the camera's perspective in the GUI window.

Supports two real-time control modes:
  1. Keyboard (--keyboard): WASD move, QE up/down, arrow keys rotate, +/- speed
  2. TCP socket (--socket-port): external process sends JSON pose updates

Usage:
    cd /home/shu22/nvidia/isaacsim_5.1.0

    # Just view (viewport shows camera angle)
    ./python.sh .../load_scene_view.py --camera-pos -730.0 490.0 1.5

    # With keyboard control
    ./python.sh .../load_scene_view.py --keyboard --camera-pos -730.0 490.0 1.5

    # Socket control (for agent integration)
    ./python.sh .../load_scene_view.py --socket-port 9090
    # Then from another process:
    #   echo '{"x":-730,"y":490,"z":1.5,"yaw":90}' | nc localhost 9090

    # Save frames
    ./python.sh .../load_scene_view.py --save-dir /tmp/frames
"""

import argparse
import json
import os
import socket
import sys
import threading
from datetime import datetime

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_SCRIPT_DIR)  # sim_vln_outdoor/
sys.path.insert(0, _PACKAGE_ROOT)

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

# Intel RealSense D435i RGB camera intrinsics (640x480 default)
D435I_WIDTH = 640
D435I_HEIGHT = 480
D435I_FX = 615.0
D435I_FY = 615.0
D435I_CX = 320.0
D435I_CY = 240.0
D435I_FOCAL_LENGTH = 1.88  # mm
D435I_H_APERTURE = D435I_FOCAL_LENGTH * D435I_WIDTH / D435I_FX   # ~1.956 mm
D435I_V_APERTURE = D435I_FOCAL_LENGTH * D435I_HEIGHT / D435I_FY  # ~1.468 mm


# ── Camera pose state (shared between main loop and socket thread) ──────────

class CameraPose:
    """Thread-safe mutable camera pose."""

    def __init__(self, pos, rot):
        self.x, self.y, self.z = pos
        self.roll, self.pitch, self.yaw = rot  # degrees
        self.speed = 0.5   # m per step for keyboard
        self.rot_speed = 5.0  # deg per step for keyboard
        self._lock = threading.Lock()
        self._dirty = True  # flag: pose changed since last consume
        self._snapshot_requested = False

    def set_pos(self, x, y, z):
        with self._lock:
            self.x, self.y, self.z = x, y, z
            self._dirty = True

    def set_rot(self, roll, pitch, yaw):
        with self._lock:
            self.roll, self.pitch, self.yaw = roll, pitch, yaw
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


# ── Helpers ─────────────────────────────────────────────────────────────────

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a D435i camera view with real-time pose control."
    )
    parser.add_argument(
        "--usd-path", type=str, default=DEFAULT_USD,
        help="Path to the USD scene file to load.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--camera-pos", type=float, nargs=3, default=[-693.5133185566787, 496.8050450996168, 2.0],
        help="Initial camera position [x, y, z] in meters.",
    )
    parser.add_argument(
        "--camera-rot", type=float, nargs=3, default=[0.0, 0.0, 360.0],
        help="Initial camera orientation [roll, pitch, yaw] in degrees.",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Save rendered frames to this directory.",
    )
    parser.add_argument(
        "--keyboard", action="store_true",
        help="Enable WASD keyboard control in the Isaac Sim viewport.",
    )
    parser.add_argument(
        "--socket-port", type=int, default=None,
        help="Start a TCP socket server on this port for external pose input.",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index for rendering (default: 0).",
    )
    return parser.parse_args()


# ── Camera creation ─────────────────────────────────────────────────────────

def create_d435i_camera(camera_pos, camera_rot):
    """Create a camera prim with D435i intrinsics."""
    from omni.isaac.sensor import Camera
    from pxr import UsdGeom

    quat = euler_to_quat_wxyz(*camera_rot)
    camera = Camera(
        prim_path="/World/D435i_Camera",
        position=np.array(camera_pos, dtype=np.float64),
        orientation=quat,
        frequency=20,
        resolution=(D435I_WIDTH, D435I_HEIGHT),
    )

    cam_api = UsdGeom.Camera(camera.prim)
    cam_api.GetFocalLengthAttr().Set(D435I_FOCAL_LENGTH)
    cam_api.GetHorizontalApertureAttr().Set(D435I_H_APERTURE)
    cam_api.GetVerticalApertureAttr().Set(D435I_V_APERTURE)
    cam_api.GetClippingRangeAttr().Set((0.1, 100.0))

    camera.initialize()

    # Set the Isaac Sim viewport to use this camera
    from omni.kit.viewport.utility import get_active_viewport
    viewport = get_active_viewport()
    viewport.camera_path = "/World/D435i_Camera"
    print("[Info] Viewport active camera set to /World/D435i_Camera")

    print(f"[Info] D435i camera created at pos={camera_pos}, rot={camera_rot}")
    print(f"[Info] Intrinsics: fx={D435I_FX}, fy={D435I_FY}, "
          f"cx={D435I_CX}, cy={D435I_CY}, res={D435I_WIDTH}x{D435I_HEIGHT}")
    return camera


# ── Keyboard control (Isaac Sim native) ────────────────────────────────────

def setup_keyboard_control(pose: CameraPose):
    """Subscribe to Isaac Sim keyboard events via carb.input.

    Controls:
        W/S     - forward / backward (along yaw direction)
        A/D     - strafe left / right
        Q/E     - move up / down
        ↑/↓     - pitch up / down
        ←/→     - yaw left / right
        =/−     - increase / decrease move speed
        P       - snapshot (save current D435i view)
    """
    import carb.input
    import omni.appwindow

    input_iface = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    K = carb.input.KeyboardInput

    def _on_key(event):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True

        key = event.input

        # Snapshot request (handled in main loop)
        if key == K.P:
            pose._snapshot_requested = True
            return True

        yaw_rad = np.radians(pose.yaw)
        dx_fwd = np.cos(yaw_rad) * pose.speed
        dy_fwd = np.sin(yaw_rad) * pose.speed
        dx_right = np.cos(yaw_rad + np.pi / 2) * pose.speed
        dy_right = np.sin(yaw_rad + np.pi / 2) * pose.speed

        with pose._lock:
            if key == K.W:
                pose.x += dx_fwd; pose.y += dy_fwd
            elif key == K.S:
                pose.x -= dx_fwd; pose.y -= dy_fwd
            elif key == K.A:
                pose.x += dx_right; pose.y += dy_right
            elif key == K.D:
                pose.x -= dx_right; pose.y -= dy_right
            elif key == K.Q:
                pose.z += pose.speed
            elif key == K.E:
                pose.z -= pose.speed
            elif key == K.UP:
                pose.pitch -= pose.rot_speed
            elif key == K.DOWN:
                pose.pitch += pose.rot_speed
            elif key == K.LEFT:
                pose.yaw += pose.rot_speed
            elif key == K.RIGHT:
                pose.yaw -= pose.rot_speed
            elif key == K.EQUAL:
                pose.speed = min(pose.speed * 1.5, 50.0)
                print(f"[Info] Move speed: {pose.speed:.2f}")
            elif key == K.MINUS:
                pose.speed = max(pose.speed / 1.5, 0.01)
                print(f"[Info] Move speed: {pose.speed:.2f}")
            else:
                return True  # unknown key, don't mark dirty
            pose._dirty = True
        return True

    sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_key)
    return sub  # caller must keep a reference to prevent GC


# ── Socket server ───────────────────────────────────────────────────────────

def start_socket_server(port, pose: CameraPose):
    """Run a TCP server that accepts JSON pose updates.

    Each line is a JSON object with optional fields:
        {"x": float, "y": float, "z": float,
         "roll": float, "pitch": float, "yaw": float}
    Missing fields keep their current value.
    Responds with the updated pose as JSON.
    """

    def _handle_client(conn, addr):
        print(f"[Socket] Client connected: {addr}")
        buf = b""
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        conn.sendall(b'{"error":"invalid json"}\n')
                        continue

                    with pose._lock:
                        pose.x = msg.get("x", pose.x)
                        pose.y = msg.get("y", pose.y)
                        pose.z = msg.get("z", pose.z)
                        pose.roll = msg.get("roll", pose.roll)
                        pose.pitch = msg.get("pitch", pose.pitch)
                        pose.yaw = msg.get("yaw", pose.yaw)
                        pose._dirty = True
                        reply = {
                            "x": pose.x, "y": pose.y, "z": pose.z,
                            "roll": pose.roll, "pitch": pose.pitch,
                            "yaw": pose.yaw,
                        }
                    conn.sendall((json.dumps(reply) + "\n").encode())
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            conn.close()
            print(f"[Socket] Client disconnected: {addr}")

    def _serve():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", port))
        srv.listen(2)
        print(f"[Socket] Listening on port {port}")
        print(f'[Socket] Example: echo \'{{"x":-730,"y":490,"z":2,"yaw":45}}\' | nc localhost {port}')
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=_handle_client, args=(conn, addr),
                             daemon=True).start()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return t


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    from env import IsaacSimEnv
    env = IsaacSimEnv(usd_path=args.usd_path, headless=args.headless,
                      gpu_id=args.gpu)

    camera = create_d435i_camera(args.camera_pos, args.camera_rot)
    pose = CameraPose(args.camera_pos, args.camera_rot)

    if args.socket_port:
        start_socket_server(args.socket_port, pose)

    # Subscribe to Isaac Sim keyboard events (must keep reference)
    _kb_sub = None
    if args.keyboard:
        _kb_sub = setup_keyboard_control(pose)
        print("[Info] Keyboard controls (click viewport to focus):")
        print("  W/S=forward/back  A/D=strafe  Q/E=up/down")
        print("  Arrows=rotate  =/- =speed  P=snapshot  Ctrl+C=quit")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"[Info] Saving frames to {args.save_dir}")

    # Snapshot directory: data/urbanverse/load_scene_view/<start_time>/
    _SNAPSHOT_BASE = os.path.join(
        _PACKAGE_ROOT, os.pardir, os.pardir, "data", "urbanverse", "load_scene_view",
    )
    _snapshot_dir = os.path.join(
        _SNAPSHOT_BASE, datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    _snapshot_count = 0

    print("[Info] Running simulation loop (Ctrl+C to exit)...")
    step_count = 0
    try:
        while env.is_running:
            env.step()
            step_count += 1

            if step_count < 5:
                continue

            # Update camera pose if changed
            new_pos, new_quat, dirty = pose.consume_if_dirty()
            if dirty:
                camera.set_world_pose(position=new_pos, orientation=new_quat)

            # Snapshot on P key
            if pose._snapshot_requested:
                pose._snapshot_requested = False
                rgba = camera.get_rgba()
                if rgba is not None and rgba.size > 0:
                    from PIL import Image
                    os.makedirs(_snapshot_dir, exist_ok=True)
                    pos_list, rot_list = pose.get_pos_rot()
                    fname = f"snap_{_snapshot_count:04d}.png"
                    fpath = os.path.join(_snapshot_dir, fname)
                    Image.fromarray(rgba[:, :, :3]).save(fpath)
                    _snapshot_count += 1
                    print(f"[Snapshot] Saved {fpath}  "
                          f"pos=({pos_list[0]:.1f}, {pos_list[1]:.1f}, "
                          f"{pos_list[2]:.1f})  rot=({rot_list[0]:.0f}, "
                          f"{rot_list[1]:.0f}, {rot_list[2]:.0f})")

            # Save frame periodically
            if args.save_dir and step_count % 10 == 0:
                rgba = camera.get_rgba()
                if rgba is not None and rgba.size > 0:
                    from PIL import Image
                    frame_path = os.path.join(
                        args.save_dir, f"frame_{step_count:06d}.png",
                    )
                    Image.fromarray(rgba[:, :, :3]).save(frame_path)

            if step_count % 500 == 0:
                pos_list, rot_list = pose.get_pos_rot()
                print(f"[Step {step_count}] pos={pos_list} rot={rot_list}")

    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")

    env.close()


if __name__ == "__main__":
    main()
