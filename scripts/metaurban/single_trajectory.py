"""
MetaUrban single trajectory collection — headless, simple forward policy.
Collects: position, heading, route_completion, RGB images per step.

Run: conda activate metaurban && python scripts/metaurban_single_trajectory.py
"""
import os
import sys
sys.path.insert(0, os.environ.get("METAURBAN_ROOT", "/home/shu22/navigation/metaurban"))

import numpy as np
import os
import json
import cv2
from pathlib import Path

from metaurban.envs.sidewalk_static_env import SidewalkStaticMetaUrbanEnv
from metaurban.component.sensors.rgb_camera import RGBCamera
from metaurban.component.sensors.depth_camera import DepthCamera

# ── Output ──
OUT_DIR = Path("data/metaurban_test/episode_0000")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ──
config = dict(
    # Scene: longer route
    map="SSSXSS",
    crswalk_density=1,
    object_density=0.3,
    walk_on_all_regions=True,
    drivable_area_extension=55,
    height_scale=1,
    tiny=True,

    # Rendering
    use_render=False,
    image_observation=True,
    stack_size=1,
    norm_pixel=False,
    sensors=dict(
        rgb_camera=(RGBCamera, 640, 480),
        depth_camera=(DepthCamera, 640, 480),
    ),

    # Agent
    agent_type="coco",
    vehicle_config=dict(
        show_lidar=False,
        show_navi_mark=False,
        show_line_to_navi_mark=False,
        show_dest_mark=False,
        enable_reverse=True,
    ),

    # Episode
    horizon=500,
    num_scenarios=10,
    random_spawn_lane_index=False,
    out_of_route_done=True,
    relax_out_of_road_done=True,
    max_lateral_dist=5.0,
    accident_prob=0,

    # Rewards
    success_reward=8.0,
    driving_reward=2.0,
)

print("Creating environment...")
env = SidewalkStaticMetaUrbanEnv(config)

# Reset — skip seed 0 if it immediately terminates
for seed in range(10):
    obs, info = env.reset(seed=seed)
    if not info.get("arrive_dest", False):
        print(f"Using seed={seed}")
        break
    print(f"Seed {seed} arrive_dest on reset, skipping...")

agent = env.agents["default_agent"]
print(f"Start position: {[round(x, 2) for x in agent.position]}")
print(f"Obs keys: {list(obs.keys()) if isinstance(obs, dict) else obs.shape}")
if isinstance(obs, dict):
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

# ── Collect trajectory ──
trajectory = []
max_steps = 300

print(f"\nCollecting trajectory (max {max_steps} steps)...")
for step_idx in range(max_steps):
    # Simple policy: go forward
    action = [0.0, 0.5]

    obs_next, reward, terminated, truncated, info = env.step(action)
    agent = env.agents["default_agent"]

    step_data = {
        "step": step_idx,
        "position": [round(float(x), 4) for x in agent.position],
        "heading": round(float(agent.heading_theta), 4),
        "action": action,
        "reward": round(float(reward), 4),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "route_completion": round(float(info.get("route_completion", 0)), 4),
        "arrive_dest": bool(info.get("arrive_dest", False)),
        "crash": bool(info.get("crash", False)),
    }
    trajectory.append(step_data)

    # ── Save images ──
    step_dir = OUT_DIR / f"step_{step_idx:03d}"
    step_dir.mkdir(exist_ok=True)

    if isinstance(obs_next, dict):
        for key, val in obs_next.items():
            if not isinstance(val, np.ndarray) or val.ndim < 2:
                continue
            # Remove stack dim if present: (H, W, C, 1) -> (H, W, C)
            if val.ndim == 4 and val.shape[3] == 1:
                val = val[:, :, :, 0]
            if "image" in key or "rgb" in key:
                img = val
                if img.dtype in [np.float32, np.float64]:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(step_dir / "rgb_front.jpg"), img)
            elif "depth" in key:
                d = val
                if d.dtype in [np.float32, np.float64]:
                    d = (d * 1000).clip(0, 65535).astype(np.uint16)
                if d.ndim == 3:
                    d = d[:, :, 0]
                cv2.imwrite(str(step_dir / "depth_front.png"), d)

    # Progress
    if step_idx % 50 == 0 or terminated or truncated:
        pos = step_data["position"]
        rc = step_data["route_completion"]
        print(f"  Step {step_idx:3d}: pos=({pos[0]:8.2f}, {pos[1]:8.2f}), "
              f"route={rc:.1%}, reward={reward:.3f}")

    if terminated or truncated:
        reason = "arrive_dest" if info.get("arrive_dest") else \
                 "crash" if info.get("crash") else \
                 "out_of_road" if info.get("out_of_road") else "other"
        print(f"  Episode ended at step {step_idx}: {reason}")
        break

# ── Save metadata ──
meta = {
    "episode_id": "episode_0000",
    "seed": seed,
    "map": "SSSXSS",
    "total_steps": len(trajectory),
    "success": trajectory[-1]["arrive_dest"] if trajectory else False,
    "final_route_completion": trajectory[-1]["route_completion"] if trajectory else 0,
    "trajectory": trajectory,
}
with open(OUT_DIR / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

env.close()

# ── Summary ──
print(f"\n{'='*60}")
print(f"Trajectory saved to: {OUT_DIR}")
print(f"Total steps: {len(trajectory)}")
print(f"Success: {meta['success']}")
print(f"Route completion: {meta['final_route_completion']:.1%}")

# Check saved images
img_count = sum(1 for d in OUT_DIR.iterdir() if d.is_dir())
sample_img = OUT_DIR / "step_000" / "rgb_front.jpg"
if sample_img.exists():
    img = cv2.imread(str(sample_img))
    print(f"Sample image: {sample_img} → {img.shape}")
else:
    print(f"No images saved (check obs format)")
print(f"Total step directories: {img_count}")
print(f"{'='*60}")
