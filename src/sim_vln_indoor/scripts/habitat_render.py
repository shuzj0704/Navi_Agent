"""
Habitat-Sim 离线渲染器
======================
沿导航网格随机游走，渲染4视角图像，保存为视频和图片。
"""
import os, sys
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
sys.path.insert(0, _SRC_ROOT)
os.chdir(_PROJECT_ROOT)
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["MAGNUM_GPU_VALIDATION"] = "OFF"

import habitat_sim
import numpy as np
import cv2
import math
import json

# ========== 配置 ==========
SCENE_DIR = "data/scene_data/mp3d"
OUTPUT_DIR = "output/render"
FPS = 10
MAX_STEPS = 200

# ========== 4个相机配置 ==========
# 每个相机: name, position [x,y,z](米), orientation [pitch,yaw,roll](度), hfov(度), width, height
CAMERA_CONFIGS = [
    {
        "name": "front",
        "position": [0.0, 1.5, 0.0],      # 眼睛高度1.5m，居中
        "pitch": -20.0,                       # 俯仰角: 正值低头, 负值抬头
        "yaw": 0.0,                         # 偏航角: 正前方
        "roll": 0.0,                        # 翻滚角: 通常为0
        "hfov": 90,                         # 水平视场角(度)
        "width": 640,
        "height": 480,
    },
    {
        "name": "left",
        "position": [0.0, 1.5, 0.0],       # 同一位置
        "pitch": -20.0,
        "yaw": 90.0,                        # 向左看90度
        "roll": 0.0,
        "hfov": 90,
        "width": 640,
        "height": 480,
    },
    {
        "name": "right",
        "position": [0.0, 1.5, 0.0],
        "pitch": -20.0,
        "yaw": -90.0,                       # 向右看90度
        "roll": 0.0,
        "hfov": 90,
        "width": 640,
        "height": 480,
    },
    {
        "name": "back",
        "position": [0.0, 1.5, 0.0],
        "pitch": -20.0,
        "yaw": 180.0,                       # 向后看
        "roll": 0.0,
        "hfov": 90,
        "width": 640,
        "height": 480,
    },
]


def make_cfg(scene_path):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0

    sensors = []
    for cam in CAMERA_CONFIGS:
        # RGB 传感器
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = cam["name"]
        spec.sensor_type = habitat_sim.SensorType.COLOR
        spec.resolution = [cam["height"], cam["width"]]
        spec.hfov = cam["hfov"]
        spec.position = cam["position"]
        # orientation = [pitch, yaw, roll] 弧度
        spec.orientation = [
            math.radians(cam["pitch"]),
            math.radians(cam["yaw"]),
            math.radians(cam["roll"]),
        ]
        sensors.append(spec)

        # 为 front 相机添加深度传感器（与 RGB 完全同参数）
        if cam["name"] == "front":
            depth_spec = habitat_sim.CameraSensorSpec()
            depth_spec.uuid = "front_depth"
            depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_spec.resolution = [cam["height"], cam["width"]]
            depth_spec.hfov = cam["hfov"]
            depth_spec.position = cam["position"]
            depth_spec.orientation = [
                math.radians(cam["pitch"]),
                math.radians(cam["yaw"]),
                math.radians(cam["roll"]),
            ]
            sensors.append(depth_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def random_walk(sim, steps=MAX_STEPS):
    """在导航网格上随机游走，返回每步的观测和位置"""
    agent = sim.get_agent(0)
    actions = ["move_forward", "move_forward", "move_forward", "turn_left", "turn_right"]

    trajectory = []
    for step in range(steps):
        obs = sim.get_sensor_observations()
        state = agent.get_state()
        pos = state.position
        rot = state.rotation

        # 提取 yaw
        siny = 2.0 * (rot.w * rot.y + rot.x * rot.z)
        cosy = 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z)
        yaw = math.degrees(math.atan2(siny, cosy))

        trajectory.append({
            "step": step,
            "position": [float(pos[0]), float(pos[1]), float(pos[2])],
            "yaw": float(yaw),
        })

        # 保存4视角图像
        for cam in CAMERA_CONFIGS:
            view = cam["name"]
            img = obs[view][:, :, :3]  # RGBA -> RGB
            trajectory[-1][f"img_{view}"] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 保存 front 深度图（float32，单位：米）
        trajectory[-1]["front_depth"] = obs["front_depth"].copy()

        # 随机选动作
        action = np.random.choice(actions)
        sim.get_agent(0).act(action)
        trajectory[-1]["action"] = action

    return trajectory


def save_output(trajectory, scene_name, output_dir):
    """保存为视频 + 拼接图片 + 轨迹json"""
    scene_dir = os.path.join(output_dir, scene_name)
    img_dir = os.path.join(scene_dir, "frames")
    os.makedirs(img_dir, exist_ok=True)

    # 视频 writer（4视角拼成2x2网格，统一resize到第一个相机的尺寸）
    cell_w = CAMERA_CONFIGS[0]["width"]
    cell_h = CAMERA_CONFIGS[0]["height"]
    grid_w, grid_h = cell_w * 2, cell_h * 2
    video_path = os.path.join(scene_dir, "walkthrough.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, FPS, (grid_w, grid_h))

    traj_data = []

    for t in trajectory:
        step = t["step"]
        views = {}
        for cam in CAMERA_CONFIGS:
            name = cam["name"]
            img = t[f"img_{name}"]
            # 统一 resize 到网格单元尺寸
            if img.shape[1] != cell_w or img.shape[0] != cell_h:
                img = cv2.resize(img, (cell_w, cell_h))
            # 标注相机名称 + 参数
            label = f"{name.upper()} hfov={cam['hfov']} yaw={cam['yaw']}"
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            views[name] = img

        # 2x2 网格: 左上=left, 右上=front, 左下=right, 右下=back
        top = np.hstack([views["left"], views["front"]])
        bottom = np.hstack([views["right"], views["back"]])
        grid = np.vstack([top, bottom])

        # 叠加步数和位置信息
        info = f"Step {step}/{len(trajectory)}  Pos: ({t['position'][0]:.1f}, {t['position'][2]:.1f})  Yaw: {t['yaw']:.0f}  Act: {t['action']}"
        cv2.putText(grid, info, (10, grid_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        video.write(grid)

        # 每20步保存一张拼接图
        if step % 20 == 0:
            cv2.imwrite(os.path.join(img_dir, f"step_{step:04d}.jpg"), grid)

        traj_data.append({
            "step": step,
            "position": t["position"],
            "yaw": t["yaw"],
            "action": t["action"],
        })

    video.release()

    # 保存轨迹 json
    with open(os.path.join(scene_dir, "trajectory.json"), "w") as f:
        json.dump(traj_data, f, indent=2)

    print(f"  视频: {video_path}")
    print(f"  帧图: {img_dir}/ ({len(os.listdir(img_dir))} 张)")
    print(f"  轨迹: {os.path.join(scene_dir, 'trajectory.json')}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=int, default=0, help="场景编号")
    parser.add_argument("--count", type=int, default=1, help="渲染几个场景")
    parser.add_argument("--steps", type=int, default=MAX_STEPS, help="游走步数")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scenes = []
    for name in sorted(os.listdir(SCENE_DIR)):
        glb = os.path.join(SCENE_DIR, name, f"{name}.glb")
        if os.path.exists(glb):
            scenes.append((name, glb))

    print(f"找到 {len(scenes)} 个场景")
    selected = scenes[args.scene:args.scene + args.count]

    for scene_name, scene_path in selected:
        print(f"\n渲染场景: {scene_name}")

        # 加载导航网格
        navmesh_path = scene_path.replace(".glb", ".navmesh")

        cfg = make_cfg(scene_path)
        sim = habitat_sim.Simulator(cfg)

        if os.path.exists(navmesh_path):
            sim.pathfinder.load_nav_mesh(navmesh_path)
            # 设置随机起点
            agent = sim.get_agent(0)
            state = agent.get_state()
            state.position = sim.pathfinder.get_random_navigable_point()
            agent.set_state(state)

        print(f"  随机游走 {args.steps} 步...")
        trajectory = random_walk(sim, args.steps)

        print(f"  保存输出...")
        save_output(trajectory, scene_name, OUTPUT_DIR)

        sim.close()

    print(f"\n完成! 输出在 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
