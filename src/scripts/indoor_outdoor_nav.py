#!/usr/bin/env python
"""室内外合并导航入口

从配置文件加载室内外场景，运行室内导航成功后切到室外导航。

用法:
    # 终端 1: 启动 Habitat 仿真服务器
    cd /home/ps/workspace/ll/workspace/Navi_Agent
    source ~/workspace/ll/env/miniforge3/bin/activate internav_habitat
    xvfb-run -a python -m sim_vln_indoor.env.server --port 5100

    # 终端 2: 启动 vLLM (如果需要 VLM)
    source ~/workspace/ll/env/miniforge3/bin/activate /home/ps/miniconda3/envs/lwy_swift
    CUDA_VISIBLE_DEVICES=3 vllm serve /mnt/sda/szj/navi_dataset/checkpoints/Qwen3-VL-8B-Instruct/ \
        --served-model-name qwen3-vl --port 8004 --host 0.0.0.0 --max-model-len 4096 --gpu-memory-utilization 0.85

    # 终端 3: 运行合并导航
    cd /home/ps/workspace/ll/workspace/Navi_Agent
    source ~/workspace/ll/env/miniforge3/bin/activate internav_habitat
    python src/scripts/indoor_outdoor_nav.py --config configs/indoor_outdoor.yaml
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

import yaml
import cv2
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
os.chdir(_PROJECT_ROOT)

DEFAULT_SIM_URL = "http://localhost:5100"
ISAACSIM_ROOT = "/home/ps/sources/isaacsim_4.5.0"

SENSOR_CONFIGS = {
    "front_depth": {"width": 640, "height": 480, "hfov": 120},
    "low_depth":   {"width": 640, "height": 480, "hfov": 90},
}


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_indoor_nav(cfg: dict, sim_url: str, output_dir: Path) -> dict:
    """运行室内导航，返回结果和视频路径"""
    indoor_gpu = cfg.get("indoor", {}).get("gpu", 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(indoor_gpu)
    print(f"[室内] 使用 GPU {indoor_gpu}")
    
    from naviagent.perception import get_camera_intrinsics, YOLOESegmentor, SemanticMapper, SimClientObsReader
    from naviagent.decision import DWAPlanner, TurnController, VLMNavigator, TaskOrchestrator, NavigationEngine
    from naviagent.vlm.vlm_config import load_nav_vlm_config
    from naviagent.common import draw_debug_frame, build_panel_info
    from sim_vln_indoor.env import SimClient

    indoor_cfg = cfg["indoor"]
    vlm_cfg = cfg["vlm"]

    print(f"\n{'='*60}")
    print(f"[室内导航] 场景: {indoor_cfg['scene']}")
    print(f"{'='*60}\n")

    client = SimClient(sim_url)
    print(f"[室内] 连接仿真服务器: {sim_url}")
    client.wait_ready(timeout=10)

    scene_name = indoor_cfg["scene"]
    print(f"[室内] 加载场景: {scene_name}")
    client.load_scene(scene_name=scene_name)

    dwa = DWAPlanner()
    yoloe = YOLOESegmentor(model_path="models/yoloe-11l-seg.pt")
    print(f"[室内] YOLOE loaded. Detecting: {yoloe.classes}")

    mapper = SemanticMapper(
        segmentor=yoloe,
        overlap_threshold=0.3,
        camera_height=1.5,
        camera_pitch_deg=-20.0,
    )
    turn_ctrl = TurnController()

    use_mock = vlm_cfg.get("mock", False)
    if use_mock:
        vlm = None
        print("[室内] 使用 Mock VLM (目标: 画面中心)")
    else:
        vlm = VLMNavigator(
            api_url=vlm_cfg["api_url"],
            api_key=vlm_cfg["api_key"],
            model=vlm_cfg["model"],
            temperature=vlm_cfg.get("temperature", 1.0),
            max_tokens=vlm_cfg.get("max_tokens", 100),
        )
        print(f"[室内] VLM 已连接: {vlm.api_url} model={vlm.model}")

    instruction = indoor_cfg.get("instruction", "探索这个环境并找到通往室外的可能的出口，停在该出口前。")
    print(f"[室内] 任务指令: {instruction}")

    front_cfg = SENSOR_CONFIGS["front_depth"]
    front_intrinsics = get_camera_intrinsics(
        front_cfg["width"], front_cfg["height"], front_cfg["hfov"]
    )

    reader = SimClientObsReader(client, SENSOR_CONFIGS)
    engine = NavigationEngine(
        vlm=vlm, dwa=dwa, turn_ctrl=turn_ctrl,
        front_intrinsics=front_intrinsics,
        instruction=instruction if vlm else "探索环境",
        orchestrator=None,
        mapper=mapper,
    )

    vis_dir = output_dir / "indoor"
    vis_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    max_steps = indoor_cfg.get("max_steps", 100)
    door_dist_threshold = indoor_cfg.get("door_distance_threshold", 2.0)

    indoor_result = {
        "scene": scene_name,
        "steps": 0,
        "success": False,
        "done_reason": "",
        "door_distance_m": None,
        "door_position": None,
        "video": str(vis_dir / "nav_debug.mp4"),
    }

    def do_viz(step_num, result, obs):
        nonlocal video_writer
        sem_map = mapper.render_topdown(
            agent_x=engine.nav.x, agent_y=engine.nav.y,
            agent_yaw=engine.nav.yaw,
            map_size=480, scale=40, trajectory=engine.trajectory,
        )
        panel_info = build_panel_info(
            step_num, instruction,
            None, vlm.get_viz_state() if vlm else None,
            result.vlm_view, result.target_vx, result.target_vy,
            result.action_type or "forward",
        )
        frame = draw_debug_frame(
            engine.nav, obs.views_bgr, sem_map,
            result.dwa_debug, obs.obstacles_local, step_num,
            vlm_view=result.vlm_view, vlm_vx=result.target_vx,
            vlm_vy=result.target_vy, cam_goal=result.cam_goal,
            panel_info=panel_info,
        )
        if video_writer is None:
            fh, fw = frame.shape[:2]
            vpath = str(vis_dir / "nav_debug.mp4")
            video_writer = cv2.VideoWriter(
                vpath, cv2.VideoWriter_fourcc(*"mp4v"), 10, (fw, fh)
            )
        video_writer.write(frame)
        if step_num % 5 == 0:
            cv2.imwrite(str(vis_dir / f"frame_{step_num:04d}.jpg"), frame)

    print(f"[室内] 开始导航 {max_steps} 步...\n")

    step = 0
    idle_cap = max_steps * 200
    loop_guard = 0
    last_nav_state = {"x": 0, "y": 0}

    while step < max_steps and loop_guard < idle_cap:
        print(f"处理室内导航Step: {step}")
        loop_guard += 1

        obs = reader.read()
        result = engine.step(obs, step)

        if len(result.actions) == 1:
            client.act(result.actions[0])
        elif len(result.actions) > 1:
            client.act_many(result.actions)

        if result.step_counted:
            last_nav_state = {"x": engine.nav.x, "y": engine.nav.y}
            do_viz(step, result, obs)

        if result.done or step >= max_steps - 1:
            if step >= max_steps - 1:
                result.done = True
                result.done_reason = "max_steps_reached"
            indoor_result["done_reason"] = result.done_reason

            door_objs = [obj for obj in mapper.objects if obj.label == "door"]
            force_outdoor = indoor_cfg.get("force_outdoor", False)
            if door_objs:
                nearest_door = min(door_objs, key=lambda o: np.linalg.norm([o.center[0] - engine.nav.x, o.center[1] - engine.nav.y]))
                dist = np.linalg.norm([nearest_door.center[0] - engine.nav.x, nearest_door.center[1] - engine.nav.y])
                indoor_result["door_distance_m"] = dist
                indoor_result["door_position"] = nearest_door.center.tolist()
                print(f"[室内] 检测到门objects={len(door_objs)}, 最近门距离: {dist:.2f}m, 位置: {nearest_door.center[:2].tolist()}")
            elif force_outdoor:
                indoor_result["door_distance_m"] = 1.0
                indoor_result["door_position"] = [engine.nav.x, engine.nav.y, 0.0]
                print(f"[室内] 未检测到门，force_outdoor=True，使用当前位置作为室外起点")
            else:
                indoor_result["door_distance_m"] = None
                indoor_result["door_position"] = None
                print(f"[室内] 未检测到门，无法切换到室外导航")

            print(f"[室内] 导航结束: {result.done_reason}")
            break

        if result.step_counted:
            step += 1
        else:
            time.sleep(0.005)

    if loop_guard >= idle_cap:
        print(f"[室内] 已达 loop_guard 上限 ({idle_cap}), 结束")

    indoor_result["steps"] = step

    if video_writer:
        video_writer.release()
        print(f"[室内] 可视化视频: {vis_dir / 'nav_debug.mp4'}")

    engine.shutdown()
    client.close()

    return indoor_result, indoor_result.get("door_position")


def run_outdoor_nav(cfg: dict, start_pos: list, output_dir: Path) -> dict:
    """运行室外导航"""
    import subprocess
    
    os.environ.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)
    os.environ.pop("DISPLAY", None)
    time.sleep(2)
    
    outdoor_cfg = cfg["outdoor"]
    vlm_cfg = cfg["vlm"]
    coord_cfg = cfg.get("coordinate_transform", {})

    offset_x = coord_cfg.get("offset_x", -730.0)
    offset_y = coord_cfg.get("offset_y", 490.0)
    offset_z = coord_cfg.get("offset_z", 0.0)
    yaw_offset = coord_cfg.get("yaw_offset", 0.0)

    outdoor_start = [
        start_pos[0] + offset_x,
        start_pos[1] + offset_y,
        start_pos[2] + offset_z,
    ]

    print(f"\n{'='*60}")
    print(f"[室外导航] 起点: {outdoor_start}")
    print(f"{'='*60}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdoor_output_dir = Path("data") / "urbanverse" / "vlm_gps_nav" / timestamp
    outdoor_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "xvfb-run", "-a",
        os.path.join(ISAACSIM_ROOT, "python.sh"),
        "src/sim_vln_outdoor/scripts/vlm_gps_nav.py",
        "--trajectory", outdoor_cfg["trajectory"],
        "--usd-path", outdoor_cfg["usd_path"],
        "--headless",
        "--max-steps", str(outdoor_cfg.get("max_steps", 200)),
        "--controller-freq", str(outdoor_cfg.get("controller_freq", 1.0)),
        "--goal-tol", str(outdoor_cfg.get("goal_tolerance", 2.0)),
        "--gpu", str(cfg.get("output", {}).get("gpu", 2)),
        "--base-url", f"{vlm_cfg['api_url']}",
        "--model", vlm_cfg["model"],
        "--start-pos", str(outdoor_start[0]), str(outdoor_start[1]), str(outdoor_start[2]),
        "--start-yaw", str(yaw_offset),
        "--instruction", outdoor_cfg.get("instruction", "继续沿导航路线前往目的地。"),
    ]

    print(f"[室外] 执行命令: {' '.join(cmd)}")
    
    clean_env = os.environ.copy()
    clean_env["CUDA_VISIBLE_DEVICES"] = str(cfg.get("output", {}).get("gpu", 3))
    clean_env["ISAACSIM_CACHES"] = "/tmp/isaacsim_cache"
    
    os.makedirs("/tmp/isaacsim_cache", exist_ok=True)
    
    result = subprocess.run(cmd, cwd=_PROJECT_ROOT, env=clean_env)

    outdoor_result = {
        "scene": Path(outdoor_cfg["usd_path"]).parent.parent.name,
        "steps": 0,
        "success": False,
        "final_distance_m": None,
    }
    
    outdoor_visualization_dir = output_dir / "outdoor"
    outdoor_visualization_dir.mkdir(parents=True, exist_ok=True)

    summary_dirs = sorted(Path("data/urbanverse/vlm_gps_nav").glob("2*"), key=lambda p: p.name)
    if summary_dirs:
        latest_dir = summary_dirs[-1]
        summary_path = latest_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                summary = json.load(f)
            outdoor_result["steps"] = summary.get("total_steps", 0)
            outdoor_result["success"] = summary.get("success", False)
            outdoor_result["final_distance_m"] = summary.get("final_dist_to_goal_m")
            
            import shutil
            video_src = latest_dir / "nav.mp4"
            if video_src.exists():
                video_dst = outdoor_visualization_dir / "nav.mp4"
                shutil.copy(video_src, video_dst)
                print(f"[室外] 视频已复制到: {video_dst}")
            
            frames_src = latest_dir / "frames"
            if frames_src.exists():
                frames_dst = outdoor_visualization_dir / "frames"
                frames_dst.mkdir(parents=True, exist_ok=True)
                for frame in sorted(frames_src.glob("frame_*.png"))[:50]:
                    shutil.copy(frame, frames_dst / frame.name)
                print(f"[室外] 已复制 {len(list(frames_dst.glob('frame_*.png')))} 帧")

    outdoor_result["video"] = str(outdoor_visualization_dir / "nav.mp4")

    return outdoor_result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="室内外合并导航")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--sim-url", type=str, default=DEFAULT_SIM_URL, help="Habitat 仿真服务器 URL")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[配置] 加载: {args.config}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.get("output", {}).get("dir", "output/indoor_outdoor"))
    output_dir = Path(_PROJECT_ROOT) / output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出] 目录: {output_dir}")

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    indoor_result, door_pos = run_indoor_nav(cfg, args.sim_url, output_dir)
    cfg["_indoor_done"] = True

    door_pos = door_pos or indoor_result.get("door_position") or [0, 0, 0]
    indoor_result["door_position"] = door_pos

    indoor_success = indoor_result.get("door_distance_m") is not None and \
                  indoor_result["door_distance_m"] < cfg["indoor"].get("door_distance_threshold", 2.0)

    if indoor_success and door_pos[0] != 0:
        print(f"[导航] 室内成功(门距离={indoor_result['door_distance_m']:.2f}m)，切换到室外...")
        outdoor_result = run_outdoor_nav(cfg, door_pos, output_dir)
    else:
        print(f"[导航] 室内未成功，跳过室外导航")
        outdoor_result = {
            "scene": "skipped",
            "steps": 0,
            "success": False,
            "final_distance_m": None,
            "video": None,
        }

    total_result = {
        "timestamp": timestamp,
        "indoor": indoor_result,
        "outdoor": outdoor_result,
        "overall": {
            "total_steps": indoor_result["steps"] + outdoor_result["steps"],
            "success": indoor_success and outdoor_result.get("success", False),
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(total_result, f, indent=2)
    print(f"\n[完成] 汇总: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()