"""
VLN 批量评测脚本 (HTTP 模式)
============================
通过 HTTP 连接仿真服务器, 逐 episode 运行导航, 计算 success / SPL / distance_to_goal。

启动前先在 habitat conda 环境中运行仿真服务:
  conda activate habitat
  python -m sim_vln_indoor.env.server

用法:
  python batch_eval.py --split val_seen --max-episodes 10 --steps 100
  python batch_eval.py --split val_seen --mock
"""

import os, sys
_VLN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SRC_ROOT = os.path.join(_VLN_ROOT, "Navi_Agent", "src")
sys.path.insert(0, _VLN_ROOT)
sys.path.insert(0, _SRC_ROOT)
os.chdir(_VLN_ROOT)

# 清除代理环境变量
for _k in ("all_proxy", "ALL_PROXY", "http_proxy", "HTTP_PROXY",
           "https_proxy", "HTTPS_PROXY", "socks_proxy", "SOCKS_PROXY"):
    os.environ.pop(_k, None)

import gzip
import json
import time
import argparse
import csv
import numpy as np
import cv2

from naviagent.perception import get_camera_intrinsics, YOLOESegmentor, SemanticMapper, SimClientObsReader
from naviagent.decision import DWAPlanner, TurnController, VLMNavigator, TaskOrchestrator, NavigationEngine
from naviagent.vlm.vlm_config import load_nav_vlm_config
from naviagent.common import draw_debug_frame, build_panel_info
from sim_vln_indoor.env import SimClient

DATASET_DIR = "/home/nuc/vln/data/vln_ce/R2R_VLNCE_v1-3"
SCENE_DIR = "/home/nuc/vln/data/scene_data/mp3d"
DEFAULT_SIM_URL = "http://localhost:5100"

SENSOR_CONFIGS = {
    "front_depth": {"width": 640, "height": 480, "hfov": 120},
    "low_depth":   {"width": 640, "height": 480, "hfov": 90},
}


def load_episodes(split="val_seen", episode_ids=None, max_episodes=None):
    path = os.path.join(DATASET_DIR, split, f"{split}.json.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据集文件不存在: {path}")
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    episodes = data["episodes"]
    print(f"[数据集] {split}: 共 {len(episodes)} 个 episode")
    if episode_ids is not None:
        id_set = set(episode_ids)
        episodes = [ep for ep in episodes if ep["episode_id"] in id_set]
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    return episodes


def compute_metrics(final_position, goal_position, path_length,
                    geodesic_distance, success_threshold=3.0):
    dist = np.linalg.norm(np.array(final_position) - np.array(goal_position))
    success = float(dist <= success_threshold)
    spl = success * (geodesic_distance / max(geodesic_distance, path_length)) if success and path_length > 0 else 0.0
    return {"distance_to_goal": dist, "success": success, "spl": spl, "path_length": path_length}


def run_episode(episode, client, vlm, dwa, mapper, turn_ctrl,
                max_steps=100, save_vis=False, vis_dir=None,
                orchestrator_kwargs=None):
    ep_id = episode["episode_id"]
    instruction = episode["instruction"]["instruction_text"]
    start_pos = episode["start_position"]
    start_rot = episode["start_rotation"]  # [x, y, z, w]
    goal_pos = episode["goals"][0]["position"]
    geodesic_dist = episode["info"]["geodesic_distance"]

    # 通过 HTTP 设置 agent 起始位姿
    client.set_agent_state(position=start_pos, rotation=start_rot)

    if vlm is not None:
        vlm.reset_history()
    if mapper is not None:
        mapper.objects.clear()
        mapper._next_id = 0

    orchestrator = None
    if orchestrator_kwargs is not None and vlm is not None and mapper is not None:
        orchestrator = TaskOrchestrator(
            full_instruction=instruction,
            on_subtask_change=lambda _sub: vlm.reset_history(),
            **orchestrator_kwargs,
        )

    print(f"\n  [EP {ep_id}] 指令: {instruction[:80]}...")
    print(f"  [EP {ep_id}] 测地距离: {geodesic_dist:.2f}m")

    front_cfg = SENSOR_CONFIGS["front_depth"]
    front_intrinsics = get_camera_intrinsics(
        front_cfg["width"], front_cfg["height"], front_cfg["hfov"]
    )

    reader = SimClientObsReader(client, SENSOR_CONFIGS)
    engine = NavigationEngine(
        vlm=vlm, dwa=dwa, turn_ctrl=turn_ctrl,
        front_intrinsics=front_intrinsics,
        instruction=instruction,
        orchestrator=orchestrator,
        mapper=mapper,
    )

    path_length = 0.0
    prev_pos = np.array(start_pos)
    video_writer = None

    def do_viz(step_num, result, obs):
        nonlocal video_writer
        if not (save_vis and vis_dir):
            return
        sem_map_vis = mapper.render_topdown(
            agent_x=engine.nav.x, agent_y=engine.nav.y,
            agent_yaw=engine.nav.yaw,
            map_size=480, scale=40, trajectory=engine.trajectory,
        ) if mapper else None
        panel_info = build_panel_info(
            step_num, instruction,
            orchestrator.get_viz_state() if orchestrator else None,
            vlm.get_viz_state() if vlm else None,
            result.vlm_view, result.target_vx, result.target_vy,
            result.action_type or "forward",
        )
        frame = draw_debug_frame(
            engine.nav, obs.views_bgr, sem_map_vis,
            result.dwa_debug, obs.obstacles_local, step_num,
            vlm_view=result.vlm_view, vlm_vx=result.target_vx,
            vlm_vy=result.target_vy, cam_goal=result.cam_goal,
            panel_info=panel_info,
        )
        if video_writer is None:
            ep_vis_dir = os.path.join(vis_dir, f"ep_{ep_id}")
            os.makedirs(ep_vis_dir, exist_ok=True)
            fh, fw = frame.shape[:2]
            video_writer = cv2.VideoWriter(
                os.path.join(ep_vis_dir, "nav.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), 10, (fw, fh),
            )
        video_writer.write(frame)

    try:
        step = 0
        idle_cap = max_steps * 200
        loop_guard = 0

        while step < max_steps and loop_guard < idle_cap:
            loop_guard += 1
            obs = reader.read()

            # 累计路径长度 (通过 HTTP 获取 agent 位置)
            agent_st = client.get_agent_state()
            cur_pos = np.array(agent_st.position)
            path_length += float(np.linalg.norm(cur_pos - prev_pos))
            prev_pos = cur_pos.copy()

            result = engine.step(obs, step)

            # 通过 HTTP 执行动作
            if len(result.actions) == 1:
                client.act(result.actions[0])
            elif len(result.actions) > 1:
                client.act_many(result.actions)

            if result.step_counted:
                do_viz(step, result, obs)

            if result.done:
                print(f"  [EP {ep_id}] Step {step}: {result.done_reason}")
                break

            if result.step_counted:
                step += 1
            else:
                time.sleep(0.005)
    finally:
        engine.shutdown()
        if video_writer:
            video_writer.release()

    final_st = client.get_agent_state()
    final_pos = list(final_st.position)
    metrics = compute_metrics(final_pos, goal_pos, path_length, geodesic_dist)
    metrics["episode_id"] = ep_id
    metrics["instruction"] = instruction

    status = "SUCCESS" if metrics["success"] else "FAIL"
    print(f"  [EP {ep_id}] {status} | dist={metrics['distance_to_goal']:.2f}m | "
          f"SPL={metrics['spl']:.3f} | path={path_length:.2f}m")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="VLN 批量评测 (HTTP 模式)")
    parser.add_argument("--sim-url", type=str, default=DEFAULT_SIM_URL)
    parser.add_argument("--split", type=str, default="val_seen",
                        choices=["train", "val_seen", "val_unseen", "test"])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--episode-ids", type=str, default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(_VLN_ROOT, "Navi_Agent", "output", "eval"))
    parser.add_argument("--success-threshold", type=float, default=3.0)
    parser.add_argument("--no-planner", action="store_true")
    parser.add_argument("--plan-heartbeat", type=int, default=15)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--plan-map-size", type=int, default=640)
    parser.add_argument("--plan-map-scale", type=int, default=25)
    parser.add_argument("--vlm-config", type=str, default=None,
                        help="VLM 配置 YAML 路径 (e.g. src/vlm_server/configs/nav_vlm.yaml)")
    args = parser.parse_args()

    vlm_cfg = load_nav_vlm_config(args.vlm_config)

    episode_ids = [int(x) for x in args.episode_ids.split(",")] if args.episode_ids else None
    episodes = load_episodes(args.split, episode_ids, args.max_episodes)
    if not episodes:
        print("没有找到匹配的 episode!")
        return

    from collections import defaultdict
    scene_groups = defaultdict(list)
    for ep in episodes:
        scene_groups[ep["scene_id"]].append(ep)
    print(f"\n共 {len(episodes)} 个 episode，涉及 {len(scene_groups)} 个场景\n")

    # 连接仿真服务器
    client = SimClient(args.sim_url)
    client.wait_ready(timeout=10)
    print(f"[Sim] 已连接: {args.sim_url}")

    dwa = DWAPlanner()
    turn_ctrl = TurnController()
    vlm = None if args.mock else VLMNavigator(config=vlm_cfg.system1)

    from datetime import datetime
    eval_dir = os.path.join(args.output_dir, f"{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(eval_dir, exist_ok=True)

    all_metrics = []
    total_t0 = time.time()

    for scene_id, scene_eps in scene_groups.items():
        # 提取场景名 (scene_id 格式: "mp3d/SCENE_NAME/SCENE_NAME.glb")
        scene_name = scene_id.replace("mp3d/", "").split("/")[0]

        print(f"\n{'='*60}")
        print(f"场景: {scene_id} ({len(scene_eps)} episodes)")
        print(f"{'='*60}")

        # 通过 HTTP 加载场景
        try:
            client.load_scene(scene_name=scene_name)
        except Exception as e:
            print(f"[SKIP] 加载场景失败: {e}")
            continue

        yoloe_classes = None
        try:
            yoloe = YOLOESegmentor(model_path="Navi_Agent/models/yoloe-11l-seg.pt")
            mapper = SemanticMapper(
                segmentor=yoloe, overlap_threshold=0.3,
                camera_height=1.5, camera_pitch_deg=-20.0,
            )
            yoloe_classes = list(yoloe.classes)
        except Exception as e:
            print(f"[WARN] YOLOE/Mapper 初始化失败: {e}")
            mapper = None

        orchestrator_kwargs = None
        if not args.no_planner and vlm is not None and mapper is not None:
            orchestrator_kwargs = dict(
                vlm_config=vlm_cfg.system2,
                heartbeat_steps=args.plan_heartbeat,
                map_size=args.plan_map_size,
                map_scale=args.plan_map_scale,
                mapped_classes=yoloe_classes,
            )

        for ep in scene_eps:
            try:
                metrics = run_episode(
                    ep, client, vlm, dwa, mapper, turn_ctrl,
                    max_steps=args.steps, save_vis=args.save_vis,
                    vis_dir=eval_dir, orchestrator_kwargs=orchestrator_kwargs,
                )
                all_metrics.append(metrics)
            except Exception as e:
                print(f"  [EP {ep['episode_id']}] ERROR: {e}")
                all_metrics.append({
                    "episode_id": ep["episode_id"],
                    "instruction": ep["instruction"]["instruction_text"],
                    "distance_to_goal": -1, "success": 0, "spl": 0,
                    "path_length": 0, "error": str(e),
                })

    client.close()

    # 汇总
    total_time = time.time() - total_t0
    n = len(all_metrics)
    valid = [m for m in all_metrics if "error" not in m]
    n_success = sum(1 for m in all_metrics if m.get("success", 0) > 0)

    if valid:
        avg_dist = np.mean([m["distance_to_goal"] for m in valid])
        avg_spl = np.mean([m["spl"] for m in valid])
        sr = n_success / len(valid) * 100
    else:
        avg_dist = avg_spl = sr = 0

    print(f"\n{'='*60}")
    print(f"评测结果 ({args.split}): SR={sr:.1f}% | Avg Dist={avg_dist:.2f}m | Avg SPL={avg_spl:.3f}")
    print(f"Episodes: {n} | Time: {total_time:.1f}s")

    csv_path = os.path.join(eval_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_id", "instruction", "distance_to_goal",
                                                "success", "spl", "path_length", "error"],
                                extrasaction="ignore")
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(m)

    summary_path = os.path.join(eval_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"split": args.split, "n_episodes": n, "success_rate": sr,
                    "avg_distance_to_goal": float(avg_dist), "avg_spl": float(avg_spl),
                    "total_time_s": total_time}, f, indent=2)

    print(f"结果: {csv_path}")


if __name__ == "__main__":
    main()
