"""
导航主循环
==========
通过 HTTP 连接仿真服务器, NavigationEngine 驱动导航。

数据流:
  SimClient (HTTP) → SimClientObsReader → ObsBundle → NavigationEngine → StepResult → SimClient.act()

启动前先在 habitat conda 环境中运行仿真服务:
  conda activate habitat
  python -m sim_vln_indoor.env.server
"""

import os, sys
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
sys.path.insert(0, _SRC_ROOT)
os.chdir(_PROJECT_ROOT)

import numpy as np
import cv2
import json
import time

from naviagent.perception import get_camera_intrinsics, YOLOESegmentor, SemanticMapper, SimClientObsReader
from naviagent.decision import DWAPlanner, TurnController, VLMNavigator, TaskOrchestrator, NavigationEngine
from naviagent.vlm.vlm_config import load_nav_vlm_config
from naviagent.common import draw_debug_frame, build_panel_info
from sim_vln_indoor.env import SimClient

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output", "nav")
DEFAULT_INSTRUCTION = "探索这个环境并找到通往室外的可能的出口，穿过该出口，找到一个关闭的防火门，停在这个门前。"
DEFAULT_SIM_URL = "http://localhost:5100"

# 传感器配置由仿真服务器提供 (GET /sensors), 客户端不再硬编码。


def _to_jsonable(obj):
    """ndarray/quaternion 等转成原生 Python 类型以便 json.dumps。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def save_history(vis_dir, args, scene_name, start_pose, end_pose,
                 done_reason, steps_done, engine, vlm, orchestrator, mapper):
    """
    将本次 run 的全部运行时状态写到 <vis_dir>/history/, 用于事后复现 / 调试。

    文件清单:
      history/run_meta.json        — args, scene, instruction, 起止位姿, done_reason, 时序统计
      history/trajectory.json      — [(x,y), ...] 全轨迹
      history/vlm_history.json     — System1 VLM 最近 N 次决策 (view/vx/vy/pose/step)
      history/orchestrator.json    — System2 状态 (subtask/completed/plan_count/last_reason)
      history/semantic_objects.json— 最终语义地图 (Object3D 列表)
      history/vlm_front_memory/    — VLM 前视记忆快照 (JPEG) + index.json 元信息
    """
    hist_dir = os.path.join(vis_dir, "history")
    os.makedirs(hist_dir, exist_ok=True)

    # --- run_meta.json ---
    meta = {
        "scene": scene_name,
        "instruction": args.instruction,
        "args": {k: _to_jsonable(v) for k, v in vars(args).items()},
        "start_pose": _to_jsonable(start_pose),
        "end_pose": {
            "nav_x": float(engine.nav.x),
            "nav_y": float(engine.nav.y),
            "yaw_rad": float(engine.nav.yaw),
        } if end_pose is None else _to_jsonable(end_pose),
        "done_reason": done_reason,
        "steps_done": int(steps_done),
        "vlm_call_count": len(engine.vlm_times),
        "vlm_latency_sec": {
            "mean": float(np.mean(engine.vlm_times)) if engine.vlm_times else 0.0,
            "p50": float(np.percentile(engine.vlm_times, 50)) if engine.vlm_times else 0.0,
            "p95": float(np.percentile(engine.vlm_times, 95)) if engine.vlm_times else 0.0,
            "max": float(np.max(engine.vlm_times)) if engine.vlm_times else 0.0,
        },
    }
    with open(os.path.join(hist_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # --- trajectory.json ---
    with open(os.path.join(hist_dir, "trajectory.json"), "w") as f:
        json.dump([[float(x), float(y)] for x, y in engine.trajectory], f, indent=2)

    # --- vlm_history.json + front_memory ---
    if vlm is not None:
        vlm_hist = [_to_jsonable(h) for h in vlm.history]
        with open(os.path.join(hist_dir, "vlm_history.json"), "w") as f:
            json.dump(vlm_hist, f, ensure_ascii=False, indent=2)

        mem_dir = os.path.join(hist_dir, "vlm_front_memory")
        os.makedirs(mem_dir, exist_ok=True)
        mem_index = []
        for i, entry in enumerate(vlm.front_memory):
            step_tag = entry.get("step", i)
            jpg_name = f"frame_{i:03d}_step{step_tag}.jpg"
            cv2.imwrite(os.path.join(mem_dir, jpg_name), entry["bgr"])
            mem_index.append({
                "idx": i,
                "image": jpg_name,
                "step": step_tag,
                "x": float(entry.get("x", 0.0)),
                "y": float(entry.get("y", 0.0)),
                "yaw_rad": float(entry.get("yaw", 0.0)),
                "action": _to_jsonable(entry.get("action")),
            })
        with open(os.path.join(mem_dir, "index.json"), "w") as f:
            json.dump(mem_index, f, ensure_ascii=False, indent=2)

    # --- orchestrator.json ---
    if orchestrator is not None:
        orch_state = {
            "full_instruction": orchestrator.full_instruction,
            "current_subtask": orchestrator.current_subtask,
            "completed_subtasks": list(orchestrator.completed_subtasks),
            "plan_count": int(orchestrator._plan_count),
            "last_reason": orchestrator._last_reason,
            "is_done": bool(orchestrator.is_done),
            "stop_override_count": int(orchestrator._stop_override_count),
        }
        with open(os.path.join(hist_dir, "orchestrator.json"), "w") as f:
            json.dump(orch_state, f, ensure_ascii=False, indent=2)

    # --- semantic_objects.json ---
    if mapper is not None:
        objs = []
        for o in mapper.objects:
            objs.append({
                "id": int(o.id),
                "label": o.label,
                "center": _to_jsonable(o.center),
                "size": _to_jsonable(o.size),
                "confidence": float(o.confidence),
                "color_bgr": list(o.color),
            })
        with open(os.path.join(hist_dir, "semantic_objects.json"), "w") as f:
            json.dump(objs, f, ensure_ascii=False, indent=2)

    print(f"[History] 保存运行状态 → {hist_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VLN 导航主循环 (HTTP 模式)")
    parser.add_argument("--sim-url", type=str, default=DEFAULT_SIM_URL, help="仿真服务器 URL")
    parser.add_argument("--scene", type=str, default=None, help="场景名 (e.g. 17DRP5sb8fy)")
    parser.add_argument("--scene-idx", type=int, default=0, help="场景编号 (--scene 优先)")
    parser.add_argument("--steps", type=int, default=100, help="步数")
    parser.add_argument("--save-vis", action="store_true", help="保存可视化帧")
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION, help="导航任务指令")
    parser.add_argument("--mock", action="store_true", help="使用 mock VLM (不调用模型)")
    parser.add_argument("--diagnose", action="store_true", help="每步额外调用 VLM 输出详细分析")
    parser.add_argument("--no-planner", action="store_true",
                        help="禁用 System 2 任务规划器")
    parser.add_argument("--plan-heartbeat", type=int, default=15)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--plan-map-size", type=int, default=640)
    parser.add_argument("--plan-map-scale", type=int, default=25)
    parser.add_argument("--vlm-config", type=str, default=None,
                        help="VLM 配置 YAML 路径 (e.g. src/vlm_server/configs/nav_vlm.yaml)")
    parser.add_argument("--random-start", action="store_true",
                        help="场景加载后随机选取 navmesh 上一个可行点 + 随机 yaw 作为起点 (用于测试)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机起点的 seed, 设置后结果可复现 (仅在 --random-start 下生效)")
    args = parser.parse_args()

    vlm_cfg = load_nav_vlm_config(args.vlm_config)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 连接仿真服务器
    client = SimClient(args.sim_url)
    print(f"[Sim] 连接仿真服务器: {args.sim_url}")
    client.wait_ready(timeout=10)

    # 加载场景
    if args.scene:
        scene_name = args.scene
    else:
        scenes = client.list_scenes()
        scene_name = scenes[args.scene_idx]["name"]

    print(f"[Sim] 加载场景: {scene_name}")
    client.load_scene(scene_name=scene_name)

    if args.random_start:
        start = client.random_agent_state(seed=args.seed)
        seed_str = f" (seed={args.seed})" if args.seed is not None else ""
        print(f"[Sim] 随机起点{seed_str}: pos={start.position}")
    else:
        start = client.get_agent_state()

    start_pose_log = {
        "position": _to_jsonable(start.position),
        "rotation": _to_jsonable(start.rotation),
        "random": bool(args.random_start),
        "seed": args.seed,
    }

    # 初始化组件
    dwa = DWAPlanner()
    yoloe = YOLOESegmentor(model_path="models/yoloe-11l-seg.pt")
    print(f"YOLOE loaded. Detecting: {yoloe.classes}")
    mapper = SemanticMapper(
        segmentor=yoloe,
        overlap_threshold=0.3,
        camera_height=1.5,
        camera_pitch_deg=-20.0,
    )
    turn_ctrl = TurnController()

    if args.mock:
        vlm = None
        print("使用 Mock VLM (目标: 画面中心)")
    else:
        vlm = VLMNavigator(config=vlm_cfg.system1)
        print(f"System1 VLM 已连接: {vlm.api_url} model={vlm.model}")
        print(f"任务指令: {args.instruction}")

    orchestrator = None
    if vlm is not None and not args.no_planner:
        orchestrator = TaskOrchestrator(
            full_instruction=args.instruction,
            vlm_config=vlm_cfg.system2,
            heartbeat_steps=args.plan_heartbeat,
            map_size=args.plan_map_size,
            map_scale=args.plan_map_scale,
            mapped_classes=list(yoloe.classes),
            on_subtask_change=lambda _sub: vlm.reset_history(),
        )
        print(f"[Orchestrator] 双系统已启用 (System2: {vlm_cfg.system2.api_url} model={vlm_cfg.system2.model})")
    else:
        print("[Orchestrator] 未启用")

    sensor_configs = client.get_sensors()
    front_cfg = sensor_configs["front_depth"]
    front_intrinsics = get_camera_intrinsics(
        front_cfg["width"], front_cfg["height"], front_cfg["hfov"]
    )

    # HTTP 观测读取器 + 导航引擎
    reader = SimClientObsReader(client, sensor_configs)
    engine = NavigationEngine(
        vlm=vlm, dwa=dwa, turn_ctrl=turn_ctrl,
        front_intrinsics=front_intrinsics,
        instruction=args.instruction,
        orchestrator=orchestrator,
        mapper=mapper,
    )

    # 可视化
    from datetime import datetime
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = os.path.join(OUTPUT_DIR, scene_name, run_stamp)
    os.makedirs(vis_dir, exist_ok=True)
    video_writer = None

    def do_viz(step_num, result, obs):
        nonlocal video_writer
        sem_map = mapper.render_topdown(
            agent_x=engine.nav.x, agent_y=engine.nav.y,
            agent_yaw=engine.nav.yaw,
            map_size=480, scale=40, trajectory=engine.trajectory,
        )
        panel_info = build_panel_info(
            step_num, args.instruction,
            orchestrator.get_viz_state() if orchestrator is not None else None,
            vlm.get_viz_state() if vlm is not None else None,
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
            vpath = os.path.join(vis_dir, "nav_debug.mp4")
            video_writer = cv2.VideoWriter(
                vpath, cv2.VideoWriter_fourcc(*"mp4v"), 10, (fw, fh)
            )
        video_writer.write(frame)
        if step_num % 5 == 0:
            cv2.imwrite(os.path.join(vis_dir, f"frame_{step_num:04d}.jpg"), frame)

    print(f"开始导航 {args.steps} 步...\n")

    step = 0
    idle_cap = args.steps * 200
    loop_guard = 0
    done_reason = "max_steps"

    while step < args.steps and loop_guard < idle_cap:
        loop_guard += 1

        obs = reader.read()
        result = engine.step(obs, step)

        # 通过 HTTP 执行动作
        if len(result.actions) == 1:
            client.act(result.actions[0])
        elif len(result.actions) > 1:
            client.act_many(result.actions)

        if args.diagnose and vlm is not None and result.step_counted:
            diag_dir = os.path.join(vis_dir, "diagnose")
            raw_analysis, _ = vlm.diagnose(
                obs.views_bgr, args.instruction, step, diag_dir
            )
            print(f"--- 诊断分析 ---\n{raw_analysis}\n----------------")

        if result.step_counted:
            do_viz(step, result, obs)

        if result.done:
            done_reason = result.done_reason or "engine_done"
            print(f"[Main] 导航结束: {done_reason}")
            break

        if result.step_counted:
            step += 1
        else:
            time.sleep(0.005)

    if loop_guard >= idle_cap:
        done_reason = "loop_guard"
        print(f"[Main] 已达 loop_guard 上限 ({idle_cap}), 结束")

    if video_writer:
        video_writer.release()
        print(f"\n可视化视频: {os.path.join(vis_dir, 'nav_debug.mp4')}")

    try:
        save_history(
            vis_dir=vis_dir, args=args, scene_name=scene_name,
            start_pose=start_pose_log, end_pose=None,
            done_reason=done_reason, steps_done=step,
            engine=engine, vlm=vlm, orchestrator=orchestrator, mapper=mapper,
        )
    except Exception as e:
        print(f"[History] 保存失败: {e}")

    engine.shutdown()
    client.close()
    print("导航完成!")


if __name__ == "__main__":
    main()
