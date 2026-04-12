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
import time

from naviagent.perception import get_camera_intrinsics, YOLOESegmentor, SemanticMapper, SimClientObsReader
from naviagent.decision import DWAPlanner, TurnController, VLMNavigator, TaskOrchestrator, NavigationEngine
from naviagent.vlm.vlm_config import load_nav_vlm_config
from naviagent.common import draw_debug_frame, build_panel_info
from sim_vln_indoor.env import SimClient

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output", "nav")
DEFAULT_INSTRUCTION = "探索这个环境并找到通往室外的可能的出口，停在该出口前。"
DEFAULT_SIM_URL = "http://localhost:5100"

# 传感器配置 (仅用于 ObsReader 计算内参, 需与 sim_server.yaml 一致)
SENSOR_CONFIGS = {
    "front_depth": {"width": 640, "height": 480, "hfov": 120},
    "low_depth":   {"width": 640, "height": 480, "hfov": 90},
}


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

    front_cfg = SENSOR_CONFIGS["front_depth"]
    front_intrinsics = get_camera_intrinsics(
        front_cfg["width"], front_cfg["height"], front_cfg["hfov"]
    )

    # HTTP 观测读取器 + 导航引擎
    reader = SimClientObsReader(client, SENSOR_CONFIGS)
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
            print(f"[Main] 导航结束: {result.done_reason}")
            break

        if result.step_counted:
            step += 1
        else:
            time.sleep(0.005)

    if loop_guard >= idle_cap:
        print(f"[Main] 已达 loop_guard 上限 ({idle_cap}), 结束")

    if video_writer:
        video_writer.release()
        print(f"\n可视化视频: {os.path.join(vis_dir, 'nav_debug.mp4')}")

    engine.shutdown()
    client.close()
    print("导航完成!")


if __name__ == "__main__":
    main()
