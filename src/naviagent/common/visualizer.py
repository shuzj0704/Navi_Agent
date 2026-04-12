"""
导航调试可视化
==============
把每一步的相机视图、BEV(含 DWA 候选/最优轨迹)、语义地图,以及 System1/System2
两个子系统的关键状态拼成一张大图,供主循环写视频或周期性存帧。

布局:
    +-----+-----+-----+
    | LEFT|FRONT|RIGHT|   ← 顶部全景 (三视角)
    +-----+-----+-----+
    | sys1| bev | sem | sys2|   ← 底部
    +-----+-----+-----+-----+
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# ---------- 布局尺寸 ----------
VIS_VIEW_W, VIS_VIEW_H = 400, 300
VIS_BOTTOM_H = 400
VIS_PANEL_W = 300
VIS_BEV_W = 300
VIS_SEM_W = 300
VIS_TOTAL_W = VIS_VIEW_W * 3  # 1200
VIS_TOTAL_H = VIS_VIEW_H + VIS_BOTTOM_H  # 700


# ---------- 中文字体 ----------
_CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_CJK_FONT_CACHE = {}


def _load_cjk_font(size):
    if size not in _CJK_FONT_CACHE:
        try:
            _CJK_FONT_CACHE[size] = ImageFont.truetype(_CJK_FONT_PATH, size)
        except Exception:
            _CJK_FONT_CACHE[size] = ImageFont.load_default()
    return _CJK_FONT_CACHE[size]


def _wrap_chars(text, n_chars):
    out = []
    for raw in str(text).split("\n"):
        if not raw:
            out.append("")
            continue
        i = 0
        while i < len(raw):
            out.append(raw[i:i + n_chars])
            i += n_chars
    return out


def _draw_text_panel(width, height, sections, title="", bg=(20, 20, 28)):
    """sections: list[(label, body)]; 用 PIL 渲染中文。"""
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    title_font = _load_cjk_font(16)
    label_font = _load_cjk_font(13)
    body_font = _load_cjk_font(12)

    pad = 10
    line_h = 15
    y = pad
    chars_per_line = max(10, (width - 2 * pad) // 8)

    if title:
        draw.text((pad, y), title, font=title_font, fill=(120, 220, 255))
        y += 22
        draw.line([(pad, y - 4), (width - pad, y - 4)], fill=(60, 80, 100))

    for label, body in sections:
        if y + line_h > height - pad:
            break
        draw.text((pad, y), label, font=label_font, fill=(255, 200, 100))
        y += line_h + 1
        body_str = "" if body is None else str(body)
        for line in _wrap_chars(body_str, chars_per_line):
            if y + line_h > height - pad:
                draw.text((pad + 6, y), "…", font=body_font, fill=(180, 180, 180))
                y += line_h
                break
            draw.text((pad + 6, y), line, font=body_font, fill=(220, 220, 220))
            y += line_h
        y += 4

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _draw_bev(nav, dwa_debug, obstacles_local, size=VIS_BEV_W):
    """机器人坐标系 BEV: forward=向上, right=向右; 叠加 DWA 候选/最优轨迹。"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx = size // 2
    cy = int(size * 0.65)  # 机器人略偏下,前方留更多空间
    scale = 60.0  # 像素/米 → 视野约 ±3.3m

    def to_px(x_fwd, y_rgt):
        return int(cx + y_rgt * scale), int(cy - x_fwd * scale)

    # 网格 (1 米)
    for d in range(-5, 6):
        gx = int(cx + d * scale)
        gy = int(cy - d * scale)
        if 0 <= gx < size:
            cv2.line(img, (gx, 0), (gx, size), (35, 35, 45), 1)
        if 0 <= gy < size:
            cv2.line(img, (0, gy), (size, gy), (35, 35, 45), 1)

    # 1. 障碍物 (灰)
    if obstacles_local is not None and len(obstacles_local) > 0:
        ob = obstacles_local
        step_o = max(1, len(ob) // 800)
        for i in range(0, len(ob), step_o):
            px, py = to_px(ob[i, 0], ob[i, 1])
            if 0 <= px < size and 0 <= py < size:
                img[py, px] = (170, 170, 170)

    # 2. DWA 候选轨迹 (碰撞=暗红, 有效=暗绿)
    if dwa_debug is not None:
        trajs = dwa_debug["trajs"]
        coll = dwa_debug["collision_mask"]
        sample = max(1, len(trajs) // 60)
        for i in range(0, len(trajs), sample):
            color = (50, 50, 100) if coll[i] else (10, 90, 30)
            pts = []
            for t in range(0, trajs.shape[1], 3):
                pts.append(to_px(trajs[i, t, 0], trajs[i, t, 1]))
            for j in range(1, len(pts)):
                cv2.line(img, pts[j - 1], pts[j], color, 1)

        # 3. 最优轨迹 (亮绿粗线)
        bt = dwa_debug["best_traj"]
        bp = [to_px(bt[t, 0], bt[t, 1]) for t in range(bt.shape[0])]
        for j in range(1, len(bp)):
            cv2.line(img, bp[j - 1], bp[j], (80, 255, 120), 2)

        # 4. 目标 (红)
        gx_m, gy_m = dwa_debug["goal"][0], dwa_debug["goal"][1]
        gpx, gpy = to_px(gx_m, gy_m)
        if 0 <= gpx < size and 0 <= gpy < size:
            cv2.circle(img, (gpx, gpy), 6, (0, 0, 255), -1)
            cv2.circle(img, (gpx, gpy), 8, (255, 255, 255), 1)

    # 5. 机器人 (黄) + 朝向箭头
    cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)
    cv2.arrowedLine(img, (cx, cy), (cx, cy - 22), (0, 255, 255), 2, tipLength=0.4)

    # 6. 角标
    cv2.putText(img, "BEV  fwd↑", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(img, f"v={nav.cmd_v:+.2f} w={nav.cmd_omega:+.2f}",
                (8, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    return img


def _draw_view_with_overlay(view_bgr, view_name, vlm_view, vlm_vx, vlm_vy, cam_goal):
    """单视角缩略图 + 该视角上的 VLM 标记 + 标题条。"""
    img = cv2.resize(view_bgr, (VIS_VIEW_W, VIS_VIEW_H))

    # VLM 目标点 (仅落在该视角上时)
    if (vlm_view == view_name and vlm_vx is not None
            and vlm_vy is not None and view_name == "front"):
        sx = VIS_VIEW_W / 640.0
        sy = VIS_VIEW_H / 480.0
        px = int(vlm_vx * sx)
        py = int(vlm_vy * sy)
        cv2.circle(img, (px, py), 7, (0, 0, 255), -1)
        cv2.circle(img, (px, py), 9, (255, 255, 255), 2)
        if cam_goal is not None:
            txt = f"X={cam_goal[0]:.2f} Y={cam_goal[1]:.2f}"
            cv2.putText(img, txt, (px + 10, py + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    # 标题条
    label = {"left": "LEFT [l]", "front": "FRONT [f]",
             "right": "RIGHT [r]"}[view_name]
    is_chosen = (vlm_view == view_name)
    bar_color = (0, 200, 255) if is_chosen else (50, 50, 50)
    cv2.rectangle(img, (0, 0), (VIS_VIEW_W, 22), bar_color, -1)
    cv2.putText(img, label, (8, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2)
    cv2.rectangle(img, (0, 0), (VIS_VIEW_W - 1, VIS_VIEW_H - 1),
                  (90, 90, 90), 1)
    return img


def _build_sys1_panel(panel_info):
    pi = panel_info or {}
    history = pi.get("vlm_history") or []
    if history:
        hist_lines = "\n".join(
            f"step{h.get('step', '?')}: {h.get('view','?')[0]},"
            f"{h.get('vx','?')},{h.get('vy','?')}"
            for h in history[-5:]
        )
    else:
        hist_lines = "(空)"

    last_view = pi.get("vlm_view") or "-"
    last_vx = pi.get("vlm_vx")
    last_vy = pi.get("vlm_vy")
    last_str = (f"{last_view},{last_vx},{last_vy}"
                if last_vx is not None else last_view)

    sections = [
        ("原始任务", pi.get("full_instruction", "-")),
        ("当前指令(System1 实际输入)", pi.get("current_instruction", "-")),
        ("最近一次决策", last_str),
        ("决策历史(最近5)", hist_lines),
    ]
    return _draw_text_panel(VIS_PANEL_W, VIS_BOTTOM_H, sections,
                            title="System 1  反应式 VLM")


def _build_sys2_panel(panel_info):
    pi = panel_info or {}
    completed = pi.get("completed") or []
    if completed:
        done_lines = "\n".join(f"{i+1}. {s}" for i, s in enumerate(completed[-5:]))
    else:
        done_lines = "(尚无)"

    override = pi.get("stop_override")
    override_line = override if override else "(无)"

    sections = [
        ("原始任务", pi.get("full_instruction", "-")),
        ("当前子任务", pi.get("current_subtask", "-")),
        ("已完成子任务", done_lines),
        ("最近规划 reason", pi.get("last_reason") or "(无)"),
        ("STOP override", override_line),
    ]
    return _draw_text_panel(VIS_PANEL_W, VIS_BOTTOM_H, sections,
                            title="System 2  规划器")


def draw_debug_frame(nav, views_bgr, sem_map, dwa_debug, obstacles_local,
                     step, vlm_view=None, vlm_vx=None, vlm_vy=None,
                     cam_goal=None, panel_info=None):
    """拼出一张 1600x700 的调试帧。"""
    pano = np.hstack([
        _draw_view_with_overlay(views_bgr["left"],  "left",
                                vlm_view, vlm_vx, vlm_vy, cam_goal),
        _draw_view_with_overlay(views_bgr["front"], "front",
                                vlm_view, vlm_vx, vlm_vy, cam_goal),
        _draw_view_with_overlay(views_bgr["right"], "right",
                                vlm_view, vlm_vx, vlm_vy, cam_goal),
    ])

    sys1 = _build_sys1_panel(panel_info)
    bev = _draw_bev(nav, dwa_debug, obstacles_local, size=VIS_BEV_W)
    if bev.shape[0] != VIS_BOTTOM_H:
        bev = cv2.resize(bev, (VIS_BEV_W, VIS_BOTTOM_H))
    if sem_map is None:
        sem = np.zeros((VIS_BOTTOM_H, VIS_SEM_W, 3), dtype=np.uint8)
        cv2.putText(sem, "(no sem map)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    else:
        sem = cv2.resize(sem_map, (VIS_SEM_W, VIS_BOTTOM_H))
    sys2 = _build_sys2_panel(panel_info)

    bottom = np.hstack([sys1, bev, sem, sys2])
    frame = np.vstack([pano, bottom])

    cv2.putText(frame, f"step {step}", (VIS_TOTAL_W - 90, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def build_panel_info(step, instruction, orch_viz_state, vlm_viz_state,
                     vlm_view, target_vx, target_vy, action_type):
    """主循环用的便捷工具:从运行时状态打包成 draw_debug_frame 需要的 panel_info dict。

    instruction:      任务指令字符串
    orch_viz_state:   orchestrator.get_viz_state() 返回的 dict, 或 None
    vlm_viz_state:    vlm.get_viz_state() 返回的 dict, 或 None
    """
    orch = orch_viz_state or {}
    vlm_st = vlm_viz_state or {}
    return {
        "step": step,
        "full_instruction": instruction,
        "current_subtask": orch.get("current_subtask", instruction),
        "current_instruction": orch.get("current_instruction", instruction),
        "completed": orch.get("completed_subtasks", []),
        "last_reason": orch.get("last_reason", ""),
        "stop_override": orch.get("stop_override_text"),
        "vlm_history": vlm_st.get("history", []),
        "vlm_view": vlm_view,
        "vlm_vx": target_vx if action_type == "forward" else None,
        "vlm_vy": target_vy if action_type == "forward" else None,
    }
