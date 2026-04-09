#!/usr/bin/env python3
"""Interactive VLN Demo - 支持 Claude Agent 模式

NOTE: 参考脚本，来自外部项目。依赖的 6 个本地模块（sim_wrapper, topdown_map,
topo_memory, semantic_memory, path_planner, unified_agent）不在本仓库中，
当前无法直接运行。保留供日后开发参考。
"""
import os
for k in ["ALL_PROXY", "all_proxy", "HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "NO_PROXY", "no_proxy"]:
    os.environ.pop(k, None)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io
import base64
import json
import math
import numpy as np
from PIL import Image
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

import threading

from sim_wrapper import HabitatSimWrapper
from topdown_map import generate_topdown_map
from topo_memory import TopoGraph
from semantic_memory import SemanticMemory
from path_planner import PathPlanner
from unified_agent import UnifiedNavigationAgent

print("Initializing Habitat-Sim...", flush=True)
sim = HabitatSimWrapper("apt_0")
memory = TopoGraph()
action_log = ["[System] Scene apt_0 loaded."]
current_instruction = ""
decision_id = 0
step_id = 0
CONTEXT_DIR = "/tmp/vln_agent_context"
os.makedirs(CONTEXT_DIR, exist_ok=True)

# 语义导航系统
sem_memory = SemanticMemory()
planner = PathPlanner(sem_memory)
nav_state = {
    "running": False,
    "instruction": "",
    "status": "idle",
    "mode": "idle",       # topo / visual / hybrid / idle
    "progress": [],       # 实时进度日志
    "plan": None,
    "result": None,
    "current_wp": 0,
    "total_wp": 0,
}
nav_lock = threading.Lock()
print("Ready!", flush=True)


def img_to_base64(img_array):
    img = Image.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def save_images_to_disk(views, topdown):
    """保存4视角+俯视图到磁盘，供 Claude Read 工具读取"""
    for name, img in views.items():
        Image.fromarray(img).save(os.path.join(CONTEXT_DIR, "%s.png" % name))
    Image.fromarray(topdown).save(os.path.join(CONTEXT_DIR, "topdown.png"))


def get_compass():
    """获取 agent 朝向角度"""
    rot = sim.get_agent_rotation()
    qw, qy = float(rot.w), float(rot.y)
    yaw = math.degrees(2 * math.atan2(qy, qw))
    return round(yaw % 360, 1)


def build_context():
    """构建 agent context（参考 memory_system_design.md）"""
    global decision_id
    state = sim.get_agent_state()
    pos = sim.get_agent_position()
    compass = get_compass()

    # 更新记忆
    stuck = memory.is_stuck()
    loop = memory.detect_loop()

    ctx = {
        "decision_id": decision_id,
        "step_id": step_id,
        "instruction": current_instruction,
        "views": ["front", "right", "back", "left"],
        "compass": compass,
        "height": round(float(pos[1]) - (memory.nodes[0]["position"][1] if memory.nodes else float(pos[1])), 2),
        "on_stairs": abs(state["yaw"]) > 0.5 if memory.nodes and len(memory.position_history) > 1 else False,
        "stuck": stuck,
        "loop_detected": loop,
        "current_node": memory.current_node_id,
        "unexplored_directions": memory.get_unexplored_dirs(),
        "navigation_map": memory.to_text(),
        "position": state["position"],
    }
    return ctx


def get_state_json():
    views = sim.get_four_views()
    state = sim.get_agent_state()
    topdown = generate_topdown_map(
        sim.get_pathfinder(), sim.get_agent_position(), sim.get_agent_rotation()
    )
    return json.dumps({
        "front": img_to_base64(views["front"]),
        "back": img_to_base64(views["back"]),
        "left": img_to_base64(views["left"]),
        "right": img_to_base64(views["right"]),
        "topdown": img_to_base64(topdown),
        "state": state,
        "log": "\n".join(action_log[-20:]),
        "instruction": current_instruction,
        "memory": memory.to_text(),
    })


def get_full_state_json():
    """Agent 模式用：返回 context + 保存图片到磁盘"""
    views = sim.get_four_views()
    topdown = generate_topdown_map(
        sim.get_pathfinder(), sim.get_agent_position(), sim.get_agent_rotation()
    )
    save_images_to_disk(views, topdown)
    ctx = build_context()
    with open(os.path.join(CONTEXT_DIR, "context.json"), "w") as f:
        json.dump(ctx, f, indent=2)
    return json.dumps(ctx)


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>Interactive VLN Demo</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
h1 { text-align: center; color: #e94560; }
.subtitle { text-align: center; color: #aaa; font-size: 14px; margin-bottom: 15px; }
.container { display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; }
.views { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.views img { width: 320px; height: 240px; border: 2px solid #333; border-radius: 8px; }
.sidebar { display: flex; flex-direction: column; gap: 10px; max-width: 350px; }
.sidebar img { width: 300px; border: 2px solid #333; border-radius: 8px; }
.controls { text-align: center; margin: 15px 0; }
.controls button { padding: 10px 20px; margin: 4px; font-size: 15px; cursor: pointer;
    background: #16213e; color: #eee; border: 1px solid #e94560; border-radius: 8px; }
.controls button:hover { background: #e94560; }
.controls button.primary { background: #e94560; font-weight: bold; }
.instruction { text-align: center; margin: 10px 0; }
.instruction input { width: 450px; padding: 10px; font-size: 15px; border-radius: 8px;
    border: 1px solid #555; background: #16213e; color: #eee; }
.instruction button { padding: 10px 18px; font-size: 15px; background: #e94560;
    color: white; border: none; border-radius: 8px; cursor: pointer; margin-left: 8px; }
.state { background: #16213e; padding: 8px; border-radius: 8px; font-family: monospace; font-size: 12px; }
.log { background: #0f3460; padding: 8px; border-radius: 8px; font-family: monospace;
    font-size: 11px; max-height: 120px; overflow-y: auto; white-space: pre-wrap; margin-top: 8px; }
.memory { background: #1a0a30; padding: 8px; border-radius: 8px; font-family: monospace;
    font-size: 11px; max-height: 150px; overflow-y: auto; white-space: pre-wrap;
    border: 1px solid #6a3; margin-top: 8px; }
.label { font-size: 11px; color: #aaa; text-align: center; margin-top: 2px; }
.scene-select { text-align: center; margin: 8px 0; }
.scene-select select, .scene-select button { padding: 6px 12px; font-size: 13px;
    background: #16213e; color: #eee; border: 1px solid #555; border-radius: 5px; cursor: pointer; }
.agent-mode { background: #0a2a0a; border: 2px solid #0f0; border-radius: 10px;
    padding: 10px; margin: 10px 0; text-align: center; }
.agent-mode .status { color: #0f0; font-weight: bold; }
</style>
</head>
<body>
<h1>Interactive 3D Navigation Demo</h1>
<div class="subtitle">Claude Agent Mode: images saved to /tmp/vln_agent_context/</div>

<div class="scene-select">
    Scene: <select id="scene" onchange="changeScene(this.value)">
        <option value="apt_0">apt_0</option><option value="apt_1">apt_1</option>
        <option value="apt_2">apt_2</option><option value="apt_3">apt_3</option>
    </select>
    <button onclick="doAction('reset')">Reset</button>
</div>

<div class="instruction">
    <input id="cmd_input" placeholder="Navigation instruction (e.g., walk to the kitchen)"
        onkeydown="if(event.key==='Enter')setInstruction()" />
    <button onclick="setInstruction()">Set Instruction</button>
</div>

<div class="container">
    <div>
        <div class="views">
            <div><img id="img_front" /><div class="label">Front</div></div>
            <div><img id="img_back" /><div class="label">Back</div></div>
            <div><img id="img_left" /><div class="label">Left</div></div>
            <div><img id="img_right" /><div class="label">Right</div></div>
        </div>
    </div>
    <div class="sidebar">
        <img id="img_topdown" />
        <div class="label">Top-Down Map (Red=Agent)</div>
        <div class="state" id="state_info">Loading...</div>
        <div class="label">Navigation Memory</div>
        <div class="memory" id="memory_info">No memory yet</div>
    </div>
</div>

<div class="controls">
    <div><button onclick="doAction('look_up')">Look Up</button>
         <button class="primary" onclick="doAction('move_forward')">Forward</button>
         <button onclick="doAction('look_down')">Look Down</button></div>
    <div><button onclick="doAction('turn_left')">Turn Left</button>
         <button onclick="doAction('turn_right')">Turn Right</button></div>
</div>
<div class="log" id="log_output"></div>

<script>
function updateUI(data) {
    if (data.front) document.getElementById('img_front').src = 'data:image/jpeg;base64,' + data.front;
    if (data.back) document.getElementById('img_back').src = 'data:image/jpeg;base64,' + data.back;
    if (data.left) document.getElementById('img_left').src = 'data:image/jpeg;base64,' + data.left;
    if (data.right) document.getElementById('img_right').src = 'data:image/jpeg;base64,' + data.right;
    if (data.topdown) document.getElementById('img_topdown').src = 'data:image/jpeg;base64,' + data.topdown;
    if (data.state) document.getElementById('state_info').textContent =
        'Pos: ' + JSON.stringify(data.state.position) + '  Yaw: ' + data.state.yaw + '  Steps: ' + data.state.step;
    if (data.log) document.getElementById('log_output').textContent = data.log;
    if (data.memory) document.getElementById('memory_info').textContent = data.memory;
}
function doAction(a) { fetch('/api/action?a='+a).then(r=>r.json()).then(updateUI); }
function setInstruction() {
    var cmd = document.getElementById('cmd_input').value;
    fetch('/api/set_instruction?cmd='+encodeURIComponent(cmd)).then(r=>r.json()).then(updateUI);
}
function changeScene(s) { fetch('/api/scene?id='+s).then(r=>r.json()).then(updateUI); }
fetch('/api/state').then(r=>r.json()).then(updateUI);
setInterval(function(){fetch('/api/state').then(r=>r.json()).then(updateUI).catch(function(){});},1000);
</script>
</body>
</html>"""


import time as _time

# 导航速度控制 (秒/步)
NAV_STEP_DELAY = 0.15   # 每个动作后暂停，让前端看到变化
NAV_PHASE_DELAY = 1.5   # 阶段间暂停（分析/规划/导航切换时）


def run_auto_nav(instruction):
    """后台线程：统一导航 Agent（TOPO / VISUAL / HYBRID）"""
    global nav_state, step_id, decision_id, current_instruction
    with nav_lock:
        nav_state["running"] = True
        nav_state["instruction"] = instruction
        nav_state["status"] = "analyzing"
        nav_state["mode"] = "idle"
        nav_state["progress"] = []
        nav_state["plan"] = None
        nav_state["result"] = None
        nav_state["current_wp"] = 0
        nav_state["total_wp"] = 0

    current_instruction = instruction

    try:
        agent = UnifiedNavigationAgent(
            sim=sim,
            semantic_memory=sem_memory,
            path_planner=planner,
            step_delay=NAV_STEP_DELAY,
        )
        agent.execute(instruction, nav_state["progress"], nav_state)

    except Exception as e:
        import traceback
        nav_state["progress"].append(f"[错误] {str(e)}")
        nav_state["progress"].append(traceback.format_exc())
        nav_state["status"] = "error"
    finally:
        nav_state["running"] = False


NAV_HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Semantic Navigation Agent</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #0a0a1a; color: #e0e0e0; }

.header {
    background: linear-gradient(135deg, #1a0a2e, #0a1a3e);
    padding: 15px 30px; display: flex; align-items: center; justify-content: space-between;
    border-bottom: 2px solid #e94560;
}
.header h1 { font-size: 22px; color: #e94560; }
.header .room-badge {
    background: #16213e; padding: 6px 16px; border-radius: 20px;
    border: 1px solid #0f3460; font-size: 14px;
}
.header .room-badge .room-name { color: #4CAF50; font-weight: bold; }

.main { display: flex; height: calc(100vh - 60px); }

/* 左侧：视角 */
.left-panel { flex: 1; padding: 12px; display: flex; flex-direction: column; gap: 8px; }
.views-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; flex: 1; }
.view-box { position: relative; border-radius: 8px; overflow: hidden; border: 2px solid #1a2a4a; }
.view-box img { width: 100%; height: 100%; object-fit: cover; }
.view-box .view-label {
    position: absolute; top: 6px; left: 6px; background: rgba(0,0,0,0.7);
    padding: 2px 10px; border-radius: 4px; font-size: 12px; color: #aaa;
}

/* 中间：控制面板 */
.center-panel {
    width: 360px; padding: 12px; display: flex; flex-direction: column; gap: 10px;
    border-left: 1px solid #1a2a4a; border-right: 1px solid #1a2a4a;
}

.input-section { background: #111; border-radius: 10px; padding: 14px; }
.input-section label { font-size: 12px; color: #888; margin-bottom: 6px; display: block; }
.input-row { display: flex; gap: 6px; }
.input-row input {
    flex: 1; padding: 10px 14px; font-size: 14px; border-radius: 8px;
    border: 1px solid #333; background: #0a0a1a; color: #fff;
}
.input-row input:focus { border-color: #e94560; outline: none; }
.btn { padding: 10px 18px; border: none; border-radius: 8px; cursor: pointer;
    font-size: 14px; font-weight: 600; transition: all 0.2s; }
.btn-go { background: #e94560; color: white; }
.btn-go:hover { background: #ff6b81; }
.btn-go:disabled { background: #555; cursor: not-allowed; }
.btn-stop { background: #333; color: #e94560; border: 1px solid #e94560; }
.btn-stop:hover { background: #e94560; color: white; }
.btn-reset { background: #16213e; color: #aaa; border: 1px solid #333; }
.btn-reset:hover { background: #0f3460; color: #fff; }
.btn-row { display: flex; gap: 6px; margin-top: 8px; }

/* 预设指令 */
.presets { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
.preset-btn {
    padding: 5px 10px; font-size: 11px; border-radius: 15px; cursor: pointer;
    background: #1a1a3e; color: #aaa; border: 1px solid #333; transition: all 0.2s;
}
.preset-btn:hover { border-color: #e94560; color: #e94560; }

/* 状态指示 */
.status-bar {
    display: flex; align-items: center; gap: 8px; padding: 10px 14px;
    background: #111; border-radius: 10px;
}
.status-dot { width: 10px; height: 10px; border-radius: 50%; }
.status-dot.idle { background: #555; }
.status-dot.running { background: #4CAF50; animation: pulse 1s infinite; }
.status-dot.completed { background: #2196F3; }
.status-dot.failed { background: #f44336; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
.status-text { font-size: 13px; flex: 1; }
.status-steps { font-size: 12px; color: #888; }

/* 路径信息 */
.plan-section { background: #111; border-radius: 10px; padding: 12px; }
.plan-section h3 { font-size: 13px; color: #888; margin-bottom: 8px; }
.path-vis {
    display: flex; align-items: center; gap: 0; flex-wrap: wrap;
    padding: 8px; background: #0a0a1a; border-radius: 6px;
}
.path-node {
    padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: 600;
    white-space: nowrap;
}
.path-node.current { background: #e94560; color: white; }
.path-node.done { background: #1a3a1a; color: #4CAF50; }
.path-node.pending { background: #1a1a3e; color: #666; }
.path-arrow { color: #444; margin: 0 2px; font-size: 14px; }

/* 进度日志 */
.progress-log {
    flex: 1; background: #0a0a0a; border-radius: 10px; padding: 10px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 11px;
    overflow-y: auto; line-height: 1.6; border: 1px solid #1a1a2e;
}
.progress-log .line { padding: 1px 0; }
.progress-log .loc { color: #4CAF50; }
.progress-log .plan { color: #2196F3; }
.progress-log .nav { color: #FF9800; }
.progress-log .result { color: #e94560; font-weight: bold; }
.progress-log .err { color: #f44336; }
.progress-log .stop { color: #999; }

/* 右侧：俯视图 */
.right-panel {
    width: 300px; padding: 12px; display: flex; flex-direction: column; gap: 8px;
}
.topdown-box { border-radius: 8px; overflow: hidden; border: 2px solid #1a2a4a; }
.topdown-box img { width: 100%; }
.info-card { background: #111; border-radius: 10px; padding: 12px; font-size: 12px; }
.info-card h3 { color: #888; font-size: 12px; margin-bottom: 6px; }
.info-row { display: flex; justify-content: space-between; padding: 3px 0; }
.info-val { color: #4CAF50; font-weight: 600; }

/* 手动控制 */
.manual-ctrl { display: flex; gap: 4px; justify-content: center; flex-wrap: wrap; }
.manual-ctrl button {
    padding: 6px 12px; font-size: 12px; border-radius: 6px; cursor: pointer;
    background: #16213e; color: #aaa; border: 1px solid #333;
}
.manual-ctrl button:hover { border-color: #e94560; color: #eee; }
</style>
</head>
<body>

<div class="header">
    <h1>Semantic Navigation Agent</h1>
    <div class="room-badge">
        Room: <span class="room-name" id="hdr_room">--</span>
        &nbsp;|&nbsp; Pos: <span id="hdr_pos">--</span>
    </div>
</div>

<div class="main">
    <!-- Left: 4 views -->
    <div class="left-panel">
        <div class="views-grid">
            <div class="view-box"><img id="v_front"><div class="view-label">Front</div></div>
            <div class="view-box"><img id="v_right"><div class="view-label">Right</div></div>
            <div class="view-box"><img id="v_left"><div class="view-label">Left</div></div>
            <div class="view-box"><img id="v_back"><div class="view-label">Back</div></div>
        </div>
    </div>

    <!-- Center: Controls -->
    <div class="center-panel">
        <div class="input-section">
            <label>Navigation Instruction</label>
            <div class="input-row">
                <input id="nav_input" placeholder="例: 走到厨房 / go to the sofa"
                    onkeydown="if(event.key==='Enter')startNav()">
                <button class="btn btn-go" id="btn_go" onclick="startNav()">Go</button>
            </div>
            <div class="btn-row">
                <button class="btn btn-stop" onclick="stopNav()">Stop</button>
                <button class="btn btn-reset" onclick="resetScene()">Reset Scene</button>
            </div>
            <div style="margin-top:8px; display:flex; align-items:center; gap:8px;">
                <span style="font-size:11px; color:#888;">Speed:</span>
                <input type="range" id="speed_slider" min="20" max="300" value="150"
                    style="flex:1; accent-color:#e94560;"
                    oninput="setSpeed(this.value)">
                <span id="speed_label" style="font-size:11px; color:#888; width:40px;">150ms</span>
            </div>
            <div class="presets">
                <span class="preset-btn" onclick="setCmd('走到厨房')" title="TOPO">走到厨房</span>
                <span class="preset-btn" onclick="setCmd('去找沙发坐下')" title="TOPO">去找沙发</span>
                <span class="preset-btn" onclick="setCmd('先去厨房再去客厅')" title="TOPO">厨房→客厅</span>
                <span class="preset-btn" style="border-color:#FF9800;color:#FF9800" onclick="setCmd('直走到走廊尽头')" title="VISUAL">直走到尽头</span>
                <span class="preset-btn" style="border-color:#FF9800;color:#FF9800" onclick="setCmd('右转然后一直往前走到墙')" title="VISUAL">右转走到墙</span>
                <span class="preset-btn" style="border-color:#E040FB;color:#E040FB" onclick="setCmd('去厨房然后左转靠近冰箱')" title="HYBRID">厨房找冰箱</span>
            </div>
        </div>

        <div class="status-bar">
            <div class="status-dot" id="status_dot"></div>
            <div class="status-text" id="status_text">Idle — enter an instruction</div>
            <div class="status-steps" id="status_steps"></div>
        </div>

        <div class="plan-section" id="plan_section" style="display:none">
            <h3>Planned Path</h3>
            <div class="path-vis" id="path_vis"></div>
        </div>

        <div class="progress-log" id="progress_log">
            <div class="line stop">Ready. Type an instruction and press Go.</div>
        </div>

        <div class="manual-ctrl">
            <button onclick="manual('turn_left')">Turn L</button>
            <button onclick="manual('move_forward')">Forward</button>
            <button onclick="manual('turn_right')">Turn R</button>
            <button onclick="manual('look_up')">Up</button>
            <button onclick="manual('look_down')">Down</button>
        </div>
    </div>

    <!-- Right: Topdown + info -->
    <div class="right-panel">
        <div class="topdown-box"><img id="v_topdown"></div>
        <div class="info-card">
            <h3>Agent State</h3>
            <div class="info-row"><span>Position</span><span class="info-val" id="info_pos">--</span></div>
            <div class="info-row"><span>Yaw</span><span class="info-val" id="info_yaw">--</span></div>
            <div class="info-row"><span>Room</span><span class="info-val" id="info_room">--</span></div>
            <div class="info-row"><span>Steps</span><span class="info-val" id="info_steps">--</span></div>
        </div>
        <div class="info-card" id="result_card" style="display:none">
            <h3>Navigation Result</h3>
            <div class="info-row"><span>Status</span><span class="info-val" id="res_status">--</span></div>
            <div class="info-row"><span>Total Steps</span><span class="info-val" id="res_steps">--</span></div>
            <div class="info-row"><span>Final Room</span><span class="info-val" id="res_room">--</span></div>
            <div class="info-row"><span>Distance to Target</span><span class="info-val" id="res_dist">--</span></div>
        </div>
    </div>
</div>

<script>
const STATUS_MAP = {
    idle:              {cls: 'idle',      text: 'Idle'},
    analyzing:         {cls: 'running',   text: 'LLM Analyzing...'},
    parsing:           {cls: 'running',   text: 'Parsing targets...'},
    planning:          {cls: 'running',   text: 'A* path planning...'},
    navigating:        {cls: 'running',   text: 'Navigating...'},
    visual_reasoning:  {cls: 'running',   text: 'LLM Visual Reasoning...'},
    visual_acting:     {cls: 'running',   text: 'Visual Acting...'},
    completed:         {cls: 'completed', text: 'Navigation Complete!'},
    failed:            {cls: 'failed',    text: 'Navigation Failed'},
    stopped:           {cls: 'idle',      text: 'Stopped'},
    error:             {cls: 'failed',    text: 'Error'},
};
const MODE_COLORS = {topo:'#4CAF50', visual:'#FF9800', hybrid:'#E040FB', idle:'#666'};

function setCmd(cmd) { document.getElementById('nav_input').value = cmd; }
function setSpeed(v) {
    document.getElementById('speed_label').textContent = v + 'ms';
    fetch('/api/nav/speed?v=' + (v/1000));
}

function startNav() {
    var cmd = document.getElementById('nav_input').value.trim();
    if (!cmd) return;
    document.getElementById('btn_go').disabled = true;
    document.getElementById('plan_section').style.display = 'none';
    document.getElementById('result_card').style.display = 'none';
    document.getElementById('progress_log').innerHTML = '<div class="line stop">Starting...</div>';
    fetch('/api/nav/start?cmd=' + encodeURIComponent(cmd));
}

function stopNav() { fetch('/api/nav/stop'); }

function resetScene() {
    fetch('/api/action?a=reset').then(function(){ pollStatus(); });
}

function manual(action) {
    fetch('/api/action?a=' + action);
}

function colorLine(text) {
    var cls = 'stop';
    if (text.indexOf('━━━') >= 0 || text.indexOf('Phase') >= 0) cls = 'result';
    else if (text.indexOf('定位') >= 0 || text.indexOf('Agent') >= 0 || text.indexOf('匹配') >= 0
         || text.indexOf('房间内') >= 0) cls = 'loc';
    else if (text.indexOf('指令') >= 0 || text.indexOf('解析') >= 0 || text.indexOf('目标') >= 0
         || text.indexOf('扫描') >= 0 || text.indexOf('关键词') >= 0) cls = 'plan';
    else if (text.indexOf('规划') >= 0 || text.indexOf('A*') >= 0 || text.indexOf('路段') >= 0
         || text.indexOf('航点列表') >= 0 || text.indexOf('WP') >= 0 || text.indexOf('搜索') >= 0
         || text.indexOf('起点') >= 0 || text.indexOf('终点') >= 0 || text.indexOf('总航') >= 0
         || text.indexOf('总距') >= 0) cls = 'plan';
    else if (text.indexOf('观察') >= 0 || text.indexOf('推理') >= 0 || text.indexOf('LLM') >= 0
         || text.indexOf('决策') >= 0 || text.indexOf('子目标') >= 0 || text.indexOf('视觉') >= 0
         || text.indexOf('VISUAL') >= 0 || text.indexOf('HYBRID') >= 0 || text.indexOf('模式') >= 0
         || text.indexOf('切换') >= 0) cls = 'plan';
    else if (text.indexOf('导航') >= 0 || text.indexOf('前进') >= 0 || text.indexOf('左转') >= 0
         || text.indexOf('右转') >= 0 || text.indexOf('卡住') >= 0 || text.indexOf('到达航点') >= 0
         || text.indexOf('绕行') >= 0 || text.indexOf('朝') >= 0 || text.indexOf('转向') >= 0
         || text.indexOf('继续') >= 0 || text.indexOf('搜索') >= 0) cls = 'nav';
    else if (text.indexOf('验证') >= 0 || text.indexOf('★') >= 0 || text.indexOf('成功') >= 0
         || text.indexOf('失败') >= 0 || text.indexOf('判定') >= 0) cls = 'result';
    else if (text.indexOf('错误') >= 0 || text.indexOf('✗') >= 0) cls = 'err';
    else if (text.indexOf('中止') >= 0) cls = 'stop';
    var escaped = text.replace(/</g, '&lt;');
    // highlight checkmarks and stars
    escaped = escaped.replace(/✓/g, '<span style="color:#4CAF50">✓</span>');
    escaped = escaped.replace(/✗/g, '<span style="color:#f44336">✗</span>');
    escaped = escaped.replace(/★/g, '<span style="color:#FFD700">★</span>');
    return '<div class="line ' + cls + '">' + escaped + '</div>';
}

function renderPath(plan, currentWp) {
    if (!plan || !plan.segments) return;
    var box = document.getElementById('plan_section');
    box.style.display = 'block';
    var vis = document.getElementById('path_vis');
    var html = '';
    // collect all room names in order
    var rooms = [];
    plan.segments.forEach(function(seg) {
        seg.rooms.forEach(function(r) {
            if (rooms.length === 0 || rooms[rooms.length-1] !== r) rooms.push(r);
        });
    });
    // determine which waypoint index each room corresponds to
    rooms.forEach(function(r, i) {
        var cls = 'pending';
        if (i < currentWp) cls = 'done';
        else if (i === currentWp) cls = 'current';
        html += '<span class="path-node ' + cls + '">' + r + '</span>';
        if (i < rooms.length - 1) html += '<span class="path-arrow">→</span>';
    });
    vis.innerHTML = html;
}

function pollStatus() {
    fetch('/api/nav/status').then(function(r){return r.json();}).then(function(d) {
        // images
        if (d.front) document.getElementById('v_front').src = 'data:image/jpeg;base64,' + d.front;
        if (d.right) document.getElementById('v_right').src = 'data:image/jpeg;base64,' + d.right;
        if (d.left)  document.getElementById('v_left').src  = 'data:image/jpeg;base64,' + d.left;
        if (d.back)  document.getElementById('v_back').src  = 'data:image/jpeg;base64,' + d.back;
        if (d.topdown) document.getElementById('v_topdown').src = 'data:image/jpeg;base64,' + d.topdown;

        // header
        document.getElementById('hdr_room').textContent = d.room || '--';
        var p = d.state ? d.state.position : [0,0,0];
        document.getElementById('hdr_pos').textContent =
            '(' + p[0].toFixed(2) + ', ' + p[2].toFixed(2) + ')';

        // info card
        document.getElementById('info_pos').textContent =
            '(' + p[0].toFixed(2) + ', ' + p[2].toFixed(2) + ')';
        document.getElementById('info_yaw').textContent = (d.state ? d.state.yaw.toFixed(1) : '--') + '°';
        document.getElementById('info_room').textContent = d.room + (d.room_cn ? ' ' + d.room_cn : '');
        document.getElementById('info_steps').textContent = d.state ? d.state.step : '--';

        // nav status
        var nav = d.nav || {};
        var si = STATUS_MAP[nav.status] || STATUS_MAP.idle;
        document.getElementById('status_dot').className = 'status-dot ' + si.cls;
        var stxt = si.text;
        if (nav.status === 'navigating' && nav.total_wp > 0) {
            stxt += ' (WP ' + nav.current_wp + '/' + nav.total_wp + ')';
        }
        var mode = nav.mode || 'idle';
        var modeColor = MODE_COLORS[mode] || '#666';
        stxt = (mode !== 'idle' ? '[' + mode.toUpperCase() + '] ' : '') + stxt;
        document.getElementById('status_text').textContent = stxt;
        document.getElementById('status_text').style.color = modeColor;
        document.getElementById('btn_go').disabled = nav.running;

        // plan
        if (nav.plan) renderPath(nav.plan, nav.current_wp);

        // progress log
        if (nav.progress && nav.progress.length > 0) {
            var logEl = document.getElementById('progress_log');
            logEl.innerHTML = nav.progress.map(colorLine).join('');
            logEl.scrollTop = logEl.scrollHeight;
        }

        // result card
        if (nav.result) {
            var rc = document.getElementById('result_card');
            rc.style.display = 'block';
            document.getElementById('res_status').textContent = nav.result.success ? 'SUCCESS' : 'FAILED';
            document.getElementById('res_status').style.color = nav.result.success ? '#4CAF50' : '#f44336';
            document.getElementById('res_steps').textContent = nav.result.total_steps || '--';
            document.getElementById('res_room').textContent = nav.result.final_room || '--';
            document.getElementById('res_dist').textContent = (nav.result.final_dist || '--') + 'm';
        }

    }).catch(function(){});
}

// poll every 300ms
setInterval(pollStatus, 300);
pollStatus();

// keyboard shortcuts
document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT') return;
    switch(e.key) {
        case 'w': case 'ArrowUp':    manual('move_forward'); break;
        case 'a': case 'ArrowLeft':  manual('turn_left'); break;
        case 'd': case 'ArrowRight': manual('turn_right'); break;
    }
});
</script>
</body>
</html>"""


class DemoHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        global decision_id, step_id, current_instruction, memory
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._respond(200, "text/html", HTML_PAGE)

        elif path == "/api/state":
            self._respond(200, "application/json", get_state_json())

        elif path == "/api/action":
            action = params.get("a", [""])[0]
            if action == "reset":
                sim.reset()
                memory = TopoGraph()
                decision_id = 0
                step_id = 0
                action_log.append("[System] Reset")
            else:
                sim.act(action)
                step_id += 1
                action_log.append("[Manual] %s" % action)
            self._respond(200, "application/json", get_state_json())

        elif path == "/api/set_instruction":
            current_instruction = params.get("cmd", [""])[0]
            memory = TopoGraph()
            decision_id = 0
            step_id = 0
            action_log.append("[Instruction] %s" % current_instruction)
            self._respond(200, "application/json", get_state_json())

        elif path == "/api/scene":
            scene_id = params.get("id", ["apt_0"])[0]
            sim.change_scene(scene_id)
            memory = TopoGraph()
            decision_id = 0
            step_id = 0
            current_instruction = ""
            action_log.append("[System] Loaded %s" % scene_id)
            self._respond(200, "application/json", get_state_json())

        # ========== Agent Mode API ==========
        elif path == "/api/agent/context":
            # Claude 调用：获取当前 context + 保存4张图到 /tmp
            pos = sim.get_agent_position()
            compass = get_compass()
            memory.update(pos, compass)
            ctx_json = get_full_state_json()
            self._respond(200, "application/json", ctx_json)

        elif path == "/api/agent/act":
            # Claude 调用：执行动作序列
            actions_str = params.get("actions", [""])[0]  # 逗号分隔: "1,1,3,1"
            scene_desc = params.get("scene", [""])[0]
            direction = params.get("dir", [""])[0]

            # 更新记忆中的场景描述
            if scene_desc and 0 <= memory.current_node_id < len(memory.nodes):
                memory.nodes[memory.current_node_id]["scene_description"] = scene_desc

            action_map = {"0": "stop", "1": "move_forward", "2": "turn_left", "3": "turn_right",
                          "4": "look_up", "5": "look_down"}
            actions = actions_str.split(",") if actions_str else []
            executed = 0
            stopped = False
            for a in actions:
                a = a.strip()
                if a == "0":
                    stopped = True
                    break
                act_name = action_map.get(a, "")
                if act_name:
                    sim.act(act_name)
                    step_id += 1
                    executed += 1

            # 碰撞检测
            if memory.is_stuck() and direction:
                memory.mark_blocked(direction)

            decision_id += 1
            action_log.append("[Agent D%d] %d actions, dir=%s %s" % (
                decision_id - 1, executed, direction, "(STOP)" if stopped else ""))

            # 返回新的 context
            pos = sim.get_agent_position()
            compass = get_compass()
            memory.update(pos, compass, direction_taken=direction)
            ctx_json = get_full_state_json()

            result = json.loads(ctx_json)
            result["executed"] = executed
            result["stopped"] = stopped
            self._respond(200, "application/json", json.dumps(result))

        elif path == "/api/agent/pixel":
            # Claude 调用：像素导航（点击前方图像某个位置前进）
            row = int(params.get("row", ["240"])[0])
            col = int(params.get("col", ["320"])[0])
            direction = params.get("dir", ["front"])[0]
            scene_desc = params.get("scene", [""])[0]

            # 像素导航：先转向目标方向，再前进
            if direction == "right":
                sim.act("turn_right"); sim.act("turn_right"); sim.act("turn_right")  # 90度
                step_id += 3
            elif direction == "left":
                sim.act("turn_left"); sim.act("turn_left"); sim.act("turn_left")
                step_id += 3
            elif direction == "back":
                for _ in range(6): sim.act("turn_left")
                step_id += 6

            # 根据像素位置微调方向
            center_col = 320
            if col < center_col - 80:
                sim.act("turn_left"); step_id += 1
            elif col > center_col + 80:
                sim.act("turn_right"); step_id += 1

            # 前进
            for _ in range(3):
                sim.act("move_forward"); step_id += 1

            if scene_desc and 0 <= memory.current_node_id < len(memory.nodes):
                memory.nodes[memory.current_node_id]["scene_description"] = scene_desc

            decision_id += 1
            action_log.append("[Agent D%d] pixel(%d,%d) dir=%s" % (decision_id - 1, row, col, direction))

            pos = sim.get_agent_position()
            compass = get_compass()
            memory.update(pos, compass, direction_taken=direction)
            self._respond(200, "application/json", get_full_state_json())

        # ========== Auto Navigation API ==========
        elif path == "/nav":
            self._respond(200, "text/html", NAV_HTML_PAGE)

        elif path == "/api/nav/start":
            instruction = params.get("cmd", [""])[0]
            if nav_state["running"]:
                self._respond(200, "application/json",
                    json.dumps({"error": "Navigation already running"}))
                return
            # 启动后台导航线程
            t = threading.Thread(target=run_auto_nav, args=(instruction,), daemon=True)
            t.start()
            self._respond(200, "application/json",
                json.dumps({"status": "started", "instruction": instruction}))

        elif path == "/api/nav/stop":
            nav_state["running"] = False
            nav_state["status"] = "stopped"
            nav_state["progress"].append("[中止] 用户停止导航")
            self._respond(200, "application/json", json.dumps({"status": "stopped"}))

        elif path == "/api/nav/speed":
            global NAV_STEP_DELAY
            speed = params.get("v", ["0.15"])[0]
            NAV_STEP_DELAY = max(0.02, min(0.5, float(speed)))
            self._respond(200, "application/json",
                json.dumps({"step_delay": NAV_STEP_DELAY}))

        elif path == "/api/nav/ctrl":
            # Claude 终端直接控制 nav_state（LLM 大脑接口）
            act = params.get("action", [""])[0]
            if act == "start":
                nav_state["running"] = True
                nav_state["instruction"] = params.get("cmd", [""])[0]
                nav_state["status"] = "analyzing"
                nav_state["mode"] = params.get("mode", ["visual"])[0]
                nav_state["progress"] = []
                nav_state["plan"] = None
                nav_state["result"] = None
                nav_state["current_wp"] = 0
                nav_state["total_wp"] = 0
            elif act == "log":
                msg = params.get("msg", [""])[0]
                nav_state["progress"].append(msg)
            elif act == "status":
                nav_state["status"] = params.get("s", ["idle"])[0]
                if "mode" in params:
                    nav_state["mode"] = params["mode"][0]
            elif act == "finish":
                nav_state["running"] = False
                nav_state["status"] = "completed" if params.get("ok", ["1"])[0] == "1" else "failed"
                nav_state["result"] = {
                    "success": params.get("ok", ["1"])[0] == "1",
                    "total_steps": int(params.get("steps", ["0"])[0]),
                    "final_room": params.get("room", ["?"])[0],
                    "final_dist": float(params.get("dist", ["0"])[0]),
                }
            self._respond(200, "application/json", json.dumps({"ok": True}))

        elif path == "/api/nav/status":
            # 返回导航状态 + 当前视图
            views = sim.get_four_views()
            st = sim.get_agent_state()
            topdown = generate_topdown_map(
                sim.get_pathfinder(), sim.get_agent_position(), sim.get_agent_rotation()
            )
            room = sem_memory.localize_room(st["position"][0], st["position"][2])
            self._respond(200, "application/json", json.dumps({
                "front": img_to_base64(views["front"]),
                "right": img_to_base64(views["right"]),
                "back": img_to_base64(views["back"]),
                "left": img_to_base64(views["left"]),
                "topdown": img_to_base64(topdown),
                "state": st,
                "room": room["name"] if room else "Unknown",
                "room_cn": room["label_cn"] if room else "",
                "nav": {
                    "running": nav_state["running"],
                    "status": nav_state["status"],
                    "mode": nav_state.get("mode", "idle"),
                    "instruction": nav_state["instruction"],
                    "progress": nav_state["progress"][-50:],
                    "current_wp": nav_state["current_wp"],
                    "total_wp": nav_state["total_wp"],
                    "plan": nav_state["plan"],
                    "result": nav_state["result"],
                },
            }))

        else:
            self.send_response(404)
            self.end_headers()

    def _respond(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode() if isinstance(body, str) else body)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    port = 7860
    print("Starting server on http://127.0.0.1:%d" % port, flush=True)
    print("Agent context images: %s" % CONTEXT_DIR, flush=True)
    print("\nAgent API:", flush=True)
    print("  GET /api/agent/context  - 获取4视角图+context", flush=True)
    print("  GET /api/agent/act?actions=1,1,3,1&dir=front&scene=hallway", flush=True)
    print("  GET /api/agent/pixel?row=200&col=300&dir=front&scene=kitchen", flush=True)
    server = HTTPServer(("127.0.0.1", port), DemoHandler)
    server.serve_forever()
