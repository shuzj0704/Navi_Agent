#!/usr/bin/env python3
"""Local HTTP server for browsing vlm_gps_nav replays.

Starts a zero-dep HTTP server that lists every run directory under --base-dir
and serves a single-page app where you can pick a run and scrub through its
trajectory, FPV frames, and VLM I/O.

Usage:
    python scripts/utils/serve_replay.py
    # -> prints URL, opens browser automatically

    # custom base dir / port
    python scripts/utils/serve_replay.py \
        --base-dir data/urbanverse/vlm_gps_nav --port 8765

    # expose on all interfaces (for remote access)
    python scripts/utils/serve_replay.py --host 0.0.0.0
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


# ── data loading ────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_next_turn(prompt: str) -> str:
    for line in prompt.splitlines():
        if "NEXT TURN" in line:
            return line.split("NEXT TURN:", 1)[-1].strip()
    return ""


def extract_lookahead(prompt: str) -> list:
    out, in_block = [], False
    for line in prompt.splitlines():
        if "WAYPOINTS" in line and "NEXT" in line:
            in_block = True
            continue
        if in_block:
            stripped = line.strip()
            if not stripped:
                break
            if stripped[:2] in ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."):
                out.append(stripped)
            else:
                break
    return out


def extract_field(prompt: str, key: str) -> str:
    """Pull a `  key: value` line out of the structured prompt."""
    needle = f"{key}:"
    for line in prompt.splitlines():
        s = line.strip()
        if s.startswith(needle):
            return s[len(needle):].strip()
    return ""


def extract_instruction(prompt: str) -> str:
    for line in prompt.splitlines():
        if "Task:" in line:
            return line.split("Task:", 1)[-1].strip()
    return ""


def parse_reply(reply: str) -> tuple:
    """Split a two-line 'REASON: ... / ACTION: ...' reply into (reason, action).

    Falls back gracefully on older single-keyword replies.
    """
    if not reply:
        return ("", "")
    reason, action = "", ""
    for raw in reply.splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("REASON:"):
            reason = line.split(":", 1)[-1].strip()
        elif upper.startswith("ACTION:"):
            action = line.split(":", 1)[-1].strip()
    if not action:
        # legacy: whole reply is just the keyword
        for kw in ("FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"):
            if kw in reply.upper():
                action = kw
                break
    return (reason, action)


def resolve_trajectory(project_root: Path, summary: dict) -> Path:
    tp = summary.get("trajectory_path", "")
    if tp:
        cand = project_root / tp
        if cand.exists():
            return cand
    return project_root / "data/urbanverse/trajectory/scene_09/dense_trajectory.json"


def list_runs(base_dir: Path) -> list:
    """Return list of run summaries sorted newest first."""
    runs = []
    for child in sorted(base_dir.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        summary_path = child / "summary.json"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            continue
        runs.append({
            "name": child.name,
            "success": bool(summary.get("success", False)),
            "total_steps": int(summary.get("total_steps", 0)),
            "final_dist_to_goal_m": float(summary.get("final_dist_to_goal_m", 0)),
            "route_length_m": float(summary.get("route_length_m", 0)),
        })
    return runs


def build_run_payload(run_dir: Path, project_root: Path) -> dict:
    with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    traj = load_jsonl(run_dir / "trajectory.jsonl")
    vlm = load_jsonl(run_dir / "vlm_io.jsonl")
    if len(traj) != len(vlm):
        raise ValueError(f"length mismatch: trajectory={len(traj)} vlm={len(vlm)}")

    traj_path = resolve_trajectory(project_root, summary)
    if not traj_path.exists():
        raise FileNotFoundError(f"dense trajectory not found: {traj_path}")
    with open(traj_path, "r", encoding="utf-8") as f:
        dense = json.load(f)

    steps = []
    for i, (t, v) in enumerate(zip(traj, vlm)):
        pose = t["pose"]
        action = t.get("action", {})
        prompt = v.get("prompt", "")
        steps.append({
            "step": i,
            "x": pose["x"],
            "y": pose["y"],
            "yaw": pose["yaw"],
            "forward": action.get("forward", 0.0),
            "yaw_act": action.get("yaw", 0.0),
            "done": action.get("done", False),
            "d2g": t.get("dist_to_goal_m", 0.0),
            "progress_idx": v.get("progress_idx", -1),
            "reply": v.get("reply", ""),
            "reason": parse_reply(v.get("reply", ""))[0],
            "action_kw": parse_reply(v.get("reply", ""))[1],
            "instruction": extract_instruction(prompt),
            "vlm_position": extract_field(prompt, "position"),
            "vlm_heading": extract_field(prompt, "heading"),
            "vlm_goal": extract_field(prompt, "goal"),
            "vlm_remaining": extract_field(prompt, "remaining"),
            "vlm_progress": extract_field(prompt, "progress"),
            "next_turn": extract_next_turn(prompt),
            "lookahead": extract_lookahead(prompt),
            "frame": f"/frames/{run_dir.name}/frame_{i:06d}.png",
        })

    return {
        "name": run_dir.name,
        "summary": summary,
        "steps": steps,
        "path": [[p[0], p[1]] for p in dense["path"]],
        "waypoints": [[w["pos"][0], w["pos"][1]] for w in dense["waypoints"]],
    }


# ── HTML SPA ────────────────────────────────────────────────────────────────
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NaviAgent · replay viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0d1117;
  --panel: #161b22;
  --panel-2: #1c2129;
  --border: #30363d;
  --text: #e6edf3;
  --text-dim: #8b949e;
  --text-mute: #6e7681;
  --accent: #58a6ff;
  --success: #3fb950;
  --danger: #f85149;
  --warn: #d29922;
  --shadow: 0 1px 3px rgba(0,0,0,0.4);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  font-feature-settings: "cv11", "ss01", "ss03";
  font-size: 13px;
  letter-spacing: 0.01em;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

header {
  padding: 12px 20px;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
  box-shadow: var(--shadow);
}
header .brand {
  font-family: "Inter", system-ui, sans-serif;
  font-size: 15px;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: 0.2px;
}
header .brand span {
  color: var(--text-dim);
  font-weight: 400;
  font-style: italic;
}

.run-picker {
  display: flex;
  align-items: center;
  gap: 8px;
}
.run-picker label {
  font-size: 11px;
  color: var(--text-mute);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.run-picker select {
  background: var(--panel-2);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 7px 12px;
  border-radius: 6px;
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 12px;
  cursor: pointer;
  min-width: 260px;
}
.run-picker select:focus {
  outline: none;
  border-color: var(--accent);
}
.run-picker button.refresh {
  background: var(--panel-2);
  border: 1px solid var(--border);
  color: var(--text-dim);
  padding: 7px 10px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
}
.run-picker button.refresh:hover { color: var(--accent); border-color: var(--accent); }

.badge {
  padding: 4px 11px;
  border-radius: 16px;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  border: 1px solid;
}
.badge.success { background: rgba(63,185,80,0.12); color: var(--success); border-color: rgba(63,185,80,0.4); }
.badge.fail { background: rgba(248,81,73,0.12); color: var(--danger); border-color: rgba(248,81,73,0.4); }
.badge.pending { background: rgba(139,148,158,0.12); color: var(--text-dim); border-color: rgba(139,148,158,0.3); }

.stats {
  display: flex;
  gap: 20px;
  font-size: 12px;
  color: var(--text-dim);
  font-family: "JetBrains Mono", ui-monospace, monospace;
}
.stats span b { color: var(--text); margin-left: 4px; font-weight: 600; }

main {
  flex: 1;
  display: grid;
  grid-template-columns: minmax(0, 1.8fr) minmax(0, 1fr);
  gap: 14px;
  padding: 14px;
  min-height: 0;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  box-shadow: var(--shadow);
}

.panel-fpv { position: relative; }
.panel-fpv .img-wrap {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #000;
  min-height: 0;
  overflow: hidden;
}
.panel-fpv img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}
.fpv-overlay {
  position: absolute;
  top: 14px;
  left: 14px;
  background: rgba(13,17,23,0.85);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  padding: 9px 15px;
  border-radius: 8px;
  border: 1px solid var(--border);
  font-family: "JetBrains Mono", ui-monospace, "SF Mono", monospace;
  font-size: 12px;
  display: flex;
  gap: 16px;
  align-items: center;
}
.fpv-overlay .lbl {
  color: var(--text-mute);
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  margin-right: 5px;
  font-family: "Inter", sans-serif;
  font-weight: 500;
}
.fpv-overlay .val { color: var(--text); font-weight: 600; }
.fpv-overlay .reply-val { color: var(--accent); font-weight: 700; font-size: 12.5px; letter-spacing: 0.5px; }

.fpv-tag {
  position: absolute;
  top: 14px;
  right: 14px;
  background: rgba(88,166,255,0.15);
  border: 1px solid rgba(88,166,255,0.5);
  color: var(--accent);
  padding: 5px 11px;
  border-radius: 6px;
  font-family: "Inter", sans-serif;
  font-size: 10.5px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.8px;
}
.map-tag {
  position: absolute;
  top: 8px;
  right: 12px;
  background: rgba(13,17,23,0.85);
  border: 1px solid var(--border);
  color: var(--text-mute);
  padding: 4px 10px;
  border-radius: 5px;
  font-family: "Inter", sans-serif;
  font-size: 10.5px;
  font-weight: 500;
  z-index: 10;
  pointer-events: none;
}

.empty {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-mute);
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 13px;
}

.right-col {
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-height: 0;
}
.panel-map { flex: 0 0 33%; min-height: 180px; }
#map { width: 100%; height: 100%; }

.panel-info {
  flex: 1 1 auto;
  padding: 0;
  font-family: "Inter", system-ui, sans-serif;
  font-size: 12px;
  overflow-y: auto;
  min-height: 0;
}
.mono { font-family: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace; }
.section { border-bottom: 1px solid var(--border); padding: 11px 18px; }
.section:last-child { border-bottom: none; }
.section-header {
  display: flex;
  align-items: center;
  gap: 9px;
  margin-bottom: 8px;
}
.section-header .stripe {
  width: 3px;
  height: 14px;
  border-radius: 2px;
}
.section-header .ttl {
  font-size: 11.5px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.9px;
  font-family: "Inter", system-ui, sans-serif;
}
.section-header .sub {
  font-size: 11px;
  color: var(--text-mute);
  margin-left: auto;
  font-weight: 400;
  font-style: italic;
}
.sec-input  .stripe { background: var(--accent); }
.sec-input  .ttl    { color: var(--accent); }
.sec-output .stripe { background: var(--warn); }
.sec-output .ttl    { color: var(--warn); }
.sec-action .stripe { background: var(--success); }
.sec-action .ttl    { color: var(--success); }

.row { display: flex; align-items: baseline; gap: 12px; line-height: 1.5; padding: 1px 0; }
.row .label {
  color: var(--text-mute);
  min-width: 82px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  font-weight: 500;
}
.row .value {
  color: var(--text);
  font-size: 12px;
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-weight: 500;
}
.row .value.text { font-family: "Inter", sans-serif; font-weight: 400; }
.lookahead-block {
  margin-top: 3px;
  padding-left: 94px;
  color: var(--text-dim);
  font-size: 11px;
  line-height: 1.5;
  font-family: "JetBrains Mono", ui-monospace, monospace;
}
.next-turn-warn { color: var(--warn); font-weight: 700; }
.next-turn-now { color: var(--danger); font-weight: 700; }

.reply-box {
  background: rgba(210,153,34,0.09);
  border: 1px solid rgba(210,153,34,0.38);
  border-radius: 7px;
  padding: 10px 14px;
  margin-top: 2px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.reply-line { display: flex; align-items: baseline; gap: 12px; }
.reply-line .lbl {
  color: var(--text-mute);
  font-size: 11px;
  min-width: 60px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  font-weight: 600;
}
.reply-line.reason-line .val {
  color: var(--text);
  font-size: 12px;
  line-height: 1.55;
  font-family: "Inter", sans-serif;
  font-weight: 400;
  font-style: italic;
}
.reply-line.action-line .val {
  color: var(--warn);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 1px;
  font-family: "JetBrains Mono", ui-monospace, monospace;
}
.reply-raw {
  margin-top: 4px;
  font-size: 9.5px;
  color: var(--text-mute);
  white-space: pre-wrap;
  word-break: break-word;
  display: none;
}
.reply-raw.show { display: block; }
.action-block {
  background: rgba(63,185,80,0.08);
  border: 1px solid rgba(63,185,80,0.35);
  border-radius: 7px;
  padding: 9px 14px;
  margin-top: 2px;
}
.action-block .motion {
  color: var(--success);
  font-size: 12.5px;
  font-weight: 600;
  margin-bottom: 4px;
  font-family: "Inter", sans-serif;
}
.action-block .raw {
  color: var(--text-dim);
  font-size: 11px;
  font-family: "JetBrains Mono", ui-monospace, monospace;
}
.note {
  font-size: 11px;
  color: var(--text-mute);
  font-style: italic;
  margin-top: 6px;
  line-height: 1.5;
  font-family: "Inter", sans-serif;
}

footer {
  padding: 11px 20px;
  background: var(--panel);
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 12px;
  box-shadow: 0 -1px 3px rgba(0,0,0,0.4);
}
footer button {
  background: var(--panel-2);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 7px 13px;
  border-radius: 6px;
  cursor: pointer;
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 13px;
  transition: all 0.15s ease;
  min-width: 38px;
}
footer button:hover { background: var(--border); color: var(--accent); }
footer button:disabled { opacity: 0.4; cursor: not-allowed; }
footer button:disabled:hover { color: var(--text); background: var(--panel-2); }
footer button.play { min-width: 46px; font-size: 14px; }

.slider-wrap { flex: 1; display: flex; align-items: center; height: 22px; }
footer input[type=range] {
  width: 100%;
  height: 6px;
  -webkit-appearance: none;
  appearance: none;
  background: var(--panel-2);
  border-radius: 3px;
  outline: none;
  border: 1px solid var(--border);
  cursor: pointer;
}
footer input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 18px; height: 18px;
  background: var(--accent);
  border-radius: 50%;
  cursor: pointer;
  border: 3px solid var(--bg);
  box-shadow: 0 0 0 1px var(--accent);
}
footer input[type=range]::-moz-range-thumb {
  width: 18px; height: 18px;
  background: var(--accent);
  border-radius: 50%;
  cursor: pointer;
  border: 3px solid var(--bg);
}
.step-count {
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 13px;
  color: var(--text-dim);
  min-width: 84px;
  text-align: right;
}
.step-count b { color: var(--text); font-weight: 600; }
.speed-select {
  background: var(--panel-2);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 6px 8px;
  border-radius: 6px;
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 12px;
  cursor: pointer;
}

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--panel); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-mute); }

.toast {
  position: fixed;
  top: 80px;
  right: 20px;
  background: var(--panel);
  border: 1px solid var(--danger);
  color: var(--danger);
  padding: 12px 16px;
  border-radius: 8px;
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 12px;
  max-width: 400px;
  box-shadow: var(--shadow);
  opacity: 0;
  transform: translateX(20px);
  transition: all 0.3s ease;
  pointer-events: none;
}
.toast.show { opacity: 1; transform: translateX(0); }

@media (max-width: 1100px) {
  main { grid-template-columns: 1fr; }
  .panel-fpv { min-height: 360px; }
}
</style>
</head>
<body>

<header>
  <div class="brand">NaviAgent <span>· replay</span></div>
  <div class="run-picker">
    <label>run</label>
    <select id="run-select"><option value="">loading...</option></select>
    <button class="refresh" id="btn-refresh" title="reload run list">↻</button>
  </div>
  <span class="badge pending" id="status-badge">—</span>
  <div class="stats">
    <span>steps<b id="stat-steps">—</b></span>
    <span>final d2g<b id="stat-d2g">—</b></span>
    <span>route<b id="stat-route">—</b></span>
    <span>tol<b id="stat-tol">—</b></span>
  </div>
</header>

<main>
  <div class="panel panel-fpv">
    <div class="img-wrap">
      <img id="fpv" src="" alt="" style="display:none">
      <div class="empty" id="fpv-empty">select a run</div>
    </div>
    <div class="fpv-overlay" id="fpv-overlay" style="display:none">
      <span><span class="lbl">step</span> <span class="val" id="ov-step">0</span></span>
      <span><span class="lbl">d2g</span> <span class="val" id="ov-d2g">-</span></span>
      <span><span class="lbl">reply</span> <span class="reply-val" id="ov-reply">-</span></span>
    </div>
    <div class="fpv-tag" id="fpv-tag" style="display:none">VLM input · FPV image</div>
  </div>

  <div class="right-col">
    <div class="panel panel-map" style="position:relative;">
      <div class="map-tag">debug top-down · NOT seen by VLM</div>
      <div id="map"></div>
    </div>
    <div class="panel panel-info" id="info">
      <div class="empty" style="padding:24px;">select a run to begin</div>
    </div>
  </div>
</main>

<footer>
  <button id="btn-first" title="first (Home)">⏮</button>
  <button id="btn-prev" title="prev (←)">◀</button>
  <button id="btn-play" class="play" title="play/pause (Space)">▶</button>
  <button id="btn-next" title="next (→)">▶</button>
  <button id="btn-last" title="last (End)">⏭</button>
  <select class="speed-select" id="speed">
    <option value="500">0.5×</option>
    <option value="200" selected>1×</option>
    <option value="100">2×</option>
    <option value="50">4×</option>
    <option value="25">8×</option>
  </select>
  <div class="slider-wrap"><input type="range" id="slider" min="0" max="0" value="0"></div>
  <div class="step-count"><b id="cur-step">—</b> / <span id="max-step">—</span></div>
</footer>

<div class="toast" id="toast"></div>

<script>
let RUN = null;       // current run payload
let cur = 0;
let playing = false;
let playTimer = null;
let playSpeed = 200;

// ── toast ──
function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 3500);
}

// ── initial: list runs ──
async function loadRunList() {
  const sel = document.getElementById('run-select');
  sel.innerHTML = '<option value="">loading...</option>';
  try {
    const res = await fetch('/api/runs');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    sel.innerHTML = '';
    if (!data.runs || !data.runs.length) {
      sel.innerHTML = '<option value="">(no runs found)</option>';
      return;
    }
    for (const r of data.runs) {
      const opt = document.createElement('option');
      opt.value = r.name;
      const status = r.success ? '✓' : '✗';
      opt.textContent = `${status} ${r.name}  ·  ${r.total_steps} steps  ·  d2g ${r.final_dist_to_goal_m.toFixed(1)}m`;
      sel.appendChild(opt);
    }
    // auto-load first
    if (data.runs.length) {
      sel.value = data.runs[0].name;
      await loadRun(data.runs[0].name);
    }
  } catch (e) {
    toast('failed to list runs: ' + e.message);
    sel.innerHTML = '<option value="">(error)</option>';
  }
}

async function loadRun(name) {
  if (!name) return;
  try {
    const res = await fetch('/api/run/' + encodeURIComponent(name));
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    RUN = data;
    cur = 0;
    stopPlay();
    onRunLoaded();
  } catch (e) {
    toast('failed to load run: ' + e.message);
  }
}

function onRunLoaded() {
  const sm = RUN.summary;
  // header stats
  const badge = document.getElementById('status-badge');
  badge.className = 'badge ' + (sm.success ? 'success' : 'fail');
  badge.textContent = sm.success ? 'success' : 'failed';
  document.getElementById('stat-steps').textContent = sm.total_steps;
  document.getElementById('stat-d2g').textContent = (sm.final_dist_to_goal_m || 0).toFixed(2) + ' m';
  document.getElementById('stat-route').textContent = (sm.route_length_m || 0).toFixed(2) + ' m';
  document.getElementById('stat-tol').textContent = (sm.goal_tol_m || 0).toFixed(1) + ' m';

  // show FPV area
  document.getElementById('fpv').style.display = '';
  document.getElementById('fpv-empty').style.display = 'none';
  document.getElementById('fpv-overlay').style.display = '';
  document.getElementById('fpv-tag').style.display = '';

  // slider
  const slider = document.getElementById('slider');
  slider.max = RUN.steps.length - 1;
  slider.value = 0;
  document.getElementById('max-step').textContent = RUN.steps.length - 1;

  // map
  drawMap();

  // render first step
  render(0);
}

function drawMap() {
  const planned_x = RUN.path.map(p => p[0]);
  const planned_y = RUN.path.map(p => p[1]);
  const wp_x = RUN.waypoints.map(p => p[0]);
  const wp_y = RUN.waypoints.map(p => p[1]);
  const actual_x = RUN.steps.map(s => s.x);
  const actual_y = RUN.steps.map(s => s.y);

  const traces = [
    {
      x: planned_x, y: planned_y, mode: 'lines', name: 'planned',
      line: { color: '#6e7681', width: 4, dash: 'dot' },
      hoverinfo: 'skip',
    },
    {
      x: wp_x, y: wp_y, mode: 'markers', name: 'waypoints',
      marker: { color: '#8b949e', size: 12, symbol: 'square', line: { color: '#0d1117', width: 1 } },
      hoverinfo: 'skip',
    },
    {
      x: actual_x, y: actual_y, mode: 'lines', name: 'actual',
      line: { color: '#58a6ff', width: 2.5 },
      customdata: RUN.steps.map((s, i) => i),
      hovertemplate: 'step %{customdata}<br>(%{x:.1f}, %{y:.1f})<extra></extra>',
    },
    {
      x: [planned_x[0]], y: [planned_y[0]], mode: 'markers', name: 'start',
      marker: { color: '#3fb950', size: 16, line: { color: '#0d1117', width: 2 } },
      hoverinfo: 'name',
    },
    {
      x: [planned_x[planned_x.length - 1]], y: [planned_y[planned_y.length - 1]], mode: 'markers', name: 'goal',
      marker: { color: '#f85149', size: 22, symbol: 'star', line: { color: '#0d1117', width: 2 } },
      hoverinfo: 'name',
    },
    {
      x: [RUN.steps[0].x], y: [RUN.steps[0].y], mode: 'markers', name: 'current',
      marker: { color: '#d29922', size: 18, symbol: 'circle', line: { color: '#0d1117', width: 2 } },
      hoverinfo: 'name',
    },
  ];
  const layout = {
    paper_bgcolor: '#161b22',
    plot_bgcolor: '#0d1117',
    font: { color: '#e6edf3', family: 'JetBrains Mono, ui-monospace, monospace', size: 11 },
    margin: { l: 50, r: 16, t: 16, b: 40 },
    xaxis: { scaleanchor: 'y', scaleratio: 1, gridcolor: '#21262d', zerolinecolor: '#30363d', tickfont: { color: '#8b949e' } },
    yaxis: { gridcolor: '#21262d', zerolinecolor: '#30363d', tickfont: { color: '#8b949e' } },
    showlegend: true,
    legend: { x: 0.01, y: 0.01, bgcolor: 'rgba(22,27,34,0.9)', bordercolor: '#30363d', borderwidth: 1, font: { size: 10 } },
    hovermode: 'closest',
  };
  Plotly.newPlot('map', traces, layout, { displayModeBar: false, responsive: true });
  document.getElementById('map').on('plotly_click', (ev) => {
    if (!ev.points || !ev.points.length) return;
    const p = ev.points[0];
    if (p.curveNumber !== 2) return;
    if (typeof p.customdata === 'number') goToStep(p.customdata);
  });
}

function classifyNextTurn(nt) {
  if (!nt) return '';
  const m = nt.match(/(\d+(?:\.\d+)?)m/);
  if (!m) return '';
  const dist = parseFloat(m[1]);
  if (dist <= 0.0) return 'next-turn-now';
  if (dist <= 2.5) return 'next-turn-warn';
  return '';
}

function describeAction(s) {
  if (s.done) return 'STOP — episode terminated';
  if (Math.abs(s.forward) > 1e-6) return `move forward ${s.forward.toFixed(2)} m`;
  if (s.yaw_act > 1e-6) return `turn LEFT ${s.yaw_act.toFixed(1)}° in place`;
  if (s.yaw_act < -1e-6) return `turn RIGHT ${(-s.yaw_act).toFixed(1)}° in place`;
  return 'no-op (unparseable reply)';
}

function render(i) {
  if (!RUN) return;
  cur = i;
  const s = RUN.steps[i];

  document.getElementById('fpv').src = s.frame;
  document.getElementById('ov-step').textContent = i;
  document.getElementById('ov-d2g').textContent = s.d2g.toFixed(2) + ' m';
  document.getElementById('ov-reply').textContent = s.action_kw || s.reply || '-';

  const ntClass = classifyNextTurn(s.next_turn);
  const lookaheadHtml = (s.lookahead || []).map(l => `<div>${l}</div>`).join('');

  const instr = s.instruction || RUN.summary.instruction || '(no instruction)';
  const pos = s.vlm_position || `${s.x.toFixed(1)}, ${s.y.toFixed(1)}`;
  const heading = s.vlm_heading || `${s.yaw.toFixed(0)} degrees`;
  const goal = s.vlm_goal || '-';
  const remaining = s.vlm_remaining || `${s.d2g.toFixed(1)} m`;
  const progress = s.vlm_progress || '-';
  const motion = describeAction(s);

  document.getElementById('info').innerHTML = `
    <div class="section sec-input">
      <div class="section-header">
        <div class="stripe"></div>
        <div class="ttl">VLM Input</div>
        <div class="sub">image (left panel) + text prompt below</div>
      </div>
      <div class="row"><span class="label">task</span><span class="value">${instr}</span></div>
      <div class="row"><span class="label">step</span><span class="value">${i} / ${RUN.steps.length - 1}</span></div>
      <div class="row"><span class="label">position</span><span class="value">${pos}</span></div>
      <div class="row"><span class="label">heading</span><span class="value">${heading}</span></div>
      <div class="row"><span class="label">goal</span><span class="value">${goal}</span></div>
      <div class="row"><span class="label">remaining</span><span class="value">${remaining}  ·  progress ${progress}</span></div>
      <div class="row"><span class="label">next turn</span><span class="value ${ntClass}">${s.next_turn || '(none)'}</span></div>
      <div class="row"><span class="label">lookahead</span><span class="value" style="color:var(--text-mute);font-size:10.5px;">ego frame: +x=front, +y=left</span></div>
      <div class="lookahead-block">${lookaheadHtml || '(none)'}</div>
      <div class="note">↑ everything above is packed into the text prompt sent to the VLM alongside the FPV image.</div>
    </div>

    <div class="section sec-output">
      <div class="section-header">
        <div class="stripe"></div>
        <div class="ttl">VLM Output</div>
        <div class="sub">reason + action, two lines</div>
      </div>
      <div class="reply-box">
        <div class="reply-line reason-line">
          <span class="lbl">reason</span>
          <span class="val">${s.reason || '(none)'}</span>
        </div>
        <div class="reply-line action-line">
          <span class="lbl">action</span>
          <span class="val">${s.action_kw || '(unparsed)'}</span>
        </div>
      </div>
      <div class="note">VLM replies in a fixed two-line template: a one-sentence REASON, then one of FORWARD / TURN_LEFT / TURN_RIGHT / STOP.</div>
    </div>

    <div class="section sec-action">
      <div class="section-header">
        <div class="stripe"></div>
        <div class="ttl">Robot Action</div>
        <div class="sub">parsed from keyword</div>
      </div>
      <div class="action-block">
        <div class="motion">${motion}</div>
        <div class="raw">forward = ${s.forward.toFixed(2)} m  ·  yaw = ${s.yaw_act.toFixed(2)}°  ·  done = ${s.done}</div>
      </div>
      <div class="note">applied to camera pose by apply_action() — d2g now ${s.d2g.toFixed(2)} m, dense_idx ${s.progress_idx} / ${RUN.path.length - 1}.</div>
    </div>
  `;

  Plotly.restyle('map', { x: [[s.x]], y: [[s.y]] }, [5]);

  const slider = document.getElementById('slider');
  if (parseInt(slider.value) !== i) slider.value = i;
  document.getElementById('cur-step').textContent = i;
}

function goToStep(i) {
  if (!RUN) return;
  i = Math.max(0, Math.min(RUN.steps.length - 1, i));
  render(i);
}

// ── controls ──
document.getElementById('run-select').addEventListener('change', e => loadRun(e.target.value));
document.getElementById('btn-refresh').onclick = loadRunList;
document.getElementById('slider').addEventListener('input', e => goToStep(parseInt(e.target.value)));
document.getElementById('btn-first').onclick = () => goToStep(0);
document.getElementById('btn-prev').onclick = () => goToStep(cur - 1);
document.getElementById('btn-next').onclick = () => goToStep(cur + 1);
document.getElementById('btn-last').onclick = () => { if (RUN) goToStep(RUN.steps.length - 1); };
document.getElementById('btn-play').onclick = togglePlay;
document.getElementById('speed').addEventListener('change', e => {
  playSpeed = parseInt(e.target.value);
  if (playing) { stopPlay(); startPlay(); }
});

function startPlay() {
  if (!RUN) return;
  playing = true;
  document.getElementById('btn-play').textContent = '⏸';
  playTimer = setInterval(() => {
    if (cur >= RUN.steps.length - 1) { stopPlay(); return; }
    goToStep(cur + 1);
  }, playSpeed);
}
function stopPlay() {
  playing = false;
  document.getElementById('btn-play').textContent = '▶';
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
}
function togglePlay() { if (playing) stopPlay(); else startPlay(); }

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (!RUN) return;
  if (e.key === 'ArrowRight') { goToStep(cur + (e.shiftKey ? 10 : 1)); e.preventDefault(); }
  else if (e.key === 'ArrowLeft') { goToStep(cur - (e.shiftKey ? 10 : 1)); e.preventDefault(); }
  else if (e.key === 'Home') { goToStep(0); e.preventDefault(); }
  else if (e.key === 'End') { goToStep(RUN.steps.length - 1); e.preventDefault(); }
  else if (e.key === ' ') { togglePlay(); e.preventDefault(); }
  else if (e.key >= '0' && e.key <= '9') {
    const frac = parseInt(e.key) / 10;
    goToStep(Math.round(frac * (RUN.steps.length - 1)));
  }
});

loadRunList();
</script>
</body>
</html>
"""


# ── HTTP handler ────────────────────────────────────────────────────────────
class ReplayHandler(BaseHTTPRequestHandler):
    base_dir: Path = Path(".")
    project_root: Path = Path(".")
    verbose: bool = False

    def log_message(self, fmt, *args):
        if self.verbose:
            super().log_message(fmt, *args)

    def _send_bytes(self, body: bytes, ctype: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, data, status: int = 200):
        self._send_bytes(
            json.dumps(data).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _send_html(self, html: str, status: int = 200):
        self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8", status)

    def _safe_resolve(self, rel: str) -> Path | None:
        try:
            candidate = (self.base_dir / rel).resolve()
            candidate.relative_to(self.base_dir.resolve())
            return candidate
        except (ValueError, OSError):
            return None

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path in ("/", "/index.html"):
            self._send_html(INDEX_HTML)
            return

        if path == "/api/runs":
            try:
                self._send_json({"runs": list_runs(self.base_dir)})
            except Exception as e:
                traceback.print_exc()
                self._send_json({"error": str(e)}, status=500)
            return

        if path.startswith("/api/run/"):
            run_name = path[len("/api/run/"):]
            run_dir = self._safe_resolve(run_name)
            if run_dir is None or not run_dir.exists() or not run_dir.is_dir():
                self._send_json({"error": f"run not found: {run_name}"}, status=404)
                return
            try:
                payload = build_run_payload(run_dir, self.project_root)
                self._send_json(payload)
            except Exception as e:
                traceback.print_exc()
                self._send_json({"error": str(e)}, status=500)
            return

        if path.startswith("/frames/"):
            # URL: /frames/<run_name>/frame_NNNNNN.png
            # File: <base_dir>/<run_name>/frames/frame_NNNNNN.png
            rel = path[len("/frames/"):]
            parts = rel.split("/", 1)
            if len(parts) != 2:
                self.send_error(404)
                return
            run_name, frame_name = parts
            frame_path = self._safe_resolve(f"{run_name}/frames/{frame_name}")
            if frame_path is None or not frame_path.is_file():
                self.send_error(404)
                return
            try:
                data = frame_path.read_bytes()
            except OSError:
                self.send_error(404)
                return
            self._send_bytes(data, "image/png")
            return

        self.send_error(404)


# ── entry ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", default="data/urbanverse/vlm_gps_nav",
                   help="Directory containing run subdirectories (default: %(default)s)")
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind host (default: %(default)s; use 0.0.0.0 for remote access)")
    p.add_argument("--port", type=int, default=8765, help="Bind port (default: %(default)s)")
    p.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    p.add_argument("--verbose", action="store_true", help="Log every HTTP request")
    return p.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    base_dir = Path(args.base_dir)
    if not base_dir.is_absolute():
        base_dir = (project_root / base_dir).resolve()
    else:
        base_dir = base_dir.resolve()

    if not base_dir.exists():
        sys.exit(f"[error] base dir does not exist: {base_dir}")
    if not base_dir.is_dir():
        sys.exit(f"[error] base dir is not a directory: {base_dir}")

    ReplayHandler.base_dir = base_dir
    ReplayHandler.project_root = project_root
    ReplayHandler.verbose = args.verbose

    server = ThreadingHTTPServer((args.host, args.port), ReplayHandler)
    url = f"http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}"

    print(f"[replay-server] base dir   : {base_dir}")
    runs = list_runs(base_dir)
    print(f"[replay-server] found runs : {len(runs)}")
    for r in runs[:5]:
        mark = "✓" if r["success"] else "✗"
        print(f"                  {mark} {r['name']}  steps={r['total_steps']}  d2g={r['final_dist_to_goal_m']:.2f}m")
    if len(runs) > 5:
        print(f"                  ... +{len(runs) - 5} more")
    print(f"[replay-server] serving on : {url}")
    print(f"[replay-server] press Ctrl-C to stop")

    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[replay-server] stopped")
        server.server_close()


if __name__ == "__main__":
    main()
