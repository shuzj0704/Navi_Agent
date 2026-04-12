"""Generate a self-contained HTML replay viewer for a vlm_gps_nav run.

Reads a run directory produced by src/sim_vln_outdoor/scripts/vlm_gps_nav.py
and writes <run_dir>/replay.html. Open the HTML in any browser — frames are
loaded via relative paths from <run_dir>/frames/, the trajectory data and
per-step VLM I/O are inlined as JSON, and Plotly is loaded from a CDN.

Usage:
    python scripts/utils/replay_vlm_gps_nav_html.py \
        data/urbanverse/vlm_gps_nav/20260411_180711
    # then open data/urbanverse/vlm_gps_nav/20260411_180711/replay.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── helpers ─────────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_next_turn(prompt: str) -> str:
    for line in prompt.splitlines():
        if "NEXT TURN" in line:
            return line.split("NEXT TURN:", 1)[-1].strip()
    return ""


def extract_lookahead(prompt: str) -> list[str]:
    out, in_block = [], False
    for line in prompt.splitlines():
        if "NEXT 5 WAYPOINTS" in line:
            in_block = True
            continue
        if in_block:
            stripped = line.strip()
            if not stripped:
                break
            if stripped[:2] in ("1.", "2.", "3.", "4.", "5."):
                out.append(stripped)
            else:
                break
    return out


# ── HTML template ──────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>replay · __TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
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
  --code: #0d1117;
  --shadow: 0 1px 3px rgba(0,0,0,0.4);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  font-size: 14px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* Header */
header {
  padding: 14px 22px;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 18px;
  flex-wrap: wrap;
  box-shadow: var(--shadow);
}
header h1 {
  font-size: 15px;
  font-weight: 600;
  color: var(--accent);
  font-family: ui-monospace, "SF Mono", "Menlo", monospace;
  letter-spacing: 0.2px;
}
.badge {
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  border: 1px solid;
}
.badge.success { background: rgba(63,185,80,0.12); color: var(--success); border-color: rgba(63,185,80,0.4); }
.badge.fail { background: rgba(248,81,73,0.12); color: var(--danger); border-color: rgba(248,81,73,0.4); }
.stats {
  display: flex;
  gap: 22px;
  font-size: 12px;
  color: var(--text-dim);
  font-family: ui-monospace, "SF Mono", monospace;
}
.stats span b {
  color: var(--text);
  margin-left: 4px;
  font-weight: 600;
}
.instr-pill {
  margin-left: auto;
  padding: 5px 12px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 14px;
  font-size: 11px;
  color: var(--text-dim);
  max-width: 360px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Main grid */
main {
  flex: 1;
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(0, 1fr);
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

/* FPV panel */
.panel-fpv {
  position: relative;
}
.panel-fpv .img-wrap {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #000;
  min-height: 0;
}
.panel-fpv img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}
.fpv-overlay {
  position: absolute;
  top: 14px;
  left: 14px;
  background: rgba(13,17,23,0.85);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  padding: 8px 14px;
  border-radius: 8px;
  border: 1px solid var(--border);
  font-family: ui-monospace, "SF Mono", monospace;
  font-size: 12px;
  display: flex;
  gap: 14px;
  align-items: center;
}
.fpv-overlay .lbl { color: var(--text-dim); }
.fpv-overlay .val { color: var(--text); font-weight: 600; }
.fpv-overlay .reply-val { color: var(--accent); font-weight: 700; font-size: 14px; }

/* Right column */
.right-col {
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-height: 0;
}
.panel-map {
  flex: 1.3;
  min-height: 220px;
}
#map { width: 100%; height: 100%; }

/* Info panel */
.panel-info {
  flex: 1;
  padding: 16px 20px;
  font-family: ui-monospace, "SF Mono", "Menlo", monospace;
  font-size: 12.5px;
  overflow-y: auto;
  line-height: 1.65;
}
.panel-info .row {
  display: flex;
  align-items: baseline;
  gap: 10px;
}
.panel-info .label {
  color: var(--text-mute);
  min-width: 90px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.panel-info .value { color: var(--text); }
.panel-info .lookahead-block {
  margin-top: 4px;
  padding-left: 100px;
  color: var(--text-dim);
  font-size: 11.5px;
}
.next-turn-warn { color: var(--warn); font-weight: 700; }
.next-turn-now { color: var(--danger); font-weight: 700; }

.divider {
  margin: 14px 0;
  border-top: 1px solid var(--border);
}
.reply-block {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.reply-block .lbl {
  color: var(--text-mute);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.reply-block .val {
  color: var(--accent);
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0.4px;
}

/* Footer */
footer {
  padding: 12px 22px;
  background: var(--panel);
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 14px;
  box-shadow: 0 -1px 3px rgba(0,0,0,0.4);
}
footer button {
  background: var(--panel-2);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 7px 13px;
  border-radius: 6px;
  cursor: pointer;
  font-family: ui-monospace, "SF Mono", monospace;
  font-size: 13px;
  transition: all 0.15s ease;
  min-width: 38px;
}
footer button:hover { background: var(--border); color: var(--accent); }
footer button.play { min-width: 46px; font-size: 14px; }
.slider-wrap {
  flex: 1;
  position: relative;
  display: flex;
  align-items: center;
  height: 22px;
}
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
  width: 18px;
  height: 18px;
  background: var(--accent);
  border-radius: 50%;
  cursor: pointer;
  border: 3px solid var(--bg);
  box-shadow: 0 0 0 1px var(--accent);
}
footer input[type=range]::-moz-range-thumb {
  width: 18px;
  height: 18px;
  background: var(--accent);
  border-radius: 50%;
  cursor: pointer;
  border: 3px solid var(--bg);
}
.step-count {
  font-family: ui-monospace, "SF Mono", monospace;
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
  font-family: ui-monospace, "SF Mono", monospace;
  font-size: 12px;
  cursor: pointer;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--panel); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-mute); }

@media (max-width: 1100px) {
  main { grid-template-columns: 1fr; }
  .panel-fpv { min-height: 360px; }
}
</style>
</head>
<body>

<header>
  <h1>__TITLE__</h1>
  <span class="badge __BADGE_CLASS__">__BADGE_TEXT__</span>
  <div class="stats">
    <span>steps<b>__N_STEPS__</b></span>
    <span>final d2g<b>__FINAL_D2G__ m</b></span>
    <span>route<b>__ROUTE_LEN__ m</b></span>
    <span>tol<b>__GOAL_TOL__ m</b></span>
    <span>freq<b>__CTRL_FREQ__ Hz</b></span>
  </div>
  <div class="instr-pill" title="__INSTRUCTION__">📋 __INSTRUCTION__</div>
</header>

<main>
  <div class="panel panel-fpv">
    <div class="img-wrap"><img id="fpv" src="" alt="frame"></div>
    <div class="fpv-overlay">
      <span><span class="lbl">step</span> <span class="val" id="ov-step">0</span></span>
      <span><span class="lbl">d2g</span> <span class="val" id="ov-d2g">-</span></span>
      <span><span class="lbl">reply</span> <span class="reply-val" id="ov-reply">-</span></span>
    </div>
  </div>

  <div class="right-col">
    <div class="panel panel-map">
      <div id="map"></div>
    </div>
    <div class="panel panel-info" id="info"></div>
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
  <div class="step-count"><b id="cur-step">0</b> / <span id="max-step">0</span></div>
</footer>

<script>
const STEPS = __STEPS_JSON__;
const PATH = __PATH_JSON__;
const WAYPOINTS = __WAYPOINTS_JSON__;

let cur = 0;
let playing = false;
let playTimer = null;
let playSpeed = 200;

// ── Plotly map ──
const planned_x = PATH.map(p => p[0]);
const planned_y = PATH.map(p => p[1]);
const wp_x = WAYPOINTS.map(p => p[0]);
const wp_y = WAYPOINTS.map(p => p[1]);
const actual_x = STEPS.map(s => s.x);
const actual_y = STEPS.map(s => s.y);

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
    customdata: STEPS.map((s, i) => i),
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
    x: [STEPS[0].x], y: [STEPS[0].y], mode: 'markers', name: 'current',
    marker: { color: '#d29922', size: 18, symbol: 'circle', line: { color: '#0d1117', width: 2 } },
    hoverinfo: 'name',
  },
];

const layout = {
  paper_bgcolor: '#161b22',
  plot_bgcolor: '#0d1117',
  font: { color: '#e6edf3', family: 'ui-monospace, monospace', size: 11 },
  margin: { l: 50, r: 16, t: 16, b: 40 },
  xaxis: {
    scaleanchor: 'y', scaleratio: 1,
    gridcolor: '#21262d', zerolinecolor: '#30363d',
    tickfont: { color: '#8b949e' },
  },
  yaxis: {
    gridcolor: '#21262d', zerolinecolor: '#30363d',
    tickfont: { color: '#8b949e' },
  },
  showlegend: true,
  legend: {
    x: 0.01, y: 0.01, bgcolor: 'rgba(22,27,34,0.9)',
    bordercolor: '#30363d', borderwidth: 1,
    font: { size: 10 },
  },
  hovermode: 'closest',
};

Plotly.newPlot('map', traces, layout, { displayModeBar: false, responsive: true });

document.getElementById('map').on('plotly_click', (ev) => {
  if (!ev.points || !ev.points.length) return;
  const p = ev.points[0];
  if (p.curveNumber !== 2) return;
  const idx = p.customdata;
  if (typeof idx === 'number') goToStep(idx);
});

// ── Render ──
function classifyNextTurn(nt) {
  if (!nt) return '';
  const m = nt.match(/(\d+(?:\.\d+)?)m/);
  if (!m) return '';
  const dist = parseFloat(m[1]);
  if (dist <= 0.0) return 'next-turn-now';
  if (dist <= 2.5) return 'next-turn-warn';
  return '';
}

function render(i) {
  cur = i;
  const s = STEPS[i];

  document.getElementById('fpv').src = s.frame;
  document.getElementById('ov-step').textContent = i;
  document.getElementById('ov-d2g').textContent = s.d2g.toFixed(2) + ' m';
  document.getElementById('ov-reply').textContent = s.reply || '-';

  const ntClass = classifyNextTurn(s.next_turn);
  const lookaheadHtml = (s.lookahead || []).map(l => `<div>${l}</div>`).join('');

  document.getElementById('info').innerHTML = `
    <div class="row"><span class="label">step</span><span class="value">${i} / ${STEPS.length - 1}</span></div>
    <div class="row"><span class="label">progress</span><span class="value">${s.progress_idx} / ${PATH.length - 1}</span></div>
    <div class="row"><span class="label">d2g</span><span class="value">${s.d2g.toFixed(2)} m</span></div>
    <div class="row"><span class="label">pose</span><span class="value">x=${s.x.toFixed(2)}  y=${s.y.toFixed(2)}  yaw=${s.yaw.toFixed(1)}°</span></div>
    <div class="row"><span class="label">action</span><span class="value">fwd=${s.forward.toFixed(2)}  yaw=${s.yaw_act.toFixed(2)}  done=${s.done}</span></div>
    <div class="row"><span class="label">next turn</span><span class="value ${ntClass}">${s.next_turn || '(none)'}</span></div>
    <div class="row"><span class="label">lookahead</span></div>
    <div class="lookahead-block">${lookaheadHtml || '(none)'}</div>
    <div class="divider"></div>
    <div class="reply-block">
      <span class="lbl">VLM reply</span>
      <span class="val">${s.reply}</span>
    </div>
  `;

  Plotly.restyle('map', { x: [[s.x]], y: [[s.y]] }, [5]);

  const slider = document.getElementById('slider');
  if (parseInt(slider.value) !== i) slider.value = i;
  document.getElementById('cur-step').textContent = i;
}

function goToStep(i) {
  i = Math.max(0, Math.min(STEPS.length - 1, i));
  render(i);
}

const slider = document.getElementById('slider');
slider.max = STEPS.length - 1;
document.getElementById('max-step').textContent = STEPS.length - 1;
slider.addEventListener('input', e => goToStep(parseInt(e.target.value)));

document.getElementById('btn-first').onclick = () => goToStep(0);
document.getElementById('btn-prev').onclick = () => goToStep(cur - 1);
document.getElementById('btn-next').onclick = () => goToStep(cur + 1);
document.getElementById('btn-last').onclick = () => goToStep(STEPS.length - 1);
document.getElementById('btn-play').onclick = togglePlay;
document.getElementById('speed').addEventListener('change', e => {
  playSpeed = parseInt(e.target.value);
  if (playing) { stopPlay(); startPlay(); }
});

function startPlay() {
  playing = true;
  document.getElementById('btn-play').textContent = '⏸';
  playTimer = setInterval(() => {
    if (cur >= STEPS.length - 1) { stopPlay(); return; }
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
  if (e.key === 'ArrowRight') { goToStep(cur + (e.shiftKey ? 10 : 1)); e.preventDefault(); }
  else if (e.key === 'ArrowLeft') { goToStep(cur - (e.shiftKey ? 10 : 1)); e.preventDefault(); }
  else if (e.key === 'Home') { goToStep(0); e.preventDefault(); }
  else if (e.key === 'End') { goToStep(STEPS.length - 1); e.preventDefault(); }
  else if (e.key === ' ') { togglePlay(); e.preventDefault(); }
  else if (e.key >= '0' && e.key <= '9') {
    const frac = parseInt(e.key) / 10;
    goToStep(Math.round(frac * (STEPS.length - 1)));
  }
});

render(0);
</script>
</body>
</html>
"""


# ── main ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=str)
    p.add_argument("--trajectory", type=str, default=None,
                   help="Override dense_trajectory.json path (default: read from summary.json).")
    p.add_argument("--out", type=str, default=None,
                   help="Output HTML path (default: <run_dir>/replay.html)")
    return p.parse_args()


def resolve_trajectory(run_dir: Path, summary: dict, override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    project_root = Path(__file__).resolve().parents[2]
    tp = summary.get("trajectory_path", "")
    if tp:
        cand = project_root / tp
        if cand.exists():
            return cand
    fallback = project_root / "data/urbanverse/trajectory/scene_09/dense_trajectory.json"
    return fallback


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        sys.exit(f"[error] run dir does not exist: {run_dir}")
    for f in ("summary.json", "trajectory.jsonl", "vlm_io.jsonl", "frames"):
        if not (run_dir / f).exists():
            sys.exit(f"[error] missing {f} in {run_dir}")

    with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    traj = load_jsonl(run_dir / "trajectory.jsonl")
    vlm = load_jsonl(run_dir / "vlm_io.jsonl")
    if len(traj) != len(vlm):
        sys.exit(f"[error] trajectory ({len(traj)}) and vlm_io ({len(vlm)}) length mismatch")

    traj_path = resolve_trajectory(run_dir, summary, args.trajectory)
    if not traj_path.exists():
        sys.exit(f"[error] dense trajectory not found: {traj_path}")
    with open(traj_path, "r", encoding="utf-8") as f:
        dense = json.load(f)

    steps = []
    for i, (t, v) in enumerate(zip(traj, vlm)):
        pose = t["pose"]
        action = t.get("action", {})
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
            "next_turn": extract_next_turn(v.get("prompt", "")),
            "lookahead": extract_lookahead(v.get("prompt", "")),
            "frame": f"frames/frame_{i:06d}.png",
        })

    path_xy = [[p[0], p[1]] for p in dense["path"]]
    waypoints_xy = [[w["pos"][0], w["pos"][1]] for w in dense["waypoints"]]

    success = bool(summary.get("success", False))
    replacements = {
        "__TITLE__": run_dir.name,
        "__BADGE_CLASS__": "success" if success else "fail",
        "__BADGE_TEXT__": "success" if success else "failed",
        "__N_STEPS__": str(summary.get("total_steps", len(steps))),
        "__FINAL_D2G__": f"{summary.get('final_dist_to_goal_m', 0):.2f}",
        "__ROUTE_LEN__": f"{summary.get('route_length_m', 0):.2f}",
        "__GOAL_TOL__": f"{summary.get('goal_tol_m', 0):.1f}",
        "__CTRL_FREQ__": f"{summary.get('controller_freq_hz', 0):.1f}",
        "__INSTRUCTION__": (summary.get("instruction") or "").replace('"', "'"),
        "__STEPS_JSON__": json.dumps(steps, separators=(",", ":")),
        "__PATH_JSON__": json.dumps(path_xy, separators=(",", ":")),
        "__WAYPOINTS_JSON__": json.dumps(waypoints_xy, separators=(",", ":")),
    }
    html = HTML_TEMPLATE
    for k, v in replacements.items():
        html = html.replace(k, v)

    out = Path(args.out).resolve() if args.out else (run_dir / "replay.html")
    out.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {out}")
    print(f"     open in browser: file://{out}")


if __name__ == "__main__":
    main()
