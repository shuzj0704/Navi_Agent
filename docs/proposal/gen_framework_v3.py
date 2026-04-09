"""NaviAgent v3 framework — strict Manhattan routing, no overlaps, no text collision."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(1, 1, figsize=(26, 15))
ax.set_xlim(0, 26)
ax.set_ylim(0, 15)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Palette ──
C_IN  = "#E8F5E9"; C_TOK = "#FFFDE7"; C_VLM = "#EDE7F6"; C_MEM = "#FFF3E0"
C_API = "#E3F2FD"; C_DIT = "#FCE4EC"; C_OUT = "#ECEFF1"; C_HD  = "#F3E5F5"
C_LOOP= "#FFE0B2"
BD = "#546E7A"
STROKE = [pe.withStroke(linewidth=3, foreground="white")]

def rbox(x, y, w, h, fc, bc=BD, lw=1.5):
    ax.add_patch(FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.12",
                                fc=fc, ec=bc, lw=lw))

def txt(x, y, s, fs=11, bold=True, color="#212121", **kw):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=color, **kw)

def sub(x, y, s, fs=8):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs,
            color="#616161", style="italic")

def pill(x, y, label, fc, w=None):
    if w is None:
        w = len(label)*0.13 + 0.35
    ax.add_patch(FancyBboxPatch((x,y), w, 0.42, boxstyle="round,pad=0.06",
                                fc=fc, ec="#BDBDBD", lw=0.8))
    ax.text(x+w/2, y+0.21, label, ha="center", va="center",
            fontsize=7.5, family="monospace", color="#212121")
    return w

def harr(x1, x2, y, c=BD, lw=1.8, ls="-", head=True):
    if head:
        ax.annotate("", xy=(x2,y), xytext=(x1,y),
                    arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.12",
                                    color=c, lw=lw, linestyle=ls))
    else:
        ax.plot([x1,x2],[y,y], color=c, lw=lw, linestyle=ls)

def varr(x, y1, y2, c=BD, lw=1.8, ls="-", head=True):
    if head:
        ax.annotate("", xy=(x,y2), xytext=(x,y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.12",
                                    color=c, lw=lw, linestyle=ls))
    else:
        ax.plot([x,x],[y1,y2], color=c, lw=lw, linestyle=ls)

def label(x, y, s, c=BD, fs=8):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=c,
            fontweight="bold", path_effects=STROKE)

# ================================================================
# LAYOUT GRID  (generous spacing)
#
#   Y=14.0          Title
#   Y=12.5-13.5     Route API (col-B) | Loop Detection (col-D)
#   Y=10.5-12.0     NaVocab   (col-B) | Memory         (col-D)
#   Y=9.5           ---- gap for labels ----
#   Y=4.0-9.0       VLM  (col-B ~ col-C, wide)
#   Y=6.0-7.0       Spatial Head / Latent Goal Head (col-D)
#   Y=2.0-3.5       DiT (col-D) | Low-Level Control (col-E)
#
#   COL-A  0.3-3.2   Inputs
#   COL-B  4.0-7.5   NaVocab / left VLM
#   COL-C  7.5-14.5  VLM center-right
#   COL-D  16.0-20.5 Memory / Heads / DiT
#   COL-E  21.5-25.0 Low-Level / Legend
# ================================================================

# ── Title ──
txt(13, 14.5, "NaviAgent  —  System Architecture", fs=20, color="#1A237E")

# ================================================================
# COL-A: Inputs
# ================================================================
for i, (lab, s, cy) in enumerate([
    ("GPS + IMU", "Confidence · Heading", 9.8),
    ("Destination", "Natural language", 8.0),
    ("RGB-D (D435i)", "4-view current + history", 6.0),
]):
    rbox(0.3, cy-0.4, 2.9, 0.8, C_IN)
    txt(1.75, cy+0.1, lab, fs=10)
    sub(1.75, cy-0.18, s, fs=7.5)

# ================================================================
# COL-B top: NaVocab  (x=4.0-7.5, y=10.5-12.0)
# ================================================================
rbox(4.0, 10.5, 3.5, 1.5, C_TOK, bc="#F9A825", lw=2)
txt(5.75, 11.6, "NaVocab", fs=13)
sub(5.75, 11.2, "Navigation Token Vocabulary", fs=8.5)
ax.text(5.75, 10.8, "<gps:X>  <floor:Y>  <env:Z>  <scene:*>  <wp:dX:aY>",
        ha="center", va="center", fontsize=7, color="#6A1B9A", family="monospace")

# ================================================================
# Route API  (x=4.0-7.5, y=12.7-13.5)
# ================================================================
rbox(4.0, 12.7, 3.5, 0.8, C_API, bc="#1E88E5", lw=1.5)
txt(5.75, 13.2, "Map App / Route API", fs=10)
sub(5.75, 12.9, "AMap Walking API")

# ================================================================
# COL-D top: Memory  (x=16.0-20.5, y=10.5-12.0)
# ================================================================
rbox(16.0, 10.5, 4.5, 1.5, C_MEM, bc="#EF6C00", lw=2)
txt(18.25, 11.6, "Differentiable Spatial Memory", fs=11)
ax.text(18.25, 11.15, "Memory Bank  { m₁, m₂, … , mₖ }", ha="center", va="center",
        fontsize=8, color="#BF360C", family="monospace")
sub(18.25, 10.8, "Learned embeddings as VLM context tokens", fs=7.5)

# Loop Detection  (x=16.0-20.5, y=12.7-13.5)
rbox(16.0, 12.7, 4.5, 0.8, C_LOOP, bc="#FF9800", lw=1.5)
txt(18.25, 13.2, "Loop Detection + Progress Check", fs=10)
sub(18.25, 12.9, "learned similarity · topo distance")

# ================================================================
# VLM  (x=4.0-14.5, y=4.0-9.0)
# ================================================================
VX, VY, VW, VH = 4.0, 4.0, 10.5, 5.0
rbox(VX, VY, VW, VH, C_VLM, bc="#6A1B9A", lw=3)
txt(VX+VW/2, VY+VH-0.4, "System 2 :  Qwen3.5-9B   (2 Hz)", fs=14, color="#4A148C")

# ── Input token bar ──
ity = VY+VH - 1.3
txt(VX+0.5, ity+0.2, "Input :", fs=9, color="#616161")
tx = VX + 1.8
for tok, fc in [("[IMG]","#A5D6A7"), ("[Task]","#A5D6A7"),
                ("[NaVocab]","#FFF59D"), ("[m₁···mₖ]","#FFCC80")]:
    w = pill(tx, ity, tok, fc); tx += w + 0.15

# ── Output token bar ──
oty = ity - 0.9
txt(VX+0.5, oty+0.2, "Output :", fs=9, color="#616161")
tx = VX + 2.0
for tok, fc in [("<think>","#CE93D8"), ("<t₁>…<tₖ>","#CE93D8"),
                ("</think>","#CE93D8"), ("<tool:mem_w>","#90CAF9"),
                ("<tool:route>","#90CAF9"), ("<act>","#EF9A9A")]:
    w = pill(tx, oty, tok, fc); tx += w + 0.1

# ── Sub-modules (inside VLM) ──
sy = VY + 0.5; sh = 1.4
rbox(VX+0.5, sy, 3.0, sh, "#D1C4E9", bc="#7E57C2", lw=1.2)
txt(VX+2.0, sy+sh/2+0.2, "Latent Adaptive", fs=10)
txt(VX+2.0, sy+sh/2-0.05, "Reasoning", fs=10)
sub(VX+2.0, sy+sh/2-0.35, "text → latent → adaptive", fs=7)

rbox(VX+3.8, sy, 2.8, sh, "#BBDEFB", bc="#42A5F5", lw=1.2)
txt(VX+5.2, sy+sh/2+0.2, "Tool Token", fs=10)
txt(VX+5.2, sy+sh/2-0.05, "Dispatch", fs=10)
sub(VX+5.2, sy+sh/2-0.35, "<tool:*> → ops", fs=7)

rbox(VX+6.9, sy, 2.8, sh, "#F8BBD0", bc="#EC407A", lw=1.2)
txt(VX+8.3, sy+sh/2+0.2, "Dual-Head", fs=10)
txt(VX+8.3, sy+sh/2-0.05, "Decoding", fs=10)
sub(VX+8.3, sy+sh/2-0.35, "<act> → 2 heads", fs=7)

sub(VX+VW/2, VY+0.12, "✦ end-to-end differentiable (except Route API)", fs=8)

# ================================================================
# COL-D mid: Heads  (x=16.0-20.5)
# ================================================================
rbox(16.0, 7.2, 4.5, 0.9, C_HD, bc="#8E24AA")
txt(18.25, 7.75, "Spatial Head (MLP)", fs=10.5)
sub(18.25, 7.45, "pixel_goal  (view, x, y)")

rbox(16.0, 5.8, 4.5, 0.9, C_HD, bc="#E65100")
txt(18.25, 6.35, "Latent Goal Head (MLP)", fs=10.5)
sub(18.25, 6.05, "z_goal ∈ ℝ^d")

# ================================================================
# COL-D bot: DiT  (x=16.0-20.5, y=2.5-4.0)
# ================================================================
rbox(16.0, 2.5, 4.5, 1.5, C_DIT, bc="#C62828", lw=2)
txt(18.25, 3.45, "System 1 : DiT", fs=12)
sub(18.25, 3.0, "Diffusion Policy · 30 Hz · RGB+D")

# ================================================================
# COL-E: Low-Level  (x=21.5-25.0, y=2.5-4.0)
# ================================================================
rbox(21.5, 2.5, 3.5, 1.5, C_OUT, bc="#455A64")
txt(23.25, 3.45, "Low-Level Control", fs=12)
sub(23.25, 3.0, "RL Locomotion · 200 Hz")

# ================================================================
# ARROWS  (all horizontal / vertical, labels placed manually)
# ================================================================

# ── A1: Destination → VLM  (horizontal right, standard text tokens) ──
harr(3.2, VX, 8.0, c="#43A047", lw=1.5)
label(3.6, 8.25, "task text", c="#43A047", fs=7.5)

# ── A2: GPS+IMU → NaVocab  (L-shape: up at x=3.5, right to NaVocab) ──
varr(3.5, 9.8, 10.75, c="#558B2F", lw=1.5, head=False)    # up
harr(3.5, 4.0, 10.75, c="#558B2F", lw=1.5)                  # right → NaVocab
label(3.5, 10.3, "GPS", c="#558B2F", fs=7.5)

# ── A3: RGB Observations → VLM  (horizontal) ──
harr(3.2, VX, 6.0, c="#43A047", lw=1.5)
label(3.55, 6.25, "RGB→ViT, D→DPE", c="#43A047", fs=7.5)

# ── A5: Route API → NaVocab  (vertical down: waypoints response) ──
varr(5.0, 12.7, 12.0, c="#1E88E5", lw=1.5)
label(5.0, 12.35, "waypoints", c="#1E88E5", fs=7.5)

# ── A5b: VLM → Route API  (trigger: L-shape dashed, up at x=8.0 to avoid NaVocab) ──
#     vertical: x=8.0, y=9.0 → 13.1   |   horizontal: x=8.0 → 7.5(API right), y=13.1
varr(8.0, 9.0, 13.1, c="#1E88E5", lw=1.3, ls="--", head=False)
harr(8.0, 7.5, 13.1, c="#1E88E5", lw=1.3, ls="--")
label(8.6, 11.0, "<tool:route>", c="#1E88E5", fs=7.5)

# ── A6: NaVocab → VLM  (vertical down) ──
varr(5.75, 10.5, 9.0, c="#F9A825", lw=2)
label(6.4, 9.75, "NaVocab tokens", c="#D68600", fs=8)

# ── A8: VLM → Memory write  (L-shape: up from VLM top at x=14, right to Memory) ──
#     vertical: x=14.0, y=9.0 → y=11.0   |   horizontal: x=14.0 → 16.0, y=11.0
varr(14.0, 9.0, 11.0, c="#EF6C00", lw=2, head=False)
harr(14.0, 16.0, 11.0, c="#EF6C00", lw=2)
label(15.0, 11.25, "<tool:mem_w>", c="#EF6C00", fs=8)

# ── A9: Memory → VLM context  (L-shape: left from Memory, down to VLM right) ──
#     horizontal: x=16.0 → 14.5, y=10.7   |   vertical: x=14.5, y=10.7 → 9.0
harr(16.0, 14.5, 10.7, c="#C77700", lw=1.5, head=False)
varr(14.5, 10.7, 9.0, c="#C77700", lw=1.5)
label(15.0, 10.45, "mem tokens", c="#C77700", fs=7.5)

# ── A10: Memory ↔ Loop Detection  (vertical, at x=17.5 up, x=19.0 down) ──
varr(17.5, 12.0, 12.7, c="#FF9800", lw=1.3)
varr(19.0, 12.7, 12.0, c="#FF9800", lw=1.3)

# ── A11: Loop Det → VLM  (L-shape: left at y=13.1, down into VLM at x=13.5) ──
harr(16.0, 13.8, 13.1, c="#FF9800", lw=1.3, ls="--", head=False)
varr(13.8, 13.1, 9.0, c="#FF9800", lw=1.3, ls="--")
label(14.8, 13.35, "<loop> / <no_loop>", c="#FF9800", fs=7.5)

# ── A12: VLM → Spatial Head  (horizontal right) ──
harr(VX+VW, 16.0, 7.65, c="#8E24AA", lw=1.8)

# ── A13: VLM → Latent Goal Head  (horizontal right) ──
harr(VX+VW, 16.0, 6.25, c="#E65100", lw=1.8)

# ── A14a: Spatial Head → DiT  (pixel goal, vertical down at x=17.0) ──
varr(17.0, 7.2, 4.0, c="#8E24AA", lw=1.8)
label(16.3, 5.6, "pixel goal", c="#8E24AA", fs=8)

# ── A14b: Latent Goal → DiT  (z_goal, vertical down at x=19.0) ──
varr(19.0, 5.8, 4.0, c="#E65100", lw=2.2)
label(19.6, 4.9, "z_goal", c="#E65100", fs=9)

# ── A15: DiT → Low-Level  (horizontal right) ──
harr(20.5, 21.5, 3.25, c="#C62828", lw=2.2)
label(21.0, 3.5, "Trajectories", c="#C62828", fs=9)

# ================================================================
# LEGEND  (top-right)
# ================================================================
lx, ly = 22.0, 11.5
txt(lx+1.2, ly+0.7, "Legend", fs=11, color="#212121")
items = [
    (C_IN,   "Sensor Inputs"),
    (C_TOK,  "NaVocab Tokens"),
    (C_VLM,  "VLM Core (System 2)"),
    (C_MEM,  "Differentiable Memory"),
    (C_API,  "External API"),
    (C_DIT,  "DiT Policy (System 1)"),
    (C_OUT,  "Low-Level Control"),
]
for i, (c, lab) in enumerate(items):
    yy = ly + 0.3 - i*0.42
    ax.add_patch(FancyBboxPatch((lx, yy-0.12), 0.4, 0.28,
                                boxstyle="round,pad=0.03", fc=c, ec=BD, lw=0.8))
    ax.text(lx+0.6, yy+0.02, lab, ha="left", va="center", fontsize=8.5, color="#212121")

# line style legend
lsy = ly - 2.9
ax.plot([lx, lx+1.0], [lsy, lsy], ls="-", color="#6A1B9A", lw=2)
ax.text(lx+1.2, lsy, "differentiable", fontsize=8.5, color="#6A1B9A", va="center")
ax.plot([lx, lx+1.0], [lsy-0.4, lsy-0.4], ls="--", color="#1E88E5", lw=2)
ax.text(lx+1.2, lsy-0.4, "non-differentiable (API)", fontsize=8.5, color="#1E88E5", va="center")

# ================================================================
plt.savefig("/home/shu22/navigation/Navi_Agent/docs/plan_A/Framework_v3.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig("/home/shu22/navigation/Navi_Agent/docs/plan_A/Framework_v3.pdf",
            bbox_inches="tight", facecolor="white")
print("Saved.")
