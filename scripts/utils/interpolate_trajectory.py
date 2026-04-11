#!/usr/bin/env python3
"""Linearly interpolate sparse waypoints into a dense navigation trajectory.

Mimics how a real navigation app (Gaode, Google Maps) gives you a route:
only positions — heading is derived from the path geometry itself, not
baked into the waypoint file.

Input format (one waypoint per line, any of):
    [point 1] x y z
    [point 1] pos=[x, y, z]               (legacy, rot= suffix ignored if present)

Blank lines and lines beginning with '#' are treated as comments.

Output: dense_trajectory.json placed next to the input file by default.

Usage:
    python scripts/utils/interpolate_trajectory.py \\
        --input data/urbanverse/trajectory/scene_09/blog_point.txt \\
        --step 0.5

    # With a quick top-down PNG visualization for sanity check
    python scripts/utils/interpolate_trajectory.py \\
        --input .../blog_point.txt --visualize
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np


# Legacy format: [point N] pos=[x, y, z] rot=[...]  — rot is parsed but dropped.
_LEGACY_RE = re.compile(
    r"\[point\s+(\d+)\]\s*pos=\[([^\]]+)\](?:\s*rot=\[[^\]]+\])?"
)
# New format: [point N] x y z   (whitespace-separated, label optional)
_PLAIN_RE = re.compile(
    r"(?:\[point\s+(\d+)\]\s*)?([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)"
)


def parse_waypoints(path: Path) -> List[dict]:
    """Parse blog_point.txt into a list of {id, pos} dicts.

    Supports the new position-only format and the legacy ``pos=[..] rot=[..]``
    format. Lines beginning with ``#`` and blank lines are skipped.
    """
    waypoints = []
    auto_id = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Try legacy first because it's more specific
            m = _LEGACY_RE.match(line)
            if m:
                wp_id = int(m.group(1))
                pos = [float(x.strip()) for x in m.group(2).split(",")]
                waypoints.append({"id": wp_id, "pos": pos})
                continue

            m = _PLAIN_RE.match(line)
            if m:
                if m.group(1) is not None:
                    wp_id = int(m.group(1))
                else:
                    auto_id += 1
                    wp_id = auto_id
                pos = [float(m.group(2)), float(m.group(3)), float(m.group(4))]
                waypoints.append({"id": wp_id, "pos": pos})
                continue

            print(f"[warn] skip unparseable line: {line!r}")

    if not waypoints:
        raise ValueError(f"No waypoints parsed from {path}")
    waypoints.sort(key=lambda w: w["id"])
    return waypoints


def interpolate_path(
    waypoints: List[dict], step_m: float
) -> Tuple[np.ndarray, float]:
    """Linearly interpolate consecutive waypoints with the given step.

    Returns:
        dense_path: (N, 3) array of positions.
        total_length: scalar arc length in XY plane (meters).
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints to interpolate.")

    points = []
    total_length = 0.0
    for i in range(len(waypoints) - 1):
        p0 = np.array(waypoints[i]["pos"], dtype=np.float64)
        p1 = np.array(waypoints[i + 1]["pos"], dtype=np.float64)
        seg = p1 - p0
        seg_len_xy = float(np.linalg.norm(seg[:2]))
        total_length += seg_len_xy

        # Number of samples on this segment so that step <= step_m
        n = max(1, int(np.ceil(seg_len_xy / step_m)))
        # Include p0; the next segment will start with the next p0 (= this p1).
        for k in range(n):
            t = k / n
            points.append(p0 + t * seg)
    # Append the final waypoint exactly
    points.append(np.array(waypoints[-1]["pos"], dtype=np.float64))

    return np.asarray(points), total_length


def visualize(waypoints: List[dict], dense_path: np.ndarray, out_png: Path) -> None:
    """Render a top-down PNG visualization for sanity check (no matplotlib)."""
    from PIL import Image, ImageDraw

    img_w, img_h = 800, 600
    margin = 40
    canvas = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    xs = dense_path[:, 0]
    ys = dense_path[:, 1]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    # Pad
    pad = max((xmax - xmin), (ymax - ymin)) * 0.15 + 1.0
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad

    span_x = xmax - xmin
    span_y = ymax - ymin
    scale = min((img_w - 2 * margin) / span_x, (img_h - 2 * margin) / span_y)

    def to_pixel(x: float, y: float) -> Tuple[int, int]:
        # World y up -> image y down
        px = int(margin + (x - xmin) * scale)
        py = int(margin + (ymax - y) * scale)
        return px, py

    # Draw dense path as polyline
    pts = [to_pixel(p[0], p[1]) for p in dense_path]
    draw.line(pts, fill=(80, 80, 200), width=3)

    # Draw waypoints with id labels
    for wp in waypoints:
        px, py = to_pixel(wp["pos"][0], wp["pos"][1])
        r = 8
        color = (220, 60, 60) if wp["id"] == waypoints[0]["id"] else (
            (60, 180, 60) if wp["id"] == waypoints[-1]["id"] else (60, 60, 200)
        )
        draw.ellipse((px - r, py - r, px + r, py + r), fill=color, outline=(0, 0, 0))
        draw.text((px + r + 2, py - r), f"P{wp['id']}", fill=(0, 0, 0))

    # Legend
    draw.text((10, 10), "Red=start  Blue=mid  Green=goal", fill=(0, 0, 0))
    canvas.save(out_png)
    print(f"[info] visualization -> {out_png}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to blog_point.txt (sparse waypoints).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path. Defaults to <input dir>/dense_trajectory.json.",
    )
    parser.add_argument(
        "--step", type=float, default=0.5,
        help="Interpolation step in meters (default: 0.5).",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Also save a top-down PNG visualization next to the JSON.",
    )
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_path = Path(args.output) if args.output else in_path.parent / "dense_trajectory.json"

    waypoints = parse_waypoints(in_path)
    dense_path, total_length = interpolate_path(waypoints, step_m=args.step)

    payload = {
        "source_file": str(in_path),
        "scene": in_path.parent.name,
        "step_size_m": args.step,
        "total_length_m": float(total_length),
        "num_waypoints": len(waypoints),
        "num_dense_points": int(dense_path.shape[0]),
        "waypoints": waypoints,
        "path": dense_path.tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(
        f"[done] {len(waypoints)} waypoints -> {dense_path.shape[0]} dense points "
        f"({total_length:.1f}m) -> {out_path}"
    )

    if args.visualize:
        png_path = out_path.with_suffix(".png")
        visualize(waypoints, dense_path, png_path)


if __name__ == "__main__":
    main()
