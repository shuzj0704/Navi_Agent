"""
SAM3 image-mode smoke test
==========================
Loads Sam3Segmentor (base model), runs on a single image with a given class list,
and saves an overlay PNG next to the input.

Usage (inside `naviagent` conda env):
    python src/scripts/test_sam3.py --image path/to/img.jpg
    python src/scripts/test_sam3.py --image img.jpg --classes "chair,table,sofa"
"""
import argparse
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
sys.path.insert(0, _SRC_ROOT)

import cv2
import numpy as np

from naviagent.perception import Sam3Segmentor


PALETTE = [
    (86, 180, 233), (0, 158, 115), (230, 159, 0), (0, 114, 178),
    (213, 94, 0), (204, 121, 167), (240, 228, 66), (0, 204, 204),
    (148, 103, 189), (44, 160, 44), (214, 39, 40), (255, 127, 14),
]


def overlay(rgb_bgr, segments):
    """Draw mask (alpha 0.45) + bbox + label for each Segment."""
    vis = rgb_bgr.copy()
    label_colors = {}
    for i, seg in enumerate(segments):
        color = label_colors.setdefault(seg.label, PALETTE[len(label_colors) % len(PALETTE)])

        mask_rgb = np.zeros_like(vis)
        mask_rgb[seg.mask] = color
        vis = cv2.addWeighted(vis, 1.0, mask_rgb, 0.45, 0)

        x1, y1, x2, y2 = seg.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        tag = f"{seg.label} {seg.confidence:.2f}"
        cv2.putText(vis, tag, (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis, tag, (x1, max(y1 - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return vis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="input image path (any cv2-readable)")
    p.add_argument("--classes", default=None,
                   help="comma-separated class list; defaults to Sam3Segmentor.DEFAULT_CLASSES")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--out", default=None, help="output overlay path; default <image>_sam3.png")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read image: {args.image}")

    classes = (
        [c.strip() for c in args.classes.split(",") if c.strip()]
        if args.classes else None
    )
    print(f"[sam3] loading model (first run downloads checkpoint from HF)...")
    seg = Sam3Segmentor(classes=classes, conf=args.conf, device=args.device)
    print(f"[sam3] classes: {seg.classes}")
    print(f"[sam3] image: {img.shape}")

    segs = seg.segment(img)
    print(f"[sam3] {len(segs)} detections")
    for s in segs:
        print(f"  - {s.label:14s} conf={s.confidence:.3f}  bbox={s.bbox}")

    vis = overlay(img, segs)
    out = args.out or os.path.splitext(args.image)[0] + "_sam3.png"
    cv2.imwrite(out, vis)
    print(f"[sam3] wrote overlay → {out}")


if __name__ == "__main__":
    main()
