"""
语义分割器
==========
接口 + Mock 实现 (深度边缘分割) + SAM3 实现 + YOLOE 实现

仿真测试: MockSegmentor (无需额外依赖)
正式评测: SAM3Segmentor / YOLOESegmentor (需 ultralytics)
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Segment:
    """单个分割结果"""
    mask: np.ndarray        # (H, W) bool
    label: str              # 语义标签
    confidence: float       # 置信度
    bbox: tuple             # (x1, y1, x2, y2) 像素边界框


class YOLOESegmentor:
    """
    YOLOE open-vocabulary segmentation detector.

    Usage:
        segmentor = YOLOESegmentor("Navi_Agent/models/yoloe-11l-seg.pt")
        segments = segmentor.segment(rgb)
    """

    DEFAULT_CLASSES = [
        "refrigerator", "chair", "table", "computer", "monitor", "laptop",
        "cup", "sofa", "door", "cardboard box", "trash can",
        "toilet", "sink", "bed", "shelf", "cabinet", "window", "TV",
    ]

    def __init__(self, model_path="Navi_Agent/models/yoloe-11l-seg.pt", classes=None, conf=0.15):
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLOE 模型文件不存在: {model_path}\n"
                f"如果是从 Git 仓库克隆的项目，请先拉取 LFS 文件:\n"
                f"  git lfs install && git lfs pull"
            )
        from ultralytics import YOLOE
        self.model = YOLOE(model_path)
        self.classes = classes or self.DEFAULT_CLASSES
        self.model.set_classes(self.classes, self.model.get_text_pe(self.classes))
        self.conf = conf

    def segment(self, rgb, depth=None):
        results = self.model.predict(rgb, conf=self.conf, verbose=False)
        segments = []

        for r in results:
            if r.masks is None:
                continue
            for j, mask_data in enumerate(r.masks.data):
                mask = mask_data.cpu().numpy().astype(bool)
                # resize mask to match input image if needed
                if mask.shape != rgb.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                if mask.sum() < 100:
                    continue

                cls_id = int(r.boxes.cls[j]) if r.boxes is not None else -1
                label = r.names.get(cls_id, f"cls_{cls_id}") if hasattr(r, "names") else f"object_{j}"
                conf = float(r.boxes.conf[j]) if r.boxes is not None else 0.5

                ys, xs = np.where(mask)
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

                segments.append(Segment(
                    mask=mask, label=label,
                    confidence=conf, bbox=bbox,
                ))

        return segments
