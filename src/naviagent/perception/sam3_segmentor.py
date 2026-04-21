"""
SAM3 open-vocabulary segmentation
==================================
Image-mode inference on top of Meta's Segment Anything Model 3 (base model).

Interface matches `YOLOESegmentor.segment(rgb, depth)` → List[Segment].
SAM3 takes ONE text prompt per forward, so we loop the class list per frame.

Env: the `naviagent` conda env (Python 3.12 + torch cu128 + sam3 editable install).
Checkpoint: auto-downloaded from HuggingFace (`facebook/sam3`) on first call;
requires `huggingface-cli login` with access granted at
https://huggingface.co/facebook/sam3
"""

import numpy as np
import cv2
import torch
from PIL import Image

from .yoloe_segmentor import Segment


class Sam3Segmentor:
    """SAM3 image predictor, wrapped to mirror YOLOESegmentor."""

    DEFAULT_CLASSES = [
        "refrigerator", "chair", "table", "computer", "monitor", "laptop",
        "cup", "sofa", "door", "cardboard box", "trash can",
        "toilet", "sink", "bed", "shelf", "cabinet", "window", "TV",
    ]

    def __init__(
        self,
        classes=None,
        conf=0.5,
        device="cuda",
        resolution=1008,
        checkpoint_path=None,
        min_mask_pixels=100,
    ):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.classes = classes or self.DEFAULT_CLASSES
        self.conf = conf
        self.min_mask_pixels = min_mask_pixels
        self.device = device

        self.model = build_sam3_image_model(
            device=device,
            eval_mode=True,
            checkpoint_path=checkpoint_path,
            load_from_HF=(checkpoint_path is None),
        )
        self.processor = Sam3Processor(
            self.model, resolution=resolution, device=device,
            confidence_threshold=conf,
        )

    def set_classes(self, classes):
        self.classes = list(classes)

    def segment(self, rgb, depth=None):
        """
        Args:
            rgb:   (H, W, 3) uint8 BGR (naviagent convention, matches YOLOE wrapper)
            depth: unused, kept for interface parity
        Returns:
            List[Segment]
        """
        if rgb is None or rgb.size == 0:
            return []

        # naviagent passes BGR; SAM3 wants RGB
        rgb_pil = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        H, W = rgb.shape[:2]

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if str(self.device).startswith("cuda")
            else torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
        )

        with torch.inference_mode(), autocast_ctx:
            state = self.processor.set_image(rgb_pil)

            segments = []
            for label in self.classes:
                state = self.processor.set_text_prompt(prompt=label, state=state)

                masks = state.get("masks")
                boxes = state.get("boxes")
                scores = state.get("scores")
                if masks is None or len(masks) == 0:
                    continue

                masks_np = masks.squeeze(1).float().cpu().numpy().astype(bool)   # (N, H, W)
                boxes_np = boxes.float().cpu().numpy()                            # (N, 4) xyxy
                scores_np = scores.float().cpu().numpy()                          # (N,)

                for mask, box, score in zip(masks_np, boxes_np, scores_np):
                    if mask.shape != (H, W):
                        mask = cv2.resize(
                            mask.astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    if int(mask.sum()) < self.min_mask_pixels:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    segments.append(Segment(
                        mask=mask, label=label,
                        confidence=float(score),
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                    ))

        return segments
