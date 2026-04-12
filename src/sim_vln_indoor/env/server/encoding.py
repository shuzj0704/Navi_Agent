"""
图像 / 深度编码工具
===================
将 Habitat 传感器输出编码为 HTTP 传输格式。
"""

import cv2
import numpy as np
from typing import Tuple


def encode_rgb(image: np.ndarray, fmt: str = "jpeg",
               quality: int = 90) -> Tuple[bytes, str]:
    """编码 RGB 图像为字节流。

    Args:
        image: (H, W, 4) RGBA uint8 或 (H, W, 3) RGB uint8
        fmt: "jpeg" | "png" | "raw"
        quality: JPEG 质量 (1-100)
    Returns:
        (data_bytes, content_type)
    """
    # 去 alpha 通道
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    # RGB → BGR (cv2.imencode 期望 BGR)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if fmt == "jpeg":
        ok, buf = cv2.imencode(".jpg", bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes(), "image/jpeg"

    elif fmt == "png":
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("PNG encode failed")
        return buf.tobytes(), "image/png"

    else:  # raw
        return bgr.tobytes(), "application/octet-stream"


def encode_depth(depth: np.ndarray,
                 fmt: str = "raw_f32") -> Tuple[bytes, str]:
    """编码深度图为字节流。

    Args:
        depth: (H, W) 或 (H, W, 1) float32, 单位: 米
        fmt: "raw_f32" | "raw_f16"
    Returns:
        (data_bytes, content_type)
    """
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    if fmt == "raw_f16":
        return depth.astype(np.float16).tobytes(), "application/octet-stream"
    else:  # raw_f32
        return depth.astype(np.float32).tobytes(), "application/octet-stream"


def decode_rgb(data: bytes, content_type: str) -> np.ndarray:
    """解码 RGB 字节流为 BGR numpy 数组 (与 cv2 约定一致)。"""
    if "jpeg" in content_type or "png" in content_type:
        buf = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR
    else:
        raise ValueError(f"Unsupported RGB content type: {content_type}")


def decode_depth(data: bytes, width: int, height: int,
                 dtype: str = "float32") -> np.ndarray:
    """解码深度字节流为 (H, W) float32 numpy 数组。"""
    np_dtype = np.float16 if dtype == "float16" else np.float32
    arr = np.frombuffer(data, dtype=np_dtype)
    return arr.reshape(height, width).astype(np.float32)
