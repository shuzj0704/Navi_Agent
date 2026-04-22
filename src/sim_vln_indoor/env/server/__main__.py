"""
仿真服务器入口
=============
用法: python -m indoor_sim.server [--config indoor_sim/config/sim_server.yaml] [--port 5100]
"""

import os
import argparse
import uvicorn
from .app import create_app

# 强制 EGL 走 NVIDIA vendor (防 libglvnd 误选 Mesa 导致 offscreen EGL 失败).
# 按候选文件存在性挑路径; 都找不到就不设置, 让 libglvnd 扫默认目录.
# 必须在 import habitat_sim 前设置.
_EGL_VENDOR_CANDIDATES = [
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",                             # 标准安装 (nuc)
    "/home/ps/workspace/ll/workspace/Navi_Agent/data/10_nvidia.json",           # ps 特定镜像
]
for _p in _EGL_VENDOR_CANDIDATES:
    if os.path.exists(_p):
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = _p
        break

# 过滤 LD_LIBRARY_PATH: ROS humble / gazebo / rviz_ogre 都自带 GL 栈,
# 会和 habitat Magnum 的 EGL 初始化冲突 (EGL_BAD_PARAMETER 常见根因).
_LDP = os.environ.get("LD_LIBRARY_PATH", "")
if _LDP:
    _BANNED = ("/opt/ros/", "gazebo-", "rviz_ogre_vendor", "cyclonedds")
    _kept = [p for p in _LDP.split(":") if p and not any(b in p for b in _BANNED)]
    os.environ["LD_LIBRARY_PATH"] = ":".join(_kept)

os.environ["DISPLAY"] = os.environ.get("DISPLAY", ":0")

_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config", "sim_server.yaml"
)


def main():
    parser = argparse.ArgumentParser(description="VLN Sim Server")
    parser.add_argument("--config", default=_DEFAULT_CONFIG,
                        help="YAML 配置文件路径")
    parser.add_argument("--port", type=int, default=None,
                        help="覆盖配置文件中的端口号")
    args = parser.parse_args()

    app = create_app(args.config)
    port = args.port or app.state.config.port
    host = app.state.config.host

    print(f"[SimServer] Starting on {host}:{port}")
    print(f"[SimServer] Config: {args.config}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()