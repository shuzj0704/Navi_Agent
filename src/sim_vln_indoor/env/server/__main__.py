"""
仿真服务器入口
==============
用法: python -m indoor_sim.server [--config indoor_sim/config/sim_server.yaml] [--port 5100]
"""

import os
import argparse
import uvicorn
from .app import create_app

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
