#!/usr/bin/env python3
"""Launch Qwen3.5-9B vLLM server. Defaults from configs/qwen3_5_9b.yaml."""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vlm_serve.server import launch, load_config

DEFAULT_CONFIG = REPO_ROOT / "src/vlm_serve/configs/qwen3_5_9b.yaml"


def main():
    parser = argparse.ArgumentParser(description="Launch Qwen3.5-9B vLLM server")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override model_path from YAML")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--gpu", type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES, e.g. '1' or '0,1'")
    parser.add_argument("--max-model-len", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model_path is not None:
        cfg.model_path = args.model_path
    if args.port is not None:
        cfg.port = args.port
    if args.gpu is not None:
        cfg.gpu = args.gpu
    if args.max_model_len is not None:
        cfg.max_model_len = args.max_model_len

    launch(cfg)


if __name__ == "__main__":
    main()
