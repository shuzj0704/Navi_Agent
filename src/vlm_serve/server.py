"""vLLM OpenAI-compatible server launcher.

Wraps `python -m vllm.entrypoints.openai.api_server` so that model_path,
GPU, port, and other knobs live in YAML configs instead of being hardcoded
in entry scripts. Designed to be reused by Teacher Model labeling, training
data generation, and inference evaluation pipelines.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class VLLMServerConfig:
    model_path: str
    served_model_name: str
    port: int
    gpu: str  # CUDA_VISIBLE_DEVICES value, e.g. "1" or "0,1"
    host: str = "0.0.0.0"
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = True
    extra_args: List[str] = field(default_factory=list)


def build_command(cfg: VLLMServerConfig) -> List[str]:
    """Assemble the vLLM server command from a config."""
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        cfg.model_path,
        "--served-model-name",
        cfg.served_model_name,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
        "--max-model-len",
        str(cfg.max_model_len),
    ]
    if cfg.trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.extend(cfg.extra_args)
    return cmd


def launch(cfg: VLLMServerConfig) -> None:
    """Launch vLLM server in foreground (blocks until process exits)."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": cfg.gpu}
    cmd = build_command(cfg)
    print(
        f"[vlm_serve] launching {cfg.served_model_name} "
        f"on {cfg.host}:{cfg.port} (GPU {cfg.gpu})"
    )
    print(f"[vlm_serve] cmd: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=False)


def load_config(yaml_path: str | Path) -> VLLMServerConfig:
    """Load a server config from YAML."""
    import yaml  # local import so non-server users don't need PyYAML

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return VLLMServerConfig(**data)
