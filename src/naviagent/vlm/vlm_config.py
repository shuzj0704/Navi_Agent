"""
VLM 客户端配置
==============
统一 System 1 (VLMNavigator) 和 System 2 (System2Planner) 的模型连接配置。
支持本地 vLLM、DashScope、OpenAI 兼容 API 之间切换。

用法:
    cfg = load_nav_vlm_config("vlm_server/configs/nav_vlm.yaml")
    vlm = VLMNavigator(config=cfg.system1)
    planner = System2Planner(config=cfg.system2)
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class VLMEndpointConfig:
    """单个 VLM 端点的连接 + 生成参数。"""
    api_url: str = "http://10.100.0.1:8000/v1"
    api_key: str = "none"
    model: str = "qwen3-vl"
    temperature: float = 1.0
    max_tokens: int = 100
    enable_thinking: bool = False
    # 透传给 OpenAI client 的 extra_body, 不同 API 格式不同:
    #   vLLM:      {"chat_template_kwargs": {"enable_thinking": false}}
    #   DashScope: {"enable_thinking": true}
    #   OpenAI:    null (不传)
    extra_body: Optional[Dict[str, Any]] = None


@dataclass
class NavVLMConfig:
    """导航用 VLM 配置, 包含 System 1 和 System 2。"""
    system1: VLMEndpointConfig = field(default_factory=VLMEndpointConfig)
    system2: VLMEndpointConfig = field(default_factory=lambda: VLMEndpointConfig(
        temperature=0.3, max_tokens=2048, enable_thinking=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    ))


def _expand_env(val: str) -> str:
    """$ENV_VAR 形式的值展开为环境变量。"""
    if isinstance(val, str) and val.startswith("$"):
        return os.environ.get(val[1:], val)
    return val


def load_nav_vlm_config(path: Optional[str] = None) -> NavVLMConfig:
    """从 YAML 加载配置。path 为 None 时返回默认配置。"""
    if path is None:
        return NavVLMConfig()

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = NavVLMConfig()
    for section_name in ("system1", "system2"):
        if section_name not in raw:
            continue
        section = raw[section_name]
        endpoint = VLMEndpointConfig()
        for key, val in section.items():
            if hasattr(endpoint, key):
                if key == "api_key":
                    val = _expand_env(val)
                setattr(endpoint, key, val)
        setattr(cfg, section_name, endpoint)

    return cfg
