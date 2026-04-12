"""Thin OpenAI client wrapper for vLLM-served VLMs.

Centralizes the boilerplate (base64 image encoding, enable_thinking toggle,
streaming token iterator) so downstream code (Teacher Model labeling,
inference evaluation, interactive testing) can share one client API.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Iterator, List, Optional


class VLMClient:
    """Wraps an OpenAI-compatible chat client pointed at a vLLM server."""

    def __init__(self, base_url: str, model: str, api_key: str = "none"):
        # Local import keeps this module importable without `openai` installed
        # in environments that only need `server.py`.
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def chat(
        self,
        messages: List[dict],
        stream: bool = False,
        enable_thinking: bool = False,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Forward to chat.completions.create with Qwen-style thinking toggle."""
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": enable_thinking}
            },
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return self.client.chat.completions.create(**kwargs)

    def chat_stream_text(
        self,
        messages: List[dict],
        enable_thinking: bool = False,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """Yield content deltas as plain text strings."""
        stream = self.chat(
            messages,
            stream=True,
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    @staticmethod
    def encode_image(image_path: str | Path) -> str:
        """Read an image file and return its base64-encoded string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def chat_with_image(
        self,
        prompt: str,
        image_path: str | Path,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
        image_format: str = "png",
    ) -> str:
        """Single-image chat completion. Returns the assistant text content."""
        b64 = self.encode_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{b64}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        resp = self.chat(
            messages,
            stream=False,
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
