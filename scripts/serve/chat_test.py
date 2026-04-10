#!/usr/bin/env python3
"""Interactive chat client for testing a vLLM-served VLM.

Examples:
    # Test Qwen3.5 (default)
    python scripts/serve/chat_test.py

    # Test Qwen3-VL
    python scripts/serve/chat_test.py --base-url http://localhost:8004/v1 --model qwen3-vl
"""
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vlm_serve.client import VLMClient


def main():
    parser = argparse.ArgumentParser(description="Interactive chat client for vLLM server")
    parser.add_argument("--base-url", default="http://localhost:8003/v1")
    parser.add_argument("--model", default="qwen3.5")
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    client = VLMClient(base_url=args.base_url, model=args.model)
    messages = []
    print(f"{args.model} 交互式对话 (输入 'quit' 退出)\n")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见!")
            break

        messages.append({"role": "user", "content": user_input})
        t0 = time.time()
        print(f"\n{args.model}: ", end="", flush=True)
        reply = ""
        for delta in client.chat_stream_text(
            messages, enable_thinking=args.enable_thinking
        ):
            print(delta, end="", flush=True)
            reply += delta
        print(f"\n[{time.time() - t0:.2f}s]\n")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
