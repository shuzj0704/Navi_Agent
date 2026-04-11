"""VLM-based navigation controller.

Calls a vLLM-served VLM (e.g. Qwen3-VL) with the current first-person RGB
observation and parses a discrete action keyword from the model reply.

Usage with nav_eval.py:
    ./python.sh src/sim_vln_outdoor/scripts/nav_eval.py \\
        --headless --max-steps 100 --controller-freq 1.0 \\
        --controller "nav.vlm_controller:VLMNavController" \\
        --controller-kwargs '{"instruction":"explore the construction site"}'

Requires the VLM server running first:
    conda activate lwy_swift
    python scripts/serve/start_qwen3vl.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

from .controller import Action, NavController, Observation


# Resolve repo root so we can import vlm_serve which lives in src/vlm_serve.
# This file is at src/sim_vln_outdoor/nav/vlm_controller.py
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


class VLMNavController(NavController):
    """Closed-loop navigation controller backed by a vision-language model.

    Each step:
      1. Save obs.rgb to disk (under <project>/data/.../vlm_inputs/)
      2. Send the image + action prompt to the VLM
      3. Parse the keyword reply (FORWARD / TURN_LEFT / TURN_RIGHT / STOP)
         into an Action.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8004/v1",
        model: str = "qwen3-vl",
        instruction: str = "Explore the urban street and avoid obstacles.",
        forward_step: float = 0.5,   # meters per FORWARD action
        yaw_step: float = 15.0,      # degrees per TURN action
        max_tokens: int = 32,
        save_inputs: bool = True,
        input_dir: str | None = None,
    ):
        from vlm_serve.client import VLMClient

        self.client = VLMClient(base_url=base_url, model=model)
        self.instruction = instruction
        self.forward_step = forward_step
        self.yaw_step = yaw_step
        self.max_tokens = max_tokens
        self.save_inputs = save_inputs

        if input_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_dir = str(
                _REPO_ROOT / "data" / "urbanverse" / "vlm_inputs" / timestamp
            )
        self.input_dir = Path(input_dir)
        if self.save_inputs:
            self.input_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[VLMNavController] base_url={base_url} model={model} "
            f"forward_step={forward_step}m yaw_step={yaw_step}deg"
        )
        print(f"[VLMNavController] instruction: {instruction}")
        if self.save_inputs:
            print(f"[VLMNavController] inputs -> {self.input_dir}")

    # ── public interface ────────────────────────────────────────────────

    def act(self, obs: Observation) -> Action:
        img_path = self.input_dir / f"step_{obs.step:06d}.png"
        # Always need an on-disk path for the VLMClient API; whether we
        # *keep* the file is controlled by save_inputs.
        Image.fromarray(obs.rgb).save(img_path)

        prompt = self._build_prompt()
        try:
            reply = self.client.chat_with_image(
                prompt=prompt,
                image_path=img_path,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            print(f"[VLMNavController] VLM call failed: {e} -- no-op")
            if not self.save_inputs:
                img_path.unlink(missing_ok=True)
            return Action()

        if not self.save_inputs:
            img_path.unlink(missing_ok=True)

        action = self._parse_action(reply)
        print(f"[VLMNavController] step={obs.step} reply={reply!r} -> {action}")
        return action

    # ── helpers ─────────────────────────────────────────────────────────

    def _build_prompt(self) -> str:
        return (
            f"You are an outdoor navigation robot. Task: {self.instruction}\n"
            f"Looking at this first-person camera view, choose ONE next action:\n"
            f"- FORWARD: move forward {self.forward_step} meters\n"
            f"- TURN_LEFT: turn left {self.yaw_step} degrees in place\n"
            f"- TURN_RIGHT: turn right {self.yaw_step} degrees in place\n"
            f"- STOP: task complete or unsafe to proceed\n"
            f"Reply with ONE keyword only (FORWARD / TURN_LEFT / TURN_RIGHT / STOP). "
            f"No explanation."
        )

    def _parse_action(self, reply: str) -> Action:
        text = (reply or "").strip().upper()
        # Order matters: TURN_LEFT/TURN_RIGHT before bare LEFT/RIGHT,
        # FORWARD before STOP since "stop forward" would be ambiguous.
        if "TURN_LEFT" in text or "TURN LEFT" in text:
            return Action(yaw=self.yaw_step)
        if "TURN_RIGHT" in text or "TURN RIGHT" in text:
            return Action(yaw=-self.yaw_step)
        if "FORWARD" in text:
            return Action(forward=self.forward_step)
        if "STOP" in text:
            return Action(done=True)
        # Loose fallbacks for one-word replies
        if text == "LEFT":
            return Action(yaw=self.yaw_step)
        if text == "RIGHT":
            return Action(yaw=-self.yaw_step)
        print(f"[VLMNavController] unparseable reply: {reply!r} -- no-op")
        return Action()
