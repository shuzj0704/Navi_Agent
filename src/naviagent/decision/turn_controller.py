"""
转向决策控制器
==============
根据 VLM 输出的 view 决定机器人行为:
  front → 交给 DWA (vx, vy)
  left  → 左转
  right → 右转
"""


class TurnController:
    def decide(self, view, vx=0, vy=0):
        """
        Args:
            view: "front" / "left" / "right"
            vx, vy: 目标像素坐标 (仅 front 有意义)
        Returns:
            (action, vx, vy)
            action: "forward" / "turn_left" / "turn_right"
            vx, vy: 仅 action="forward" 时有效
        """
        if view == "front":
            return "forward", vx, vy
        elif view == "left":
            return "turn_left", None, None
        elif view == "right":
            return "turn_right", None, None

        # fallback
        return "forward", 320, 240
