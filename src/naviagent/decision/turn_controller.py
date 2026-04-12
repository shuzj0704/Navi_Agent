"""
转向决策控制器
==============
根据 VLM 输出的 view 决定机器人行为:
  front → 交给 DWA (vx, vy)
  left  → 左转
  right → 右转
  back  → 迟滞转向 (防止左右振荡)
"""


class TurnController:
    def __init__(self, back_turn_threshold=3):
        """
        Args:
            back_turn_threshold: back 视角连续转向多少次才允许切换方向
        """
        self.back_turn_direction = None
        self.back_turn_count = 0
        self.threshold = back_turn_threshold

    def decide(self, view, vx=0, vy=0):
        """
        Args:
            view: "front" / "left" / "right" / "back"
            vx, vy: 目标像素坐标 (仅 front 有意义)
        Returns:
            (action, vx, vy)
            action: "forward" / "turn_left" / "turn_right"
            vx, vy: 仅 action="forward" 时有效
        """
        if view == "front":
            self._reset_back()
            return "forward", vx, vy

        elif view == "left":
            self._reset_back()
            return "turn_left", None, None

        elif view == "right":
            self._reset_back()
            return "turn_right", None, None

        elif view == "back":
            # 迟滞: 一旦选了方向就坚持转够次数
            if self.back_turn_direction is None:
                self.back_turn_direction = "left"
                self.back_turn_count = 1
            else:
                self.back_turn_count += 1

            action = "turn_left" if self.back_turn_direction == "left" else "turn_right"
            return action, None, None

        # fallback
        return "forward", 320, 240

    def _reset_back(self):
        self.back_turn_direction = None
        self.back_turn_count = 0
