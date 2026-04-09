"""Navigation controller interface for closed-loop evaluation.

Defines Observation, Action, and the NavController abstract base class.
Users implement NavController.act() to plug in their own navigation policy.
"""

import abc
import dataclasses

import numpy as np


@dataclasses.dataclass
class Observation:
    """Observation delivered to the controller each step.

    Attributes:
        rgb: Camera image, shape (480, 640, 3), dtype uint8.
        pose: Current camera pose (x, y, z, roll, pitch, yaw).
              Position in meters, angles in degrees.
        step: Current controller step index (0-based).
    """

    rgb: np.ndarray
    pose: tuple
    step: int


@dataclasses.dataclass
class Action:
    """Navigation action output by the controller.

    Sign conventions match load_scene_view.py keyboard controls (Z-up RHS):
        forward  > 0 : move along yaw direction  (W key)
        strafe   > 0 : move left                 (A key)
        vertical > 0 : move up                   (Q key)
        pitch    > 0 : tilt up                    (UP arrow)
        yaw      > 0 : turn left                 (LEFT arrow)
    """

    forward: float = 0.0
    strafe: float = 0.0
    vertical: float = 0.0
    pitch: float = 0.0      # deg/step
    yaw: float = 0.0        # deg/step
    done: bool = False


class NavController(abc.ABC):
    """Abstract base class for navigation controllers."""

    @abc.abstractmethod
    def act(self, obs: Observation) -> Action:
        """Given an observation, produce a navigation action."""
        ...

    def reset(self) -> None:
        """Called at the start of each episode. Override for stateful controllers."""
        pass

    def on_episode_end(self, trajectory: list) -> None:
        """Called when episode finishes. Override for logging/metrics."""
        pass
