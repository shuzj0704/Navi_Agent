"""Demo navigation controllers for testing the evaluation loop."""

import numpy as np

from .controller import Action, NavController, Observation


class ForwardOnlyController(NavController):
    """Walk forward at constant speed."""

    def __init__(self, speed: float = 0.5):
        self.speed = speed

    def act(self, obs: Observation) -> Action:
        return Action(forward=self.speed)


class RandomWalkController(NavController):
    """Random movement and rotation each step."""

    def __init__(self, speed: float = 0.5, rot_speed: float = 5.0, seed: int = 42):
        self.speed = speed
        self.rot_speed = rot_speed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.rng.integers(0, 2**31))

    def act(self, obs: Observation) -> Action:
        return Action(
            forward=self.rng.uniform(-self.speed, self.speed),
            strafe=self.rng.uniform(-self.speed, self.speed),
            yaw=self.rng.uniform(-self.rot_speed, self.rot_speed),
        )
