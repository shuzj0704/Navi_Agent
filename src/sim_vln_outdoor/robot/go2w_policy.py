"""Go2W RL locomotion policy: observation construction, inference, torque computation.

Reproduces the rl_sar C++ SDK logic in Python for Isaac Sim deployment.
"""

import os

import numpy as np
import yaml


def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q.

    Args:
        q: quaternion (w, x, y, z), shape (4,)
        v: vector, shape (3,)
    Returns:
        Rotated vector, shape (3,)
    """
    w, x, y, z = q
    q_vec = np.array([x, y, z])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


# Default asset paths
_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
)
DEFAULT_BASE_CFG = os.path.join(_ASSETS_DIR, "policy", "go2w", "base.yaml")
DEFAULT_POLICY_CFG = os.path.join(_ASSETS_DIR, "policy", "go2w", "robot_lab", "config.yaml")
DEFAULT_POLICY_MODEL = os.path.join(_ASSETS_DIR, "policy", "go2w", "robot_lab", "policy.pt")


class Go2WPolicy:
    """RL locomotion policy for Unitree Go2W (wheel-legged robot).

    Loads config from rl_sar format and runs TorchScript policy inference.
    Implements the same observation construction and PD torque computation
    as the rl_sar C++ SDK.
    """

    def __init__(self, base_cfg_path: str = DEFAULT_BASE_CFG,
                 policy_cfg_path: str = DEFAULT_POLICY_CFG,
                 model_path: str = DEFAULT_POLICY_MODEL):
        import torch

        with open(base_cfg_path) as f:
            base_cfg = yaml.safe_load(f)
        with open(policy_cfg_path) as f:
            policy_cfg = yaml.safe_load(f)

        self.base = base_cfg["go2w"]
        self.cfg = policy_cfg["go2w/robot_lab"]

        self.num_dofs = self.cfg["num_of_dofs"]  # 16
        self.num_obs = self.cfg["num_observations"]  # 57
        self.obs_names = self.cfg["observations"]

        # Scaling factors
        self.ang_vel_scale = self.cfg["ang_vel_scale"]
        self.dof_pos_scale = self.cfg["dof_pos_scale"]
        self.dof_vel_scale = self.cfg["dof_vel_scale"]
        self.commands_scale = np.array(self.cfg["commands_scale"], dtype=np.float32)
        self.action_scale = np.array(self.cfg["action_scale"], dtype=np.float32)
        self.clip_obs = self.cfg["clip_obs"]
        self.clip_actions_lower = np.array(self.cfg["clip_actions_lower"], dtype=np.float32)
        self.clip_actions_upper = np.array(self.cfg["clip_actions_upper"], dtype=np.float32)

        # PD gains and torque limits
        self.rl_kp = np.array(self.cfg["rl_kp"], dtype=np.float32)
        self.rl_kd = np.array(self.cfg["rl_kd"], dtype=np.float32)
        self.torque_limits = np.array(self.cfg["torque_limits"], dtype=np.float32)

        # Default joint positions and wheel config
        self.default_dof_pos = np.array(self.cfg["default_dof_pos"], dtype=np.float32)
        self.wheel_indices = self.cfg["wheel_indices"]  # [12, 13, 14, 15]

        # Joint ordering from base config
        self.joint_names = self.base["joint_names"]

        # Simulation timing
        self.dt = self.base["dt"]  # 0.005
        self.decimation = self.base["decimation"]  # 4

        # Runtime state
        self.prev_actions = np.zeros(self.num_dofs, dtype=np.float32)
        self.commands = np.zeros(3, dtype=np.float32)

        # Load TorchScript model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print(f"[Info] Policy loaded on {self.device}: {model_path}")

    def set_commands(self, vx: float, vy: float, vyaw: float):
        self.commands[:] = [vx, vy, vyaw]

    def compute_observation(self, ang_vel, base_quat, dof_pos, dof_vel):
        """Build the 57-dim observation vector from robot state.

        Observation layout: [ang_vel(3), gravity_vec(3), commands(3),
                             dof_pos(16), dof_vel(16), actions(16)]
        """
        parts = []
        for name in self.obs_names:
            if name == "ang_vel":
                parts.append(ang_vel * self.ang_vel_scale)
            elif name == "gravity_vec":
                gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
                parts.append(quat_rotate_inverse(base_quat, gravity_world))
            elif name == "commands":
                parts.append(self.commands * self.commands_scale)
            elif name == "dof_pos":
                # Relative joint positions (wheel positions zeroed)
                rel = (dof_pos - self.default_dof_pos).copy()
                for i in self.wheel_indices:
                    rel[i] = 0.0
                parts.append(rel * self.dof_pos_scale)
            elif name == "dof_vel":
                parts.append(dof_vel * self.dof_vel_scale)
            elif name == "actions":
                parts.append(self.prev_actions.copy())

        obs = np.concatenate(parts).astype(np.float32)
        return np.clip(obs, -self.clip_obs, self.clip_obs)

    def infer(self, obs):
        """Run policy forward pass and return clipped actions."""
        import torch

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            actions = self.model(obs_t).cpu().numpy().flatten()
        actions = np.clip(actions, self.clip_actions_lower, self.clip_actions_upper)
        self.prev_actions = actions.copy()
        return actions

    def compute_torques(self, actions, dof_pos, dof_vel):
        """Compute PD control torques from policy actions.

        For leg joints (0-11):  tau = kp * (action_scaled + default - pos) - kd * vel
        For wheel joints (12-15): tau = -kd * (vel - action_scaled)  (kp=0)
        """
        scaled = actions * self.action_scale

        pos_target = scaled.copy()
        vel_target = np.zeros_like(scaled)
        for i in self.wheel_indices:
            vel_target[i] = scaled[i]
            pos_target[i] = 0.0

        target_pos = pos_target + self.default_dof_pos
        torques = self.rl_kp * (target_pos - dof_pos) - self.rl_kd * (dof_vel - vel_target)
        return np.clip(torques, -self.torque_limits, self.torque_limits)
