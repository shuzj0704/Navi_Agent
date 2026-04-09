"""Go2W robot: URDF import, articulation setup, state read/write for Isaac Sim."""

import os
import sys

import numpy as np

from .go2w_policy import Go2WPolicy, quat_rotate_inverse


_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
)
DEFAULT_URDF = os.path.join(
    _ASSETS_DIR, "rl_sar_zoo", "go2w_description", "urdf", "go2w_description.urdf",
)


class Go2WRobot:
    """Manages Go2W URDF import, articulation, and policy-driven control."""

    PRIM_PATH = "/go2w_description"

    def __init__(self, env, policy: Go2WPolicy,
                 spawn_pos: list[float] = (-730.0, 490.0, 0.0),
                 urdf_path: str = DEFAULT_URDF):
        self.env = env
        self.policy = policy

        urdf_path = os.path.abspath(urdf_path)
        if not os.path.isfile(urdf_path):
            print(f"[Error] URDF file not found: {urdf_path}")
            sys.exit(1)

        self._import_urdf(urdf_path)
        self._init_articulation(spawn_pos)

    def _import_urdf(self, urdf_path: str):
        """Import Go2W URDF into the current stage."""
        import omni.kit.commands
        import omni.usd

        stage = self.env.stage
        stale = stage.GetPrimAtPath(self.PRIM_PATH)
        if stale and stale.IsValid():
            stage.RemovePrim(self.PRIM_PATH)
            print(f"[Info] Removed existing {self.PRIM_PATH} prim from scene")

        _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False
        import_config.fix_base = False
        import_config.make_default_prim = False
        import_config.create_physics_scene = False

        omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=import_config,
        )

        if not self.env.stage.GetPrimAtPath(self.PRIM_PATH).IsValid():
            print(f"[Error] Robot prim not found at {self.PRIM_PATH}")
            self.env.close()
            sys.exit(1)

        print(f"[Info] Robot imported at {self.PRIM_PATH}")

    def _init_articulation(self, spawn_pos):
        """Initialize articulation, joint mapping, and default pose."""
        from isaacsim.core.prims import SingleArticulation

        self.env.world.reset()

        self.articulation = SingleArticulation(
            prim_path=self.PRIM_PATH, name="go2w",
        )
        self.articulation.initialize()

        self.num_dofs = self.articulation.num_dof
        print(f"[Info] Articulation DOFs: {self.num_dofs}, "
              f"names: {self.articulation.dof_names}")

        # Build joint index mapping: policy order -> sim DOF order
        self.joint_index_map = np.array([
            self.articulation.get_dof_index(name)
            for name in self.policy.joint_names
        ])
        print(f"[Info] Joint index mapping (policy->sim): {self.joint_index_map}")

        # Set joint drives to effort mode (kp=0, kd=0)
        self.articulation._articulation_view.set_gains(
            kps=np.zeros((1, self.num_dofs), dtype=np.float32),
            kds=np.zeros((1, self.num_dofs), dtype=np.float32),
        )
        self.articulation._articulation_view.set_max_efforts(
            values=np.full((1, self.num_dofs), 23.5, dtype=np.float32),
        )
        print("[Info] Joint drives set to effort mode (kp=0, kd=0)")

        # Set initial pose
        pos = np.array(spawn_pos, dtype=np.float64)
        self.articulation.set_world_pose(
            position=pos, orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )

        init_joint_pos = np.zeros(self.num_dofs, dtype=np.float32)
        for i, sim_idx in enumerate(self.joint_index_map):
            init_joint_pos[sim_idx] = self.policy.default_dof_pos[i]
        self.articulation.set_joint_positions(init_joint_pos)
        self.articulation.set_joint_velocities(
            np.zeros(self.num_dofs, dtype=np.float32),
        )
        print("[Info] Robot initialized at default standing pose")

    def get_state(self):
        """Read robot state and return in policy joint order.

        Returns:
            tuple: (ang_vel_body, base_quat, dof_pos, dof_vel), all float32
        """
        _, base_quat = self.articulation.get_world_pose()  # (4,) w,x,y,z
        ang_vel_world = self.articulation.get_angular_velocity()
        ang_vel_body = quat_rotate_inverse(base_quat, ang_vel_world)

        sim_pos = self.articulation.get_joint_positions()
        sim_vel = self.articulation.get_joint_velocities()

        # Reorder to policy joint order
        dof_pos = sim_pos[self.joint_index_map].astype(np.float32)
        dof_vel = sim_vel[self.joint_index_map].astype(np.float32)

        return (
            ang_vel_body.astype(np.float32),
            base_quat.astype(np.float32),
            dof_pos,
            dof_vel,
        )

    def apply_torques(self, torques_policy):
        """Map policy-order torques to sim order and apply."""
        torques_sim = np.zeros(self.num_dofs, dtype=np.float32)
        for i, sim_idx in enumerate(self.joint_index_map):
            torques_sim[sim_idx] = torques_policy[i]
        self.articulation.set_joint_efforts(torques_sim)

    def get_position(self):
        """Return world position (3,)."""
        pos, _ = self.articulation.get_world_pose()
        return pos
