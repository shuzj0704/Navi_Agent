"""Isaac Sim environment: load USD scene and manage simulation lifecycle.

IMPORTANT: SimulationApp must be created before any omni.* imports.
Always instantiate IsaacSimEnv before importing robot modules.
"""

import os
import sys


class IsaacSimEnv:
    """Wraps Isaac Sim application, USD scene loading, and World stepping."""

    def __init__(self, usd_path: str, headless: bool = False,
                 physics_dt: float = 0.005, rendering_dt: float = 0.02,
                 gpu_id: int = 0):
        usd_path = os.path.abspath(usd_path)
        if not os.path.isfile(usd_path):
            print(f"[Error] USD file not found: {usd_path}")
            sys.exit(1)

        print(f"[Info] Loading scene: {usd_path}")

        # SimulationApp must be created before any other omniverse imports
        from isaacsim import SimulationApp
        self.app = SimulationApp({
            "headless": headless,
            "multi_gpu": False,
            "active_gpu": gpu_id,
        })

        import omni.usd
        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import open_stage

        open_stage(usd_path)

        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
        )
        self.world.reset()
        print("[Info] Scene loaded.")

    @property
    def stage(self):
        import omni.usd
        return omni.usd.get_context().get_stage()

    def step(self):
        """Advance one rendering step (= decimation x physics steps)."""
        self.world.step(render=True)

    @property
    def is_running(self) -> bool:
        return self.app.is_running()

    def close(self):
        self.app.close()
        print("[Info] Simulation closed.")
