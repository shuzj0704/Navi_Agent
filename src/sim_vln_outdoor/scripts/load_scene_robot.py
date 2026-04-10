"""Launch Isaac Sim, load a USD scene, spawn Go2W robot, and run RL locomotion policy.

Usage:
    cd /home/shu22/nvidia/isaacsim_5.1.0
    ./python.sh /home/shu22/navigation/Navi_Agent/src/sim_vln_outdoor/scripts/load_scene_robot.py
    ./python.sh /home/shu22/navigation/Navi_Agent/src/sim_vln_outdoor/scripts/load_scene_robot.py --headless
    ./python.sh /home/shu22/navigation/Navi_Agent/src/sim_vln_outdoor/scripts/load_scene_robot.py --cmd-vel 1.0 0.0 0.0
"""

import argparse
import os
import sys

# Add package root to path so `env` / `robot` are importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_SCRIPT_DIR)  # sim_vln_outdoor/
sys.path.insert(0, _PACKAGE_ROOT)

_CRAFTBENCH_ROOT = os.environ.get(
    "CRAFTBENCH_ROOT",
    os.path.expanduser("~/navigation/urban_verse/CraftBench"),
)
DEFAULT_USD = os.path.join(
    _CRAFTBENCH_ROOT,
    "scene_09_cbd_t_intersection_construction_sites",
    "Collected_export_version",
    "export_version.usd",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load CraftBench scene with Go2W robot running RL locomotion policy."
    )
    parser.add_argument(
        "--usd-path", type=str, default=DEFAULT_USD, help="Path to USD scene file."
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless mode (no GUI)."
    )
    parser.add_argument(
        "--spawn-pos", type=float, nargs=3, default=[-730.0, 490.0, 0.0],
        help="Robot spawn position [x, y, z] in meters.",
    )
    parser.add_argument(
        "--cmd-vel", type=float, nargs=3, default=[0.5, 0.0, 0.0],
        help="Velocity commands [vx, vy, vyaw].",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index for rendering (default: 0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Step 1: Create env (must happen before any omni imports) ---
    from env import IsaacSimEnv

    policy_dt = 0.005
    policy_decimation = 4
    env = IsaacSimEnv(
        usd_path=args.usd_path,
        headless=args.headless,
        physics_dt=policy_dt,
        rendering_dt=policy_dt * policy_decimation,
        gpu_id=args.gpu,
    )

    # --- Step 2: Create policy and robot (omni imports are now safe) ---
    from robot import Go2WPolicy, Go2WRobot

    policy = Go2WPolicy()
    policy.set_commands(*args.cmd_vel)

    robot = Go2WRobot(env, policy, spawn_pos=args.spawn_pos)

    print(f"[Info] Spawn: {args.spawn_pos}")
    print(f"[Info] Commands: vx={args.cmd_vel[0]:.2f}, "
          f"vy={args.cmd_vel[1]:.2f}, vyaw={args.cmd_vel[2]:.2f}")
    print("[Info] Running simulation (Ctrl+C to exit)...")

    # --- Step 3: Simulation loop ---
    step_count = 0
    try:
        while env.is_running:
            env.step()
            step_count += 1

            ang_vel, quat, dof_pos, dof_vel = robot.get_state()
            obs = policy.compute_observation(ang_vel, quat, dof_pos, dof_vel)
            actions = policy.infer(obs)
            torques = policy.compute_torques(actions, dof_pos, dof_vel)
            robot.apply_torques(torques)

            if step_count % 250 == 0:
                pos = robot.get_position()
                print(
                    f"[Step {step_count}] pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                    f" cmd=({policy.commands[0]:.2f}, {policy.commands[1]:.2f}, "
                    f"{policy.commands[2]:.2f})"
                )

    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")

    env.close()


if __name__ == "__main__":
    main()
