"""Launch Isaac Sim and load a USD scene from CraftBench."""

import argparse
import os
import sys

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
    parser = argparse.ArgumentParser(description="Load a CraftBench USD scene in Isaac Sim.")
    parser.add_argument(
        "--usd-path", type=str, default=DEFAULT_USD,
        help="Path to the USD scene file to load.",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run in headless mode (no GUI).",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device index for rendering (default: 0).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from env import IsaacSimEnv

    env = IsaacSimEnv(usd_path=args.usd_path, headless=args.headless,
                      gpu_id=args.gpu)

    print("[Info] Running simulation loop (Ctrl+C to exit)...")
    try:
        while env.is_running:
            env.step()
    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")

    env.close()


if __name__ == "__main__":
    main()
