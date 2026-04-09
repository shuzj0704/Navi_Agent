#!/bin/bash
# Wrapper to invoke IsaacSim's Python with the correct environment.
ISAACSIM_ROOT="${ISAACSIM_ROOT:-/home/shu22/nvidia/isaacsim_5.1.0}"
"${ISAACSIM_ROOT}/python.sh" "$@"
