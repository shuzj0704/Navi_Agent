#!/bin/bash
# Wrapper to invoke IsaacSim's Python with the correct environment.
ISAACSIM_ROOT="${ISAACSIM_ROOT:-/home/ps/sources/isaacsim_4.5.0}"
"${ISAACSIM_ROOT}/python.sh" "$@"
