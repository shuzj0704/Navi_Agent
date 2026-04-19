#!/usr/bin/env bash
# 一键安装 Navi_Agent habitat 环境
# Usage:  bash install.sh [env_name]
set -euo pipefail

ENV_NAME="${1:-lwy_habitat}"
PY_VER="3.9"

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERR] conda not found. Please install Miniconda/Anaconda first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[INFO] conda env '${ENV_NAME}' already exists, reusing."
else
    echo "[1/3] Creating conda env '${ENV_NAME}' (python=${PY_VER})..."
    conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

conda activate "${ENV_NAME}"

echo "[2/3] Installing habitat-sim via conda..."
conda install -y -n "${ENV_NAME}" \
    habitat-sim=0.3.3 withbullet headless \
    -c conda-forge -c aihabitat

echo "[3/3] pip install -r requirements.txt ..."
pip install -r "$(dirname "$0")/requirements.txt"

echo ""
echo "[DONE] Environment '${ENV_NAME}' ready."
echo "       Activate:  conda activate ${ENV_NAME}"
