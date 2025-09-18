#!/bin/bash
set -euo pipefail

PORT="${1:-8000}"

source /data/scratch/shenli/miniconda3/etc/profile.d/conda.sh
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OPENPI_DATA_HOME=/data/scratch/shenli/openpi/.cache
export LIBERO_CONFIG_PATH=/data/scratch/shenli/openpi/libero_cfg
export PYTHONPATH=/data/scratch/shenli/openpi:/data/scratch/shenli/openpi/packages/openpi-client/src:/data/scratch/shenli/openpi/third_party/libero
cd /data/scratch/shenli/openpi

conda activate /data/scratch/shenli/envs/libero
python examples/libero/print_task_descriptions.py --args.task-suite-name libero_10 --args.num-trials-per-task 1 --args.host localhost --args.port "$PORT"
