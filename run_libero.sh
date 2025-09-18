#!/bin/bash
set -euo pipefail

PORT="${1:-8000}"

# --- shared env ---
source /data/scratch/shenli/miniconda3/etc/profile.d/conda.sh
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OPENPI_DATA_HOME=/data/scratch/shenli/openpi/.cache
export LIBERO_CONFIG_PATH=/data/scratch/shenli/openpi/libero_cfg
export PYTHONPATH=/data/scratch/shenli/openpi:/data/scratch/shenli/openpi/packages/openpi-client/src:/data/scratch/shenli/openpi/third_party/libero
cd /data/scratch/shenli/openpi

# clean any stale server on this port
if ss -ltn 'sport = :'$PORT | grep -q ":$PORT"; then
  echo "[cleanup] port $PORT busy; killing old serve_policy.py..."
  pkill -f "scripts/serve_policy.py" || true
  sleep 1
fi

# 1) start SERVER in background
conda activate /data/scratch/shenli/envs/pi-srv
python scripts/serve_policy.py --env LIBERO --libero-model-type BASE --port "$PORT" &
SERVER_PID=$!

# wait for server (raw TCP probe; may log a harmless handshake error)
echo "[wait] waiting for server on localhost:$PORT ..."
for i in $(seq 1 1200); do
  if python - <<PY 2>/dev/null; then
import socket; s=socket.socket(); s.settimeout(0.5)
s.connect(("127.0.0.1",$PORT)); s.close()
PY
    echo "[wait] server is up!"
    break
  fi
  if (( i % 10 == 0 )); then echo "[wait] still waiting... ($((i/10))0s)"; fi
  sleep 1
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[error] server exited early"; exit 1
  fi
done

# 2) run CLIENT in foreground
conda activate /data/scratch/shenli/envs/libero
python examples/libero/main.py \
  --args.task-suite-name libero_10 \
  --args.num-trials-per-task 1 \
  --args.host localhost \
  --args.port "$PORT"

# 3) cleanup
echo "[info] client finished; stopping server..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
echo "[done]"
