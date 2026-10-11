#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
pi_host="${PI_HOST:-pi@raspberrypi.local}"
pi_dir="${PI_DIR:-/home/pi/edgeface}"
if [[ ! "$pi_dir" =~ ^/home/[a-zA-Z0-9_-]+/[a-zA-Z0-9_/-]+$ ]]; then
  echo 'PI_DIR must be an absolute simple /home/<user>/<directory> path' >&2; exit 1
fi
ssh "$pi_host" "mkdir -p '$pi_dir'"
# Never overwrite private config, environment, dataset or existing model artifacts.
rsync -az --exclude '.env*' --exclude .venv --exclude .git --exclude .agents --exclude .aws --exclude .codex \
  --exclude dataset --exclude runs --exclude reports/generated --exclude models --exclude mqtt/secrets \
  --exclude '__pycache__' --exclude '.pytest_cache' --exclude '*.egg-info' ./ "$pi_host:$pi_dir/"
ssh "$pi_host" "cd '$pi_dir' && bash scripts/setup_pi.sh"
if [[ -f models/face_detection_yunet_2023mar.onnx ]]; then
  rsync -az models/face_detection_yunet_2023mar.onnx "$pi_host:$pi_dir/models/"
fi
echo 'Code deployed. Use deploy_model.py for validated, versioned model activation.'
