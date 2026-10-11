#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo 'This pinned ONNX Runtime setup requires aarch64. Do not change/reinstall the OS automatically.' >&2
  exit 1
fi
/usr/bin/python3 -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10–3.12 for the pinned ARM64 wheel"'
if [[ ! -x .venv/bin/python ]]; then
  /usr/bin/python3 -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install --no-cache-dir '.[edge]'
.venv/bin/python -c 'import cv2, onnxruntime as ort; print("OpenCV", cv2.__version__, "ORT", ort.__version__, "providers", ort.get_available_providers())'
mkdir -p dataset models reports/generated
chmod 700 dataset
echo 'Native Pi dependencies ready. Configure private .env before starting agent.'
