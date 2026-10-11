#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
pi_host="${PI_HOST:-pi@raspberrypi.local}"
pi_dir="${PI_DIR:-/home/pi/edgeface}"
mkdir -p dataset/raw
chmod 700 dataset dataset/raw
rsync -az --chmod=D700,F600 "$pi_host:$pi_dir/dataset/raw/" dataset/raw/
echo 'Dataset synchronized locally. No deletion performed on either device.'
