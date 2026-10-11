#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
app_dir="$PWD"
app_user="$(id -un)"
[[ -f .env ]] || { echo 'Missing .env' >&2; exit 1; }
chmod 600 .env
unit_path="$(mktemp)"
cat > "$unit_path" <<EOF
[Unit]
Description=EdgeFace local camera ONNX inference and MQTT
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
User=$app_user
SupplementaryGroups=video audio
WorkingDirectory=$app_dir
ExecStart=$app_dir/.venv/bin/python -m raspberry_pi.agent
Restart=on-failure
RestartSec=5
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
TimeoutStopSec=20
[Install]
WantedBy=multi-user.target
EOF
sudo install -m 0644 "$unit_path" /etc/systemd/system/edgeface.service
sudo systemctl daemon-reload
sudo systemctl enable --now edgeface.service
sudo systemctl status --no-pager edgeface.service
