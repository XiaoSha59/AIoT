#!/usr/bin/env bash
# Native installation on the supported WSL/Pi hosts. Never reads account secrets.
set -euo pipefail
node_name="${1:-edgeface-wsl}"
node_operator="${2:-${SUDO_USER:-$(id -un)}}"
[[ "$node_name" =~ ^[a-zA-Z0-9-]+$ ]] || { echo 'Invalid hostname' >&2; exit 2; }
[[ "$node_operator" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid operator' >&2; exit 2; }
source /etc/os-release
case "${ID}:${VERSION_CODENAME:-}" in
  ubuntu:noble|ubuntu:jammy|debian:bookworm|debian:trixie) ;;
  *) echo "Unsupported distribution: ${ID} ${VERSION_ID}" >&2; exit 2 ;;
esac
[[ -c /dev/net/tun ]] || { echo '/dev/net/tun is missing' >&2; exit 1; }
privilege=()
if [[ "$EUID" -ne 0 ]]; then
  sudo -v # Enter the password in your terminal, never in chat or a config file.
  privilege=(sudo)
fi
if ! command -v tailscale >/dev/null; then
  work_dir="$(mktemp -d)"
  trap 'rm -rf -- "$work_dir"' EXIT
  curl -fsSL --retry 2 --connect-timeout 10 --max-time 90 \
    "https://pkgs.tailscale.com/stable/${ID}/${VERSION_CODENAME}.noarmor.gpg" \
    -o "$work_dir/tailscale.gpg"
  curl -fsSL --retry 2 --connect-timeout 10 --max-time 90 \
    "https://pkgs.tailscale.com/stable/${ID}/${VERSION_CODENAME}.tailscale-keyring.list" \
    -o "$work_dir/tailscale.list"
  "${privilege[@]}" install -m 0644 "$work_dir/tailscale.gpg" /usr/share/keyrings/tailscale-archive-keyring.gpg
  "${privilege[@]}" install -m 0644 "$work_dir/tailscale.list" /etc/apt/sources.list.d/tailscale.list
  "${privilege[@]}" apt-get update \
    -o Dir::Etc::sourcelist=sources.list.d/tailscale.list \
    -o Dir::Etc::sourceparts=- -o APT::Get::List-Cleanup=0
  "${privilege[@]}" apt-get install -y --no-install-recommends tailscale
fi
"${privilege[@]}" mkdir -p /etc/systemd/system/tailscaled.service.d
printf '[Service]\nRestart=always\nRestartSec=5\n' \
  | "${privilege[@]}" tee /etc/systemd/system/tailscaled.service.d/edgeface.conf >/dev/null
"${privilege[@]}" systemctl daemon-reload
"${privilege[@]}" systemctl enable --now tailscaled
"${privilege[@]}" tailscale set --operator="$node_operator"
tailscale version
"${privilege[@]}" systemctl is-enabled tailscaled
"${privilege[@]}" systemctl is-active tailscaled
printf '\nInstall complete. Sign in with your existing Tailscale account:\n'
printf 'tailscale up --hostname=%s --accept-dns=false --accept-routes=false\n' "$node_name"
