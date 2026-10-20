"""Enable the private Pi snapshot endpoint and Telegram photo notifications."""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import secrets
import shlex
import subprocess
from pathlib import Path

from dotenv import dotenv_values, set_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="SSH destination for the Pi")
    parser.add_argument("--directory", default="/home/pi/edgeface")
    args = parser.parse_args()
    os.umask(0o077)
    remote_ip = subprocess.run(["ssh", args.host, "tailscale ip -4"], check=True,
                               text=True, capture_output=True).stdout.strip()
    if ipaddress.ip_address(remote_ip) not in ipaddress.ip_network("100.64.0.0/10"):
        raise ValueError("Pi must have a Tailscale IPv4 address")
    env = Path(".env")
    if not env.is_file():
        raise ValueError("Create the local private .env first")
    values = dotenv_values(env)
    token = values.get("SNAPSHOT_TOKEN") or secrets.token_urlsafe(32)
    if len(token) < 32:
        raise ValueError("Existing SNAPSHOT_TOKEN must have at least 32 characters")
    updates = {"RECOGNITION_PHOTO_ENABLED": "true", "SNAPSHOT_TOKEN": token,
               "SNAPSHOT_BASE_URL": f"http://{remote_ip}:8765", "SNAPSHOT_BIND": remote_ip,
               "SNAPSHOT_PORT": "8765", "SNAPSHOT_TTL_S": "120", "SNAPSHOT_TIMEOUT_S": "5"}
    remote_code = ("import json,sys; from pathlib import Path; from dotenv import set_key; "
                   f"p=Path({str(Path(args.directory)/'.env')!r}); "
                   "assert p.is_file(), 'Pi .env missing'; "
                   "data=json.load(sys.stdin); "
                   "[set_key(str(p),k,v) for k,v in data.items()]; p.chmod(0o600)")
    python = shlex.quote(str(Path(args.directory)/".venv/bin/python"))
    subprocess.run(["ssh", args.host, python + " -c " + shlex.quote(remote_code)],
                   input=json.dumps(updates), text=True, check=True)
    for key, value in updates.items():
        set_key(str(env), key, value)
    env.chmod(0o600)
    print("Photo notifications configured privately; token transferred via SSH stdin.")
    print(f"Pi HTTP endpoint binds Tailscale only: {remote_ip}:8765")
    print("Restart the Pi agent and recreate the Telegram container to apply.")


if __name__ == "__main__":
    main()
