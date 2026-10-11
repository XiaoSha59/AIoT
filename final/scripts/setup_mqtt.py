"""Create per-role random credentials locally and hash them for Mosquitto.

Passwords travel via stdin only, never command arguments or logs. Requires docker.
"""
from __future__ import annotations

import os
import secrets
import subprocess
from pathlib import Path

from dotenv import dotenv_values, set_key


def main() -> None:
    os.umask(0o077)
    path = Path(".env")
    if not path.exists():
        path.write_text(Path(".env.example").read_text())
    values = dotenv_values(path)
    bot_password = values.get("MQTT_PASSWORD") or secrets.token_urlsafe(32)
    pi_password = values.get("MQTT_DEVICE_PASSWORD") or secrets.token_urlsafe(32)
    monitor_password = values.get("MQTT_MONITOR_PASSWORD") or secrets.token_urlsafe(32)
    set_key(str(path), "MQTT_PASSWORD", bot_password)
    set_key(str(path), "MQTT_DEVICE_PASSWORD", pi_password)
    set_key(str(path), "MQTT_MONITOR_PASSWORD", monitor_password)
    if not values.get("MQTT_EXPLORER_USERNAME"):
        set_key(str(path), "MQTT_EXPLORER_USERNAME", "admin")
    if not values.get("MQTT_EXPLORER_PASSWORD"):
        set_key(str(path), "MQTT_EXPLORER_PASSWORD", secrets.token_urlsafe(24))
    if not values.get("MQTT_USERNAME"):
        set_key(str(path), "MQTT_USERNAME", "edgeface_bot")
    path.chmod(0o600)
    target = Path("mqtt/secrets")
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    # mosquitto_passwd -U hashes a plaintext file. File exists only in container tmpfs.
    script = "cat > /tmp/edgeface-passwd && mosquitto_passwd -U /tmp/edgeface-passwd && cat /tmp/edgeface-passwd"
    result = subprocess.run(["docker", "run", "--rm", "-i", "--tmpfs", "/tmp", "--entrypoint", "sh",
                             "eclipse-mosquitto:2.0.22", "-c", script],
                            input=f"edgeface_bot:{bot_password}\nedgeface_pi:{pi_password}\nedgeface_monitor:{monitor_password}\n".encode(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError("Mosquitto credential setup failed; check Docker/image availability")
    if b"edgeface_bot:$" not in result.stdout or b"edgeface_pi:$" not in result.stdout:
        raise RuntimeError("Password hashing failed")
    (target / "passwd").write_bytes(result.stdout)
    # Hashed password file is read-only in container; no plaintext saved here.
    (target / "passwd").chmod(0o644)
    print("MQTT credentials configured in private .env; hashed broker file created.")


if __name__ == "__main__":
    main()
