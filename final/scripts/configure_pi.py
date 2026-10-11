from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from dotenv import dotenv_values


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision private Pi config via SSH stdin")
    parser.add_argument("--host", default="pi@raspberrypi.local")
    parser.add_argument("--mqtt-host", default="127.0.0.1", help="WSL Tailscale IPv4; 127.0.0.1 only for legacy SSH tunnel")
    args = parser.parse_args()
    values = dotenv_values(".env")
    password = values.get("MQTT_DEVICE_PASSWORD")
    if not password:
        raise RuntimeError("Run setup_mqtt.py first")
    # Generated URL-safe password; reject control chars to prevent dotenv injection.
    if any(c in password for c in "\r\n'\\"):
        raise ValueError("MQTT device password contains unsupported dotenv characters")
    content = (f"MQTT_HOST={args.mqtt_host}\nMQTT_PORT=1883\nMQTT_USERNAME=edgeface_pi\n"
               f"MQTT_PASSWORD='{password}'\nDEVICE_ID={values.get('DEVICE_ID') or 'raspberrypi-01'}\n"
               "CAMERA_BACKEND=auto\nCAMERA_INDEX=0\nAUDIO_ENABLED=false\nORT_THREADS=2\n")
    subprocess.run(["ssh", args.host, "umask 077; cat > /home/pi/edgeface/.env"],
                   input=content.encode(), check=True)
    print("Pi configuration written privately; no Telegram token copied.")


if __name__ == "__main__":
    main()
