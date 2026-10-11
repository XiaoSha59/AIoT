"""Move Pi MQTT to native Tailscale, preserving config and validating a real ack."""
from __future__ import annotations

import argparse
import ipaddress
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi-host", required=True, help="Existing trusted SSH endpoint")
    parser.add_argument("--host-key-alias", help="Reuse a previously verified SSH host key")
    parser.add_argument("--known-hosts", type=Path)
    args = parser.parse_args()
    if args.pi_host.startswith("-") or any(c.isspace() for c in args.pi_host):
        raise ValueError("Invalid SSH endpoint")
    state = json.loads(subprocess.check_output(["tailscale", "status", "--json"], text=True))
    if state.get("BackendState") != "Running":
        raise RuntimeError("Sign the WSL node in to Tailscale first")
    addresses = state["Self"]["TailscaleIPs"]
    mqtt_host = next(address for address in addresses if ipaddress.ip_address(address).version == 4)
    if ipaddress.ip_address(mqtt_host) not in ipaddress.ip_network("100.64.0.0/10"):
        raise ValueError("Expected a Tailscale IPv4 address")
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           "-o", "StrictHostKeyChecking=yes"]
    if args.host_key_alias:
        ssh += ["-o", f"HostKeyAlias={args.host_key_alias}"]
    if args.known_hosts:
        ssh += ["-o", f"UserKnownHostsFile={args.known_hosts}"]
    ssh.append(args.pi_host)
    pi_ip = subprocess.check_output(ssh+["tailscale ip -4"], text=True).strip()
    if ipaddress.ip_address(pi_ip) not in ipaddress.ip_network("100.64.0.0/10"):
        raise ValueError("Pi is not signed in to Tailscale")
    peers = (state.get("Peer") or {}).values()
    if not any(pi_ip in p.get("TailscaleIPs", []) for p in peers):
        raise RuntimeError("Pi is not visible in the WSL tailnet")
    subprocess.run(["tailscale", "serve", "--bg", "--tcp=1883", "tcp://127.0.0.1:1883"], check=True)
    # Update only the endpoint. MQTT credentials and all camera/model settings stay local.
    remote = """import os, re, socket
from pathlib import Path
host = HOST_VALUE
with socket.create_connection((host, 1883), timeout=10):
    pass
path = Path('/home/pi/edgeface/.env')
original = path.read_text()
backup = path.with_name('.env.before-tailscale')
os.umask(0o077)
backup.write_text(original)
backup.chmod(0o600)
updated = re.sub(r'^MQTT_HOST=.*$', 'MQTT_HOST='+host, original, flags=re.MULTILINE)
if updated == original and not re.search(r'^MQTT_HOST=', original, re.MULTILINE):
    updated = original.rstrip()+'\\nMQTT_HOST='+host+'\\n'
temporary = path.with_name('.env.tailscale-pending')
temporary.write_text(updated)
temporary.chmod(0o600)
temporary.replace(path)
print('Pi MQTT endpoint updated; private credentials preserved.')
""".replace("HOST_VALUE", repr(mqtt_host))
    subprocess.run(ssh+["python3 -"], input=remote, text=True, check=True)
    try:
        subprocess.run(ssh+["sudo -n systemctl restart edgeface.service"], check=True)
        # systemctl restart returns before Python connects/subscribes. Commands are
        # intentionally not retained, so retry fresh read-only status requests.
        for attempt in range(3):
            check = subprocess.run([sys.executable, "-m", "scripts.mqtt_command", "status"],
                                   capture_output=True, text=True, timeout=35)
            if check.returncode == 0:
                break
            if attempt == 2:
                raise RuntimeError("No MQTT status acknowledgement after three fresh requests")
            time.sleep(1)
        ack = json.loads(check.stdout)
        if not ack.get("online") or not ack.get("ok"):
            raise RuntimeError("Pi did not confirm MQTT status")
    except Exception:
        subprocess.run(ssh+["cp /home/pi/edgeface/.env.before-tailscale /home/pi/edgeface/.env "
                            "&& chmod 600 /home/pi/edgeface/.env "
                            "&& sudo -n systemctl restart edgeface.service"], check=True)
        raise
    # Only retire the old route after a fresh request_id has been acknowledged.
    unit = "edgeface-mqtt-tunnel.service"
    if subprocess.run(["systemctl", "--user", "is-enabled", "--quiet", unit]).returncode == 0:
        subprocess.run(["systemctl", "--user", "disable", "--now", unit], check=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "transport": "tailscale",
              "wsl_ip": mqtt_host, "pi_ip": pi_ip, "port": 1883,
              "status_ack": ack, "ssh_tunnel_active": subprocess.run(
                  ["systemctl", "--user", "is-active", "--quiet", unit]).returncode == 0}
    output = Path("reports/generated/tailscale_network.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
