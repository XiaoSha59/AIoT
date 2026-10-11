"""Run on Pi: retain its current IPv4 as static; original DHCP profile is preserved.

An automatic 180-second rollback must be cancelled after a NEW SSH connection works.
"""
from __future__ import annotations

import argparse
import json
import subprocess


def run(*args: str) -> str:
    return subprocess.check_output(list(args), text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default="wlan0")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if args.confirm:
        active = run("nmcli", "-g", "GENERAL.CONNECTION", "device", "show", args.interface)
        if active != "edgeface-static":
            raise RuntimeError("Static profile not active; do not cancel rollback")
        run("sudo", "systemctl", "stop", "edgeface-network-rollback.timer")
        print("Static profile verified over a new SSH connection; rollback timer cancelled.")
        return
    active = run("nmcli", "-t", "-f", "UUID,DEVICE", "connection", "show", "--active")
    old_uuid = next(row.split(":")[0] for row in active.splitlines() if row.endswith(":" + args.interface))
    if run("nmcli", "-g", "ipv4.method", "connection", "show", "uuid", old_uuid) != "auto":
        raise RuntimeError("Active profile is not DHCP; inspect before changing it")
    address = run("nmcli", "-g", "IP4.ADDRESS", "device", "show", args.interface).splitlines()[0]
    gateway = run("nmcli", "-g", "IP4.GATEWAY", "device", "show", args.interface).splitlines()[0]
    dns = run("nmcli", "-g", "IP4.DNS", "device", "show", args.interface).splitlines()
    plan = {"interface": args.interface, "ipv4": address, "gateway": gateway,
            "dns": dns, "dhcp_profile_uuid": old_uuid, "profile": "edgeface-static"}
    print(json.dumps(plan), flush=True)
    if not args.apply:
        return
    profiles = run("nmcli", "-g", "NAME", "connection", "show").splitlines()
    if "edgeface-static" in profiles:
        raise RuntimeError("edgeface-static already exists; refusing to overwrite network config")
    run("sudo", "nmcli", "connection", "clone", "uuid", old_uuid, "edgeface-static")
    run("sudo", "nmcli", "connection", "modify", "edgeface-static", "ipv4.method", "manual",
        "ipv4.addresses", address, "ipv4.gateway", gateway, "ipv4.dns", ",".join(dns),
        "ipv4.ignore-auto-dns", "yes", "connection.autoconnect", "yes", "connection.autoconnect-priority", "50")
    run("sudo", "systemd-run", "--unit=edgeface-network-rollback", "--on-active=180s",
        "/usr/bin/nmcli", "connection", "up", "uuid", old_uuid)
    run("sudo", "nmcli", "connection", "up", "edgeface-static")
    print("Applied. Verify a new SSH connection, then run this script with --confirm within 180 seconds.")


if __name__ == "__main__":
    main()
