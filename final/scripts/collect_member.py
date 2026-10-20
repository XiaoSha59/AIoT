"""Collect consenting member faces locally on Pi via correlated MQTT commands."""
from __future__ import annotations

import argparse
import json
import os
import queue
import time
import uuid
from dataclasses import replace
from pathlib import Path

from edgeface.config import Config
from edgeface.mqtt import MQTTLink
from edgeface.preprocess import save_json
from edgeface.protocol import decode, make_command, safe_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--target", type=int, default=150)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--consent", action="store_true")
    parser.add_argument("--stop-first", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/generated/collection.json"))
    args = parser.parse_args()
    os.umask(0o077)
    if not args.consent:
        parser.error("Participant consent is required")
    name = safe_name(args.name)
    if name.lower() == "unknown" or not 1 <= args.target <= 1000 or args.timeout <= 0:
        parser.error("Use a member label, target 1..1000 and positive timeout")
    cfg = replace(Config.load(), mqtt_username="edgeface_bot")
    messages: queue.Queue = queue.Queue()
    link = MQTTLink(cfg, "collect-cli-" + uuid.uuid4().hex[:8],
                    lambda topic, payload, retained: messages.put((topic, decode(payload), retained)))

    def dispatch(action: str, **kwargs) -> dict:
        command = make_command(action, cfg.device_id, **kwargs)
        if not link.publish("command", command):
            raise RuntimeError("MQTT command could not be published")
        deadline = time.monotonic() + cfg.command_timeout
        while time.monotonic() < deadline:
            try:
                topic, value, retained = messages.get(timeout=0.5)
            except queue.Empty:
                continue
            if topic == "status" and not retained and value.get("request_id") == command["request_id"]:
                if not value.get("ok"):
                    raise RuntimeError(f"Pi refused {action}: {value.get('command_error') or value.get('error')}")
                return value
        raise TimeoutError(f"Pi did not acknowledge {action}")

    active_id = None
    link.start()
    try:
        if not link.connected.wait(10):
            raise RuntimeError("MQTT unavailable")
        if args.stop_first:
            dispatch("stop")
        ack = dispatch("collect", name=name, consent=True, target=args.target)
        active_id = ack["request_id"]
        print(json.dumps(ack, ensure_ascii=False), flush=True)
        save_json(args.output, ack)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            try:
                topic, value, retained = messages.get(timeout=0.5)
            except queue.Empty:
                continue
            if retained or value.get("request_id") != active_id:
                continue
            if topic == "collection":
                print(json.dumps(value, ensure_ascii=False), flush=True)
                save_json(args.output, value)
                if value.get("event") == "completed":
                    active_id = None
                    return
                if value.get("event") in {"aborted", "error", "timeout"}:
                    raise RuntimeError(f"Collection ended: {value['event']}")
            elif topic == "result" and value.get("event") == "error":
                raise RuntimeError(f"Pi collection failed: {value.get('error')}")
            elif topic == "status" and value.get("state") == "error":
                raise RuntimeError(f"Pi collection failed: {value.get('error')}")
        raise TimeoutError("Collection timed out; partial images remain on Pi")
    finally:
        if active_id:
            try:
                dispatch("stop", expected_request_id=active_id)
            except (RuntimeError, TimeoutError):
                pass
        link.close()


if __name__ == "__main__":
    main()
