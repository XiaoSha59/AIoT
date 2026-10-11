from __future__ import annotations

import argparse
import json
import queue
import time
from dataclasses import replace

from edgeface.config import Config
from edgeface.mqtt import MQTTLink
from edgeface.protocol import decode, make_command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status", "members"])
    args = parser.parse_args()
    cfg = replace(Config.load(), mqtt_username="edgeface_bot")
    messages: queue.Queue = queue.Queue()
    link = MQTTLink(cfg, "cli", lambda t, p, r: messages.put((t, decode(p), r)))
    link.start()
    try:
        if not link.connected.wait(10):
            raise RuntimeError("MQTT unavailable")
        command = make_command(args.action, cfg.device_id)
        if not link.publish("command", command):
            raise RuntimeError("Publish failed")
        deadline = time.monotonic()+cfg.command_timeout
        while time.monotonic() < deadline:
            try:
                topic, value, retained = messages.get(timeout=1)
            except queue.Empty:
                continue
            if topic == "status" and value.get("request_id") == command["request_id"] and not retained:
                print(json.dumps(value, ensure_ascii=False, indent=2))
                if not value.get("ok"):
                    raise RuntimeError("Device refused command")
                return
        raise TimeoutError("Pi did not acknowledge command")
    finally:
        link.close()


if __name__ == "__main__":
    main()
