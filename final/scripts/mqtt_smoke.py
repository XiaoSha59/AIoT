from __future__ import annotations

import argparse
import json
import queue
import time
from dataclasses import replace
from pathlib import Path

from edgeface.config import Config
from edgeface.mqtt import MQTTLink
from edgeface.preprocess import save_json
from edgeface.protocol import decode, make_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Real MQTT/Pi status-start-stop smoke test; camera frames stay on Pi")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--output", type=Path, default=Path("reports/generated/mqtt_smoke.json"))
    args = parser.parse_args()
    cfg = replace(Config.load(), mqtt_username="edgeface_bot")
    messages: queue.Queue = queue.Queue()
    link = MQTTLink(cfg, "smoke", lambda t, p, r: messages.put((t, decode(p), r)))
    # The smoke role uses bot subscriptions and bot ACL credentials.
    log = []
    events = []
    def request(action: str, command=None):
        command = command or make_command(action, cfg.device_id)
        if not link.publish("command", command):
            raise RuntimeError("MQTT publish failed")
        deadline = time.monotonic() + cfg.command_timeout
        while time.monotonic() < deadline:
            try:
                topic, value, retained = messages.get(timeout=1)
            except queue.Empty:
                continue
            if topic == "result" and not retained:
                events.append(value)
            if topic == "status" and value.get("event") == "ack" and value.get("request_id") == command["request_id"]:
                log.append(value)
                if not value.get("ok"):
                    raise RuntimeError(f"Pi command failed: {value.get('error') or value.get('command_error')}")
                return value, command
        raise TimeoutError(f"No Pi acknowledgement for {action}")
    link.start()
    try:
        if not link.connected.wait(10):
            raise RuntimeError("Broker unreachable or credentials rejected")
        request("status")
        request("members")
        response, command = request("start")
        assert response["state"] == "recognizing" and response["camera"] and response["model_version"]
        duplicate, _ = request("start", command)
        assert duplicate.get("duplicate") is True
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            try:
                topic, value, retained = messages.get(timeout=1)
            except queue.Empty:
                continue
            if topic == "result" and not retained:
                events.append(value)
        response, _ = request("stop")
        assert response["state"] == "idle" and response["camera"] is None
        save_json(args.output, {"passed": True, "actual_device": cfg.device_id, "acknowledgements": log,
                               "events": events, "camera_started_and_released": True,
                               "duplicate_start_not_reexecuted": True})
        print(json.dumps({"passed": True, "events": events, "output": str(args.output)}, indent=2))
    finally:
        if link.connected.is_set():
            link.publish("command", make_command("stop", cfg.device_id))
            time.sleep(0.2)
        link.close()


if __name__ == "__main__":
    main()
