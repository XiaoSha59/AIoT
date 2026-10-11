from __future__ import annotations

import json
import argparse
import os
import queue
import time
from dataclasses import replace
from pathlib import Path

from edgeface.config import Config
from edgeface.mqtt import MQTTLink
from edgeface.preprocess import save_json
from edgeface.protocol import decode, make_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify real Pi camera recognition triggers the running bot to stop")
    parser.add_argument("--output", type=Path, default=Path("reports/generated/auto_stop_smoke.json"))
    args = parser.parse_args()
    cfg = replace(Config.load(), mqtt_username="edgeface_bot")
    observed: queue.Queue = queue.Queue()
    monitor_cfg = replace(cfg, mqtt_username="edgeface_monitor", mqtt_password=os.environ["MQTT_MONITOR_PASSWORD"])
    monitor = MQTTLink(monitor_cfg, "auto-stop-observer", lambda t, p, r: observed.put((t, decode(p), r)))
    publisher = MQTTLink(cfg, "auto-stop-test", lambda *args: None)
    command = make_command("start", cfg.device_id)
    recognition = None
    auto_stop_id = None
    stop_ack = None
    monitor.start()
    publisher.start()
    try:
        if not monitor.connected.wait(10) or not publisher.connected.wait(10):
            raise RuntimeError("MQTT unavailable")
        monitor.client.subscribe(f"{cfg.prefix}/command", qos=1)
        time.sleep(0.2)
        if not publisher.publish("command", command):
            raise RuntimeError("Start publish failed")
        deadline = time.monotonic()+30
        while time.monotonic() < deadline:
            try:
                topic, value, retained = observed.get(timeout=1)
            except queue.Empty:
                continue
            if retained:
                continue
            if topic == "status" and value.get("request_id") == command["request_id"] and not value.get("ok"):
                raise RuntimeError("Pi refused test start; it may already be busy")
            if topic == "result" and value.get("request_id") == command["request_id"] and value.get("event") == "recognition" and value.get("person") != "Unknown":
                recognition = value
            if topic == "command" and value.get("action") == "stop" and value.get("expected_request_id") == command["request_id"]:
                auto_stop_id = value["request_id"]
            if topic == "status" and auto_stop_id and value.get("request_id") == auto_stop_id and value.get("event") == "ack":
                stop_ack = value
                break
        result = {"passed": bool(recognition and stop_ack and stop_ack.get("ok") and stop_ack.get("state") == "idle" and stop_ack.get("camera") is None),
                  "recognition": recognition, "bot_stop_request_id": auto_stop_id, "stop_ack": stop_ack,
                  "definition": "Real Pi camera → ONNX known result → running Telegram bot publishes guarded stop → Pi confirms camera closed"}
        save_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["passed"]:
            raise RuntimeError("Auto-stop not observed; a known member must be present in front of Pi camera")
    finally:
        publisher.publish("command", make_command("stop", cfg.device_id, expected_request_id=command["request_id"]))
        time.sleep(0.2)
        publisher.close()
        monitor.close()


if __name__ == "__main__":
    main()
