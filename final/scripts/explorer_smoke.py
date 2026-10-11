from __future__ import annotations

import time
from pathlib import Path

import socketio
from dotenv import dotenv_values

from edgeface.preprocess import save_json


def main() -> None:
    values = dotenv_values(".env")
    url = f"http://127.0.0.1:{values.get('MQTT_EXPLORER_PORT') or '4000'}"
    client = socketio.Client(logger=False, engineio_logger=False)
    topics: set[str] = set()
    events: set[str] = set()
    autoconnections: set[str] = set()
    def observe(event: str, *args) -> None:
        events.add(event)
        for data in args:
            if isinstance(data, dict) and str(data.get("topic", "")).startswith("aiot/edgeface/"):
                topics.add(data["topic"])
    client.on("*", observe)
    # Browser receives an auto-connect configuration and registers it with the backend.
    @client.on("connection/add/mqtt")
    def add_connection(data):
        if isinstance(data, dict) and data.get("id") not in autoconnections:
            autoconnections.add(data["id"])
            client.emit("connection/add/mqtt", data)
    try:
        client.connect(url, auth={"username": values["MQTT_EXPLORER_USERNAME"],
                                  "password": values["MQTT_EXPLORER_PASSWORD"]},
                       transports=["websocket", "polling"], wait_timeout=10)
        deadline = time.monotonic()+15
        while time.monotonic() < deadline:
            if {"aiot/edgeface/status", "aiot/edgeface/telemetry"}.issubset(topics):
                break
            time.sleep(0.2)
        result = {"websocket_authenticated": client.connected, "received_topics": sorted(topics),
                  "event_names": sorted(events), "url": url,
                  "passed": {"aiot/edgeface/status", "aiot/edgeface/telemetry"}.issubset(topics)}
        save_json(Path("reports/generated/explorer_smoke.json"), result)
        print(result)
        if not result["passed"]:
            raise RuntimeError("Explorer socket did not receive live device telemetry")
    finally:
        if client.connected:
            client.disconnect()


if __name__ == "__main__":
    main()
