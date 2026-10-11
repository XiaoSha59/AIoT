from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable

import paho.mqtt.client as mqtt

from edgeface.config import Config
from edgeface.protocol import now_iso

log = logging.getLogger(__name__)


class MQTTLink:
    def __init__(self, config: Config, role: str, callback: Callable[[str, bytes, bool], None]):
        self.config = config
        self.role = role
        self.callback = callback
        self.connected = threading.Event()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id=f"edgeface-{role}-{config.device_id}", clean_session=True)
        self.client.username_pw_set(config.mqtt_username, config.mqtt_password)
        self.client.reconnect_delay_set(1, 30)
        self.client.max_queued_messages_set(100)
        if role == "pi":
            will = {"device_id": config.device_id, "online": False, "state": "offline", "timestamp": now_iso()}
            self.client.will_set(f"{config.prefix}/status", json.dumps(will), qos=1, retain=True)
        self.client.on_connect = self._connect
        self.client.on_disconnect = self._disconnect
        self.client.on_message = self._message

    def _connect(self, client: mqtt.Client, userdata: Any, flags: Any, reason_code: Any, properties: Any) -> None:
        if reason_code.is_failure:
            log.error("MQTT connection rejected: %s", reason_code)
            return
        topics = ["command"] if self.role == "pi" else ["status", "result", "telemetry", "collection"]
        for topic in topics:
            client.subscribe(f"{self.config.prefix}/{topic}", qos=1)
        self.connected.set()
        log.info("MQTT connected (%s)", self.role)

    def _disconnect(self, client: mqtt.Client, userdata: Any, flags: Any, reason_code: Any, properties: Any) -> None:
        self.connected.clear()
        log.warning("MQTT disconnected: %s", reason_code)

    def _message(self, client: mqtt.Client, userdata: Any, message: mqtt.MQTTMessage) -> None:
        try:
            self.callback(message.topic.rsplit("/", 1)[-1], message.payload, message.retain)
        except Exception:
            log.exception("MQTT message rejected")

    def start(self) -> None:
        self.client.connect_async(self.config.mqtt_host, self.config.mqtt_port, keepalive=30)
        self.client.loop_start()

    def publish(self, topic: str, value: dict[str, Any], retain: bool = False) -> bool:
        if topic == "command" and retain:
            raise ValueError("Commands must never be retained")
        # Drop events while offline instead of replaying old recognition after reconnect.
        if not self.connected.is_set():
            return False
        info = self.client.publish(f"{self.config.prefix}/{topic}", json.dumps(value, ensure_ascii=False),
                                   qos=0 if topic == "telemetry" else 1, retain=retain)
        return info.rc == mqtt.MQTT_ERR_SUCCESS

    def close(self) -> None:
        self.client.disconnect()
        self.client.loop_stop()
