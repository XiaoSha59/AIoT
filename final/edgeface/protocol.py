from __future__ import annotations

import json
import re
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

ACTIONS = {"start", "stop", "status", "members", "collect", "collect_unknown"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[\w -]{1,64}", name, flags=re.UNICODE):
        raise ValueError("Name must be 1–64 letters, digits, spaces, '-' or '_'")
    name = name.strip()
    if not name or name in {".", ".."}:
        raise ValueError("Invalid name")
    return name


def make_command(action: str, device_id: str, **kwargs: Any) -> dict[str, Any]:
    return {"request_id": str(uuid.uuid4()), "device_id": device_id, "action": action,
            "timestamp": now_iso(), **kwargs}


def decode(payload: bytes) -> dict[str, Any]:
    if len(payload) > 16_384:
        raise ValueError("Payload too large")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Payload must be an object")
    return value


def validate_command(value: dict[str, Any], ttl: float, device_id: str) -> dict[str, Any]:
    if value.get("action") not in ACTIONS:
        raise ValueError("Unsupported command action")
    uuid.UUID(str(value.get("request_id", "")))
    if value.get("device_id") != device_id:
        raise ValueError("Wrong device_id")
    when = datetime.fromisoformat(str(value.get("timestamp", "")))
    if when.tzinfo is None:
        raise ValueError("Timestamp must include timezone")
    age = (datetime.now(timezone.utc) - when).total_seconds()
    if age > ttl or age < -10:
        raise ValueError("Expired command or clock skew")
    if value["action"] in {"collect", "collect_unknown"}:
        if value.get("consent") is not True:
            raise ValueError("Collection requires recorded consent")
        safe_name(value.get("name", "Unknown") if value["action"] == "collect" else "Unknown")
        if value["action"] == "collect" and value.get("name", "").lower() == "unknown":
            raise ValueError("Use collect_unknown for Unknown")
        if value["action"] == "collect_unknown":
            safe_name(value.get("identity", ""))
        target = value.get("target", 150)
        if isinstance(target, bool) or not isinstance(target, int) or not 1 <= target <= 1000:
            raise ValueError("Target must be an integer between 1 and 1000")
    return value


class RequestCache:
    """Bounded in-process duplicate protection; commands also expire by timestamp.

    MQTT clean sessions remove offline queued commands. Retained commands are refused.
    """
    def __init__(self, capacity: int = 1024):
        self.items: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.capacity = capacity

    def get(self, request_id: str) -> dict[str, Any] | None:
        return self.items.get(request_id)

    def put(self, request_id: str, response: dict[str, Any]) -> None:
        self.items[request_id] = response
        self.items.move_to_end(request_id)
        while len(self.items) > self.capacity:
            self.items.popitem(last=False)


class Cooldown:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.last: dict[str, float] = {}

    def allow(self, key: str, now: float) -> bool:
        if now - self.last.get(key, float("-inf")) < self.seconds:
            return False
        self.last[key] = now
        return True
