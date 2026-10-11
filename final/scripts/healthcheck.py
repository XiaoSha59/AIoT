import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

role = sys.argv[1]
try:
    data = json.loads(Path(f"reports/generated/{'bot' if role == 'bot' else 'device'}_health.json").read_text())
    timestamp = data["timestamp"]
    value = timestamp if isinstance(timestamp, (float, int)) else datetime.fromisoformat(timestamp).timestamp()
    healthy = time.time() - value < 60 and data.get("mqtt_connected")
except (OSError, ValueError, KeyError):
    healthy = False
sys.exit(0 if healthy else 1)
