from __future__ import annotations

import json
from pathlib import Path


def load_display_names(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not v.strip()
                                           or len(v) > 200 for k, v in value.items()):
        raise ValueError("Display names must be a JSON object mapping labels to non-empty names")
    return value
