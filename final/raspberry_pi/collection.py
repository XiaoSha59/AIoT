from __future__ import annotations

import hashlib
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from edgeface.config import Config
from edgeface.preprocess import save_json
from edgeface.protocol import now_iso, safe_name


def image_hash(bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


class Collector:
    def __init__(self, cfg: Config, command: dict[str, Any]):
        self.cfg = cfg
        self.name = "Unknown" if command["action"] == "collect_unknown" else safe_name(command["name"])
        self.identity = safe_name(command["identity"]) if self.name == "Unknown" else self.name
        self.target = command.get("target", cfg.collection_target)
        self.request_id = command["request_id"]
        self.session_id = uuid.uuid4().hex
        self.path = cfg.dataset_dir / "raw" / self.name / self.session_id
        self.path.mkdir(parents=True, mode=0o700)
        self.path.chmod(0o700)
        self.started = time.monotonic()
        self.last_saved = float("-inf")
        self.accepted = 0
        self.rejections: Counter[str] = Counter()
        self.hashes: list[int] = []
        self.samples: list[dict[str, Any]] = []
        self.meta = {"session_id": self.session_id, "person": self.name, "identity": self.identity,
                     "consent": True, "request_id": self.request_id, "device_id": cfg.device_id,
                     "started_at": now_iso(), "source": "pi_camera", "target": self.target,
                     "camera": cfg.camera_backend, "quality": {"min_face_size": cfg.min_face_size,
                     "min_blur": cfg.min_blur, "blur_reference_size": [224, 224],
                     "brightness": [cfg.min_brightness, cfg.max_brightness]}}
        self.finish("collecting")

    def offer(self, crop: np.ndarray, face_size: float, now: float) -> str | None:
        if now - self.last_saved < self.cfg.collection_interval:
            self.rejections["sample_interval"] += 1
            return "sample_interval"
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        reference = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
        blur = float(cv2.Laplacian(reference, cv2.CV_64F).var())
        brightness = float(gray.mean())
        reason = None
        if face_size < self.cfg.min_face_size:
            reason = "face_too_small"
        elif blur < self.cfg.min_blur:
            reason = "blur"
        elif not self.cfg.min_brightness <= brightness <= self.cfg.max_brightness:
            reason = "brightness"
        fingerprint = image_hash(crop)
        if reason is None and any((fingerprint ^ previous).bit_count() <= self.cfg.duplicate_hamming for previous in self.hashes):
            reason = "duplicate"
        if reason is not None:
            self.rejections[reason] += 1
            return reason
        filename = f"{self.accepted+1:05d}.jpg"
        path = self.path / filename
        if not cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError("Could not save face crop")
        os.chmod(path, 0o600)
        self.samples.append({"file": filename, "timestamp": now_iso(), "blur": blur,
                             "brightness": brightness, "dhash": str(fingerprint),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.hashes.append(fingerprint)
        self.accepted += 1
        self.last_saved = now
        # Persist after every accepted crop; abrupt power loss still leaves usable metadata.
        self.finish("collecting")
        return None

    def progress(self, event: str = "progress") -> dict[str, Any]:
        return {"event": event, "request_id": self.request_id, "person": self.name,
                "session_id": self.session_id, "captured": self.accepted, "target": self.target,
                "rejected": sum(self.rejections.values()), "rejections": dict(self.rejections),
                "storage": str(self.path), "timestamp": now_iso()}

    def finish(self, outcome: str) -> None:
        save_json(self.path / "session.json", {**self.meta, **self.progress(outcome), "samples": self.samples})
        (self.path / "session.json").chmod(0o600)
