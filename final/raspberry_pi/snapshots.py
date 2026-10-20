"""Annotated recognition snapshots: bounded RAM cache and authenticated HTTP over VPN."""
from __future__ import annotations

import hmac
import io
import math
import re
import threading
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from edgeface.config import Config
from edgeface.snapshot import MAX_IMAGE_BYTES


def annotated_jpeg(frame: np.ndarray, face: np.ndarray, label: str,
                   confidence: float, cfg: Config) -> tuple[bytes, dict]:
    """Draw on the original camera frame, leaving the inference input unchanged."""
    if not 0 <= confidence <= 1 or not np.isfinite(face[:4]).all():
        raise ValueError("Invalid recognition annotation")
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    scale = min(1.0, 1280 / max(image.size))
    if scale < 1:
        image = image.resize((round(image.width*scale), round(image.height*scale)))
    width, height = image.size
    x, y, w, h = map(float, face[:4])
    if w <= 0 or h <= 0:
        raise ValueError("Invalid recognition bounding box")
    x1, y1 = max(0, math.floor(x*scale)), max(0, math.floor(y*scale))
    x2, y2 = min(width-1, math.ceil((x+w)*scale)), min(height-1, math.ceil((y+h)*scale))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box outside frame")
    draw = ImageDraw.Draw(image)
    font_size = max(14, min(24, width//28))
    font = ImageFont.truetype(str(cfg.snapshot_font), font_size)
    text = f"{label}\nConfidence: {confidence:.1%}"
    while font_size > 10 and draw.multiline_textbbox((0, 0), text, font=font)[2] > width-16:
        font_size -= 1
        font = ImageFont.truetype(str(cfg.snapshot_font), font_size)
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    tw, th = box[2]-box[0]+12, box[3]-box[1]+12
    tx = max(0, min(x1, width-tw))
    ty = max(0, y1-th) if y1 >= th else min(height-th, y2+4)
    ty = max(0, ty)
    draw.rectangle((x1, y1, x2, y2), outline=(40, 230, 100), width=3)
    draw.rectangle((tx, ty, tx+tw, ty+th), fill=(15, 30, 20))
    draw.multiline_text((tx+6-box[0], ty+6-box[1]), text, font=font, fill="white", spacing=4)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    payload = output.getvalue()
    if len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("Encoded snapshot exceeds size limit")
    return payload, {"width": width, "height": height, "bbox": [x1, y1, x2-x1, y2-y1]}


class SnapshotService:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.items: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
        self.lock = threading.Lock()
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def _prune(self) -> None:
        now = time.monotonic()
        for image_id, (expires, _) in list(self.items.items()):
            if expires <= now:
                self.items.pop(image_id)

    def add(self, frame: np.ndarray, face: np.ndarray, label: str, confidence: float) -> dict:
        if len(self.cfg.snapshot_token) < 32:
            raise ValueError("Snapshot bearer token is missing")
        payload, metadata = annotated_jpeg(frame, face, label, confidence, self.cfg)
        self.start()
        image_id = uuid.uuid4().hex
        with self.lock:
            self._prune()
            self.items[image_id] = (time.monotonic()+self.cfg.snapshot_ttl, payload)
            while len(self.items) > 8:
                self.items.popitem(last=False)
        return {"id": image_id, "content_type": "image/jpeg", "expires_in_s": self.cfg.snapshot_ttl, **metadata}

    def get(self, image_id: str) -> bytes | None:
        with self.lock:
            self._prune()
            item = self.items.get(image_id)
            return item[1] if item else None

    def start(self) -> None:
        if self.server:
            return
        if len(self.cfg.snapshot_token) < 32:
            raise ValueError("Snapshot bearer token is missing")
        self.closing = False
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # Do not log face identifiers or authorization headers.

            def do_GET(self):
                authorization = self.headers.get("Authorization", "")
                expected = "Bearer " + service.cfg.snapshot_token
                if not hmac.compare_digest(authorization.encode(), expected.encode()):
                    self.send_error(401)
                    return
                match = re.fullmatch(r"/snapshots/([a-f0-9]{32})\.jpg", self.path)
                payload = service.get(match[1]) if match else None
                if payload is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer((self.cfg.snapshot_bind, self.cfg.snapshot_port), Handler)
        self.server.daemon_threads = True
        self.server.timeout = 1
        self.thread = threading.Thread(target=self._serve, name="snapshot-http", daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        assert self.server is not None
        # Also expire bytes while idle; no face images remain indefinitely in RAM.
        while not getattr(self, "closing", False):
            self.server.handle_request()
            with self.lock:
                self._prune()

    def close(self) -> None:
        self.closing = True
        if self.thread:
            self.thread.join(timeout=2)
        if self.server:
            self.server.server_close()
        with self.lock:
            self.items.clear()
        self.server = None
        self.thread = None
