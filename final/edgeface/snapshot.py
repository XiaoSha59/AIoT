"""Download a single bounded JPEG from a configured Pi origin, never from MQTT URLs."""
from __future__ import annotations

import io
import re
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from PIL import Image

from edgeface.config import Config

MAX_IMAGE_BYTES = 2 * 1024 * 1024
IMAGE_ID = re.compile(r"[a-f0-9]{32}")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_snapshot(cfg: Config, image_id: str) -> bytes:
    if not isinstance(image_id, str) or not IMAGE_ID.fullmatch(image_id):
        raise ValueError("Invalid snapshot identifier")
    origin = urlsplit(cfg.snapshot_base_url)
    if (origin.scheme not in {"http", "https"} or not origin.hostname or origin.username
            or origin.password or origin.path not in {"", "/"} or origin.query or origin.fragment):
        raise ValueError("SNAPSHOT_BASE_URL must be a fixed HTTP(S) origin")
    request = Request(cfg.snapshot_base_url.rstrip("/") + f"/snapshots/{image_id}.jpg",
                      headers={"Authorization": "Bearer " + cfg.snapshot_token})
    # Tailnet traffic stays direct; do not forward its bearer token through an env proxy.
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=cfg.snapshot_timeout) as response:
        if response.headers.get_content_type() != "image/jpeg":
            raise ValueError("Expected a JPEG snapshot")
        if int(response.headers.get("Content-Length", "0")) > MAX_IMAGE_BYTES:
            raise ValueError("Snapshot exceeds size limit")
        payload = response.read(MAX_IMAGE_BYTES + 1)
    if len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("Snapshot exceeds size limit")
    with Image.open(io.BytesIO(payload)) as image:
        if image.format != "JPEG" or max(image.size) > 1280 or min(image.size) < 1:
            raise ValueError("Invalid snapshot dimensions/format")
        image.verify()
    return payload
