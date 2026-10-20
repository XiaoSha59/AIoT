"""Synthetic pixels test rendering/transport only; no recognition accuracy claims."""
import io
import time
from dataclasses import replace
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener

import numpy as np
import pytest
from PIL import Image

from edgeface.config import Config
from edgeface.snapshot import fetch_snapshot
from raspberry_pi.snapshots import SnapshotService, annotated_jpeg


@pytest.fixture
def service():
    cfg = Config(snapshot_bind="127.0.0.1", snapshot_port=0, snapshot_token="test-only-" * 5)
    s = SnapshotService(cfg)
    yield s
    s.close()


def test_annotation_uses_original_frame_and_real_label_without_mutating_inference():
    frame = np.full((240, 420, 3), [50, 100, 180], dtype=np.uint8)
    before = frame.copy()
    payload, meta = annotated_jpeg(frame, np.array([-10, 40, 140, 160]),
                                   "Nguyễn Hồ Trường Sơn", 0.942, Config())
    np.testing.assert_array_equal(frame, before)
    assert meta == {"width": 420, "height": 240, "bbox": [0, 40, 130, 160]}
    with Image.open(io.BytesIO(payload)) as im:
        assert im.size == (420, 240)
        # Green bounding box exists, background retained in correct RGB order.
        assert im.getpixel((0, 120))[1] > im.getpixel((0, 120))[0]
        red, green, blue = im.getpixel((400, 200))
        assert red > green > blue


def test_http_authentication_fetch_and_expiry(service):
    meta = service.add(np.zeros((240, 420, 3), np.uint8), np.array([50, 50, 150, 150]), "Sơn", .95)
    url = f"http://127.0.0.1:{service.server.server_port}"
    cfg = replace(service.cfg, snapshot_base_url=url)
    with pytest.raises(HTTPError) as error:
        build_opener(ProxyHandler({})).open(url + f"/snapshots/{meta['id']}.jpg")
    assert error.value.code == 401
    with pytest.raises(HTTPError) as error:
        fetch_snapshot(replace(cfg, snapshot_token="wrong-token"), meta["id"])
    assert error.value.code == 401
    assert fetch_snapshot(cfg, meta["id"]).startswith(b"\xff\xd8")
    with service.lock:
        service.items[meta["id"]] = (time.monotonic()-1, b"expired")
    with pytest.raises(HTTPError) as error:
        fetch_snapshot(cfg, meta["id"])
    assert error.value.code == 404
    assert meta["id"] not in service.items


def test_ram_cache_is_bounded_and_stop_does_not_remove_snapshot(service):
    frame = np.zeros((240, 420, 3), np.uint8)
    ids = [service.add(frame, np.array([50, 50, 150, 150]), "son", .99)["id"] for _ in range(10)]
    assert len(service.items) == 8 and service.get(ids[0]) is None
    assert service.get(ids[-1]) is not None
    service.close()
    assert not service.items


@pytest.mark.parametrize("image_id", ["../.env", "http://attacker/image.jpg", "x"*32, "a"*33])
def test_mqtt_image_pointer_cannot_supply_a_path_or_url(image_id):
    with pytest.raises(ValueError, match="identifier"):
        fetch_snapshot(Config(), image_id)
