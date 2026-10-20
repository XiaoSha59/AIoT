"""Regression: negotiated USB MJPEG and CSI RGB888 must yield identical BGR."""
import numpy as np
import pytest

from edgeface.config import Config
from raspberry_pi.camera import Camera


@pytest.mark.parametrize("fmt,pixel", [
    ("MJPEG", [190, 110, 60]), ("BGR888", [190, 110, 60]),
    ("RGB888", [60, 110, 190]), ("XBGR8888", [190, 110, 60, 255]),
    ("XRGB8888", [60, 110, 190, 255]),
])
def test_negotiated_capture_formats_preserve_red_and_blue(fmt, pixel):
    class Capture:
        def capture_array(self, name):
            return np.array([[pixel]], dtype=np.uint8)

    camera = Camera(Config())
    camera.picam = Capture()
    camera.pixel_format = fmt
    np.testing.assert_array_equal(camera.read(), [[[60, 110, 190]]])
