from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from edgeface.config import Config

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, config: Config):
        self.cfg = config
        self.capture = None
        self.picam = None
        self.backend: str | None = None

    def open(self) -> None:
        self.close()
        errors: list[str] = []
        if self.cfg.camera_backend in {"auto", "csi"}:
            try:
                from picamera2 import Picamera2
                info = Picamera2.global_camera_info()
                if not info:
                    raise CameraError("No libcamera cameras found")
                self.picam = Picamera2(self.cfg.camera_index)
                # Picamera2 RGB888 is byte-ordered BGR, matching OpenCV.
                config = self.picam.create_video_configuration(
                    main={"size": (self.cfg.camera_width, self.cfg.camera_height), "format": "RGB888"},
                    buffer_count=2)
                self.picam.configure(config)
                self.picam.start()
                camera_info = info[self.cfg.camera_index]
                model = str(camera_info.get("Model", "")).lower()
                self.backend = "libcamera_usb" if "cybertrack" in model or "usb" in str(camera_info.get("Id", "")).lower() else "libcamera"
                return
            except Exception as exc:
                errors.append(f"CSI: {exc}")
                self.close()
        if self.cfg.camera_backend in {"auto", "usb"}:
            cap = cv2.VideoCapture(self.cfg.camera_index, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                self.capture = cap
                self.backend = "usb"
                return
            cap.release()
            errors.append(f"V4L2: cannot open /dev/video{self.cfg.camera_index}")
        raise CameraError("; ".join(errors))

    def read(self) -> np.ndarray:
        try:
            if self.picam is not None:
                frame = self.picam.capture_array("main")
            elif self.capture is not None:
                ok, frame = self.capture.read()
                if not ok:
                    raise CameraError("Camera disconnected or frame unavailable")
            else:
                raise CameraError("Camera is closed")
            if frame is None or frame.size == 0:
                raise CameraError("Empty camera frame")
            return frame
        except CameraError:
            raise
        except Exception as exc:
            raise CameraError(str(exc)) from exc

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.picam is not None:
            try:
                self.picam.stop()
            finally:
                self.picam.close()
                self.picam = None
        self.backend = None


def main() -> None:
    import argparse
    from pathlib import Path
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="Capture one local diagnostic frame; no upload")
    parser.add_argument("--output", type=Path, default=Path("reports/generated/camera_probe.jpg"))
    args = parser.parse_args()
    load_dotenv()
    camera = Camera(Config())
    try:
        camera.open()
        time.sleep(1)
        frame = camera.read()
        args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not cv2.imwrite(str(args.output), frame):
            raise CameraError("Could not save diagnostic frame")
        args.output.chmod(0o600)
        print({"camera": camera.backend, "shape": list(frame.shape), "output": str(args.output)})
    finally:
        camera.close()


if __name__ == "__main__":
    main()
