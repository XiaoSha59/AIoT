from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from edgeface.preprocess import save_json
from raspberry_pi.detection import FaceDetector

URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("models/face_detection_yunet_2023mar.onnx"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            data = response.read(2_000_000)
        if len(data) < 100_000:
            raise RuntimeError("Detector download is too small (possibly a Git LFS pointer)")
        args.output.write_bytes(data)
    FaceDetector(str(args.output))
    save_json(args.output.with_suffix(".source.json"),
              {"url": URL, "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
               "license": "MIT (OpenCV Zoo YuNet model directory)", "size_bytes": args.output.stat().st_size})
    print(f"YuNet ready: {args.output}, {args.output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
