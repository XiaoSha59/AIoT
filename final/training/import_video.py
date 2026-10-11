from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np

from edgeface.config import Config
from edgeface.protocol import make_command, safe_name
from raspberry_pi.collection import Collector
from raspberry_pi.detection import FaceDetector, crop_face


def temporal_split(fraction: float) -> str | None:
    # Contiguous blocks with gaps, never randomly interleave adjacent frames.
    if 0 <= fraction < 0.5:
        return "train"
    if 0.6 <= fraction < 0.75:
        return "val"
    if 0.85 <= fraction <= 1:
        return "test"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect/crop real faces before frame extraction")
    parser.add_argument("video", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--detector", type=Path, default=Path("models/face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--interval", type=float, default=0.15)
    parser.add_argument("--min-blur", type=float, default=60)
    parser.add_argument("--duplicate-hamming", type=int, default=2)
    parser.add_argument("--consent", action="store_true", help="Participant agreed to use video for local training")
    parser.add_argument("--single-video-temporal", "--purged-temporal", dest="single_video_temporal", action="store_true", help="Purged temporal split within this recording; not independent-session evaluation")
    args = parser.parse_args()
    if not args.consent:
        parser.error("--consent is required for participant video")
    safe_name(args.name)
    if args.interval <= 0:
        parser.error("interval must be positive")
    cfg = Config(dataset_dir=args.dataset, min_blur=args.min_blur, collection_interval=args.interval,
                 duplicate_hamming=args.duplicate_hamming, collection_target=1000)
    detector = FaceDetector(str(args.detector))
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count <= 0:
            raise RuntimeError("Video FPS/duration unavailable")
        duration = count / fps
        sha = hashlib.sha256(args.video.read_bytes()).hexdigest()
        command = make_command("collect", "video-import", name=args.name, consent=True, target=1000)
        collectors: dict[str, Collector] = {}
        modes = ["train", "val", "test"] if args.single_video_temporal else ["session"]
        for mode in modes:
            collector = Collector(cfg, command)
            collector.meta.update(source="participant_video", video=str(args.video), video_sha256=sha,
                source_recording=sha, duration_s=duration, fps=fps,
                evaluation_protocol="single_video_purged_temporal" if args.single_video_temporal else "session_split")
            if mode != "session":
                collector.meta["forced_split"] = mode
                collector.meta["limitation"] = "All known-member splits share one source video; external validity not established"
            collectors[mode] = collector
        frame_number = 0
        next_sample = 0.0
        examined = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_number / fps
            frame_number += 1
            if timestamp + 1e-6 < next_sample:
                continue
            next_sample = timestamp + args.interval
            split = temporal_split(timestamp/duration) if args.single_video_temporal else "session"
            if split is None:
                continue
            examined += 1
            collector = collectors[split]
            scale = min(1.0, 640 / max(frame.shape[:2]))
            view = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
            faces = detector.detect(view)
            if len(faces) != 1:
                collector.rejections["no_face" if not faces else "multiple_faces"] += 1
                continue
            face = np.asarray(faces[0]).copy()
            face[:14] /= scale
            crop = crop_face(frame, face)
            if collector.offer(crop, float(min(face[2:4])), timestamp) is None:
                collector.samples[-1]["video_timestamp_s"] = timestamp
        results = {}
        for mode, collector in collectors.items():
            collector.finish("completed")
            results[mode] = {"saved": collector.accepted, "rejected": dict(collector.rejections), "session": collector.session_id}
        print({"duration_s": duration, "frames_examined": examined, "results": results})
    finally:
        cap.release()


if __name__ == "__main__":
    main()
