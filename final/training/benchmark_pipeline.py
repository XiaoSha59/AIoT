"""Measure detection, preprocessing and ONNX on real camera frames or an explicit replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from collections import Counter
from pathlib import Path

import cv2
import psutil

from edgeface.config import Config
from edgeface.names import load_display_names
from edgeface.preprocess import save_json
from raspberry_pi.camera import Camera
from raspberry_pi.detection import FaceDetector, crop_face
from raspberry_pi.inference import Classifier
from raspberry_pi.snapshots import annotated_jpeg
from training.benchmark_runtime import MemorySampler, hardware, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["camera", "replay"], required=True)
    parser.add_argument("--frame", type=Path, help="Real full camera/video frame, never a generated fixture")
    parser.add_argument("--model", type=Path, default=Path("models/active"))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--opencv-threads", type=int, help="Default preserves the deployed OpenCV native thread count")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.iterations, args.warmup, args.threads) < 1:
        parser.error("Measurements and threads must be positive")
    if args.mode == "replay" and not args.frame:
        parser.error("Replay requires a real --frame")
    os.umask(0o077)
    if args.opencv_threads is not None:
        if args.opencv_threads < 1:
            parser.error("OpenCV thread count must be positive")
        cv2.setNumThreads(args.opencv_threads)
    cfg = Config.load()
    if args.output.exists():
        raise ValueError("Choose a new output; never overwrite measured results")
    process = psutil.Process()
    before = hardware()
    rss_before = process.memory_info().rss
    classifier = Classifier(args.model, args.threads)
    detector = FaceDetector(str(cfg.detector_path))
    names = load_display_names(cfg.display_names_path)
    camera = Camera(cfg)
    replay = cv2.imread(str(args.frame)) if args.frame else None
    if args.mode == "replay" and replay is None:
        raise ValueError("Real replay frame cannot be decoded")
    times = {key: [] for key in ["acquisition", "detection", "crop_preprocess_rejection",
                               "classifier", "recognition_end_to_end", "all_frames"]}
    face_counts, predictions = Counter(), Counter()
    last_known = None
    backend = pixel_format = None
    with MemorySampler() as memory:
        try:
            if args.mode == "camera":
                camera.open()
                backend, pixel_format = camera.backend, camera.pixel_format
            def read():
                return camera.read() if args.mode == "camera" else replay.copy()
            for _ in range(args.warmup):
                frame = read()
                faces = detector.detect(frame)
                if len(faces) == 1:
                    classifier.predict(crop_face(frame, faces[0]))
            cpu_start = process.cpu_times()
            started = time.perf_counter()
            for _ in range(args.iterations):
                tick = time.perf_counter()
                frame = read()
                acquired = time.perf_counter()
                faces = detector.detect(frame)
                detected = time.perf_counter()
                face_counts[str(len(faces))] += 1
                times["acquisition"].append((acquired-tick)*1000)
                times["detection"].append((detected-acquired)*1000)
                if len(faces) == 1:
                    person, confidence, latency = classifier.predict(crop_face(frame, faces[0]))
                    finished = time.perf_counter()
                    times["classifier"].append(latency)
                    times["crop_preprocess_rejection"].append(max(0, (finished-detected)*1000-latency))
                    times["recognition_end_to_end"].append((finished-tick)*1000)
                    predictions[person] += 1
                    if person != "Unknown":
                        # Only one retained frame, never an accumulating video buffer.
                        last_known = (frame, faces[0], person, confidence)
                times["all_frames"].append((time.perf_counter()-tick)*1000)
            elapsed = time.perf_counter()-started
            cpu_end = process.cpu_times()
        finally:
            camera.close()
    after = hardware()
    rss_closed = process.memory_info().rss
    annotations = []
    image_bytes = []
    if last_known:
        frame, face, person, confidence = last_known
        for _ in range(20):
            tick = time.perf_counter()
            payload, _ = annotated_jpeg(frame, face, names.get(person, person), confidence, cfg)
            annotations.append((time.perf_counter()-tick)*1000)
            image_bytes.append(len(payload))
    cpu_s = (cpu_end.user+cpu_end.system)-(cpu_start.user+cpu_start.system)
    face_frames = sum(predictions.values())
    result = {"device": platform.node(), "architecture": platform.machine(), "mode": args.mode,
              "model_version": classifier.version, "bundle_version": classifier.bundle_version,
              "threads": args.threads, "opencv_threads": cv2.getNumThreads(), "warmup": args.warmup,
              "frames": args.iterations, "resolution": [frame.shape[1], frame.shape[0]],
              "camera_backend": backend, "camera_pixel_format": pixel_format, "camera_closed": camera.backend is None,
              "single_face_frames": face_frames, "face_counts": dict(face_counts),
              "prediction_counts_without_ground_truth": dict(predictions),
              "timings": {key: summary(values) if values else None for key, values in times.items()},
              "wall_time_s": elapsed, "pipeline_all_frames_fps": args.iterations/elapsed,
              "recognition_frames_per_second": face_frames/elapsed,
              "cpu_percent_one_core_units": 100*cpu_s/elapsed,
              "cpu_percent_total_capacity": 100*cpu_s/elapsed/psutil.cpu_count(),
              "memory": {"baseline_rss_bytes": rss_before, "pipeline_peak_rss_bytes": memory.peak,
                         "rss_after_camera_closed_bytes": rss_closed,
                         "process_high_water_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},
              "hardware_before": before, "hardware_after_pipeline": after,
              "annotation_separate": {"timing": summary(annotations) if annotations else None,
                                      "jpeg_bytes_mean": sum(image_bytes)/len(image_bytes) if image_bytes else None,
                                      "definition": "20 renders/JPEG encodes on last actual known frame; excludes HTTP/Telegram; not included in pipeline FPS"},
              "definitions": {"end_to_end": "read start to detection, crop/alignment, RGB/normalize, ONNX and threshold completion; no network",
                              "fps": "Unthrottled continuous vision benchmark; production sampling/cooldown/auto-stop disabled for this measurement",
                              "replay": "Recorded real frame, copy time is not physical camera acquisition",
                              "no_identity_accuracy_claim": "Frame predictions have no verified live ground truth; count performance only"}}
    if args.frame:
        result["frame_sha256"] = hashlib.sha256(args.frame.read_bytes()).hexdigest()
    save_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
