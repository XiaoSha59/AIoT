from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

from edgeface.config import Config
from edgeface.preprocess import save_json, tensor_from_rgb
from raspberry_pi.agent import temperature
from raspberry_pi.camera import Camera
from raspberry_pi.detection import FaceDetector, crop_face
from raspberry_pi.inference import Classifier


def summary(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"samples": len(values), "mean_ms": float(np.mean(values)), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "min_ms": min(values), "max_ms": max(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark real inputs locally; classifier vs camera pipeline reported separately")
    parser.add_argument("--model", type=Path, default=Path("models/active"))
    parser.add_argument("--image", type=Path, required=True, help="Real already-cropped face, kept local")
    parser.add_argument("--checkpoint", type=Path, help="Optional PyTorch CPU comparison on this same device")
    parser.add_argument("--detector", type=Path, default=Path("models/face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("reports/generated/benchmark.json"))
    args = parser.parse_args()
    if args.iterations < 2 or args.warmup < 1:
        parser.error("Need at least 2 iterations and 1 warmup")
    cfg = Config(ort_threads=args.threads)
    classifier = Classifier(args.model, args.threads)
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError("Real benchmark face image is missing")
    tensor = tensor_from_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    process = psutil.Process()
    temp_start = temperature()
    model_bytes = (args.model / "face_classifier.onnx").stat().st_size
    latency: list[float] = []
    for _ in range(args.warmup):
        classifier.session.run(None, {classifier.input_name: tensor})
    cpu_start = process.cpu_times()
    start = time.perf_counter()
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        classifier.session.run(None, {classifier.input_name: tensor})
        latency.append((time.perf_counter()-t0)*1000)
    elapsed = time.perf_counter()-start
    cpu_end = process.cpu_times()
    cpu_percent = ((cpu_end.user+cpu_end.system)-(cpu_start.user+cpu_start.system))/elapsed*100
    result = {"device": platform.node(), "machine": platform.machine(), "os": platform.platform(),
        "threads": args.threads, "warmup": args.warmup, "model_version": classifier.version,
        "model_size_bytes": model_bytes, "input": str(args.image), "classifier_onnx": summary(latency),
        "classifier_cpu_percent_one_core_units": cpu_percent, "ram_mb": process.memory_info().rss/1024**2,
        "temperature_start_c": temp_start, "timestamp": time.time(),
        "latency_definition": "ONNX session.run only; preprocessing excluded"}
    if args.checkpoint:
        import torch
        from training.model import load_checkpoint
        torch.set_num_threads(args.threads)
        model, _ = load_checkpoint(args.checkpoint, "cpu")
        x = torch.from_numpy(tensor)
        times = []
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(x)
            for _ in range(args.iterations):
                start = time.perf_counter()
                model(x)
                times.append((time.perf_counter()-start)*1000)
        result["classifier_pytorch_cpu_same_device"] = summary(times)
    if args.camera:
        detector = FaceDetector(str(args.detector))
        camera = Camera(cfg)
        detection, e2e, classifier_camera, acquisition = [], [], [], []
        cpu_start = process.cpu_times()
        face_frames = 0
        try:
            camera.open()
            result["camera_backend"] = camera.backend
            for _ in range(args.warmup):
                detector.detect(camera.read())
            total_start = time.perf_counter()
            for _ in range(args.iterations):
                start = time.perf_counter()
                frame = camera.read()
                acquire_end = time.perf_counter()
                faces = detector.detect(frame)
                detect_end = time.perf_counter()
                acquisition.append((acquire_end-start)*1000)
                detection.append((detect_end-acquire_end)*1000)
                if len(faces) == 1:
                    crop = crop_face(frame, faces[0])
                    _, _, inference_ms = classifier.predict(crop)
                    classifier_camera.append(inference_ms)
                    face_frames += 1
                    e2e.append((time.perf_counter()-start)*1000)
            elapsed = time.perf_counter()-total_start
            cpu_end = process.cpu_times()
            result.update(camera_resolution=[frame.shape[1], frame.shape[0]], detection=summary(detection),
                acquisition=summary(acquisition), end_to_end_single_face=summary(e2e),
                classifier_live_faces=summary(classifier_camera), camera_frames=args.iterations,
                single_face_frames=face_frames, pipeline_fps=args.iterations/elapsed,
                camera_cpu_percent_one_core_units=((cpu_end.user+cpu_end.system)-(cpu_start.user+cpu_start.system))/elapsed*100,
                end_to_end_definition="Camera read starts → crop, preprocessing and classifier finished; excludes MQTT/Telegram network",
                sampling="Unthrottled benchmark; deployed agent applies configurable frame interval")
        finally:
            camera.close()
    result["temperature_end_c"] = temperature()
    result["ram_end_mb"] = process.memory_info().rss/1024**2
    save_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
