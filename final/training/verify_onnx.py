"""Evaluate real test crops with ONNX; compare releases across laptop and Pi CPUs."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil

from edgeface.preprocess import reject, save_json, softmax, tensor_from_rgb
from raspberry_pi.inference import Classifier


def checked_image(root: Path, row: dict) -> np.ndarray:
    path = root / row["file"]
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Test sample escapes dataset root")
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError(f"Sample checksum mismatch: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return tensor_from_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Laptop ONNX output for the identical test files and release")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.threads < 1 or args.warmup < 1:
        parser.error("threads and warmup must be positive")
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    classifier = Classifier(args.model, args.threads)
    calibration = json.loads((args.model / "threshold.json").read_text())
    if calibration.get("manifest_sha256") != digest or classifier.classes != manifest["classes"]:
        raise ValueError("Dataset manifest/classes do not match the calibrated release")
    rows = [r for r in manifest["samples"] if r["split"] == "test"]
    if not rows:
        raise ValueError("No real test crops")
    reference = None
    if args.reference:
        reference = json.loads(args.reference.read_text())
        if reference["bundle_version"] != classifier.bundle_version or reference["manifest_sha256"] != digest:
            raise ValueError("Reference uses a different release or dataset")
        reference = {r["file"]: r for r in reference["predictions"]}
        if set(reference) != {r["file"] for r in rows}:
            raise ValueError("Reference test file set differs")
    first = checked_image(args.data, rows[0])
    for _ in range(args.warmup):
        classifier.session.run(None, {classifier.input_name: first})
    matrix = np.zeros((len(classifier.classes), len(classifier.classes)), dtype=int)
    raw_correct = 0
    latencies, differences, output = [], [], []
    for row in rows:
        tensor = checked_image(args.data, row)
        started = time.perf_counter()
        logits = classifier.session.run(None, {classifier.input_name: tensor})[0][0]
        latency = (time.perf_counter() - started) * 1000
        person, confidence = reject(softmax(logits), classifier.classes, classifier.threshold)
        label = classifier.classes.index(row["person"])
        guess = classifier.classes.index(person)
        matrix[label, guess] += 1
        raw_correct += int(int(logits.argmax()) == label)
        latencies.append(latency)
        if reference is not None:
            expected = reference[row["file"]]
            np.testing.assert_allclose(logits, expected["logits"], rtol=1e-3, atol=1e-4)
            if expected["predicted"] != person or expected["true"] != row["person"]:
                raise RuntimeError(f"Cross-device decision mismatch: {row['file']}")
            differences.append(float(np.max(np.abs(logits - expected["logits"]))))
        output.append({"file": row["file"], "true": row["person"], "predicted": person,
                       "confidence": confidence, "logits": logits.tolist(), "inference_ms": latency})
    unknown = classifier.classes.index("Unknown")
    known = [i for i in range(len(classifier.classes)) if i != unknown]
    known_n = int(matrix[known].sum())
    unknown_n = int(matrix[unknown].sum())
    if not known_n or not unknown_n:
        raise ValueError("Test needs known members and Unknown")
    per_class = {}
    for i, name in enumerate(classifier.classes):
        tp, support, predicted = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        precision, recall = tp / predicted if predicted else 0.0, tp / support if support else 0.0
        per_class[name] = {"precision": precision, "recall": recall,
                           "f1-score": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
                           "support": support}
    report = {"device": platform.node(), "architecture": platform.machine(), "onnxruntime": ort.__version__,
              "provider": classifier.session.get_providers(), "threads": args.threads, "warmup": args.warmup,
              "model_version": classifier.version, "bundle_version": classifier.bundle_version,
              "manifest_sha256": digest, "classes": classifier.classes, "threshold": classifier.threshold,
              "samples": len(rows), "accuracy": float(matrix.trace()/matrix.sum()),
              "raw_argmax_accuracy": raw_correct/len(rows),
              "known_member_accuracy": float(sum(matrix[i, i] for i in known)/known_n),
              "unknown_rejection_rate": float(matrix[unknown, unknown]/unknown_n),
              "false_acceptance_rate": float(1-matrix[unknown, unknown]/unknown_n),
              "macro_f1": float(np.mean([r["f1-score"] for r in per_class.values()])),
              "per_class": per_class, "confusion_matrix": matrix.tolist(),
              "independent_known_test": manifest["independent_known_test"],
              "unknown_identity_disjoint": manifest["unknown_identity_disjoint"],
              "test_inference_latency": {"mean_ms": float(np.mean(latencies)),
                                         "p50_ms": float(np.percentile(latencies, 50)),
                                         "p95_ms": float(np.percentile(latencies, 95)),
                                         "definition": "One classifier session.run per test crop, after warmup; preprocessing excluded; not camera FPS"},
              "process_rss_mb": psutil.Process().memory_info().rss/1024**2,
              "cross_device_verified": reference is not None,
              "max_reference_logit_difference": max(differences) if differences else None,
              "predictions": output}
    save_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
