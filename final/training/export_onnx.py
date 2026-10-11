from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from edgeface.preprocess import PREPROCESS, save_json
from training.data import FaceDataset
from training.model import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mvp/best_model.pth"))
    parser.add_argument("--threshold", type=Path, default=Path("runs/mvp/threshold.json"))
    parser.add_argument("--data", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--output", type=Path, default=Path("models/active"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(args.checkpoint)
    calibration = json.loads(args.threshold.read_text())
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if calibration["source"] != "validation" or calibration["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Threshold/checkpoint provenance mismatch")
    data = FaceDataset(args.data, "val")
    if hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest() != checkpoint["manifest_sha256"]:
        raise ValueError("Dataset changed since training")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "face_classifier.onnx"
    sample = data[0][0][None]
    # Opset 17 is supported by ORT 1.20.1 on x86_64 and aarch64. Fixed batch one.
    export_args = {"input_names": ["images"], "output_names": ["logits"], "opset_version": 17,
                   "do_constant_folding": True, "dynamic_axes": None}
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_args["dynamo"] = False
    with torch.inference_mode():
        torch.onnx.export(model, sample, str(path), **export_args)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph, full_check=True)
    if graph.ir_version > 10:
        raise RuntimeError("Exported ONNX IR too new for pinned ORT; use matching exporter")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    deltas, matches, torch_predictions, onnx_predictions = [], [], [], []
    from edgeface.preprocess import reject, softmax
    with torch.inference_mode():
        for index in range(len(data)):
            images = data[index][0][None]
            torch_output = model(images).numpy()
            output = session.run(None, {"images": images.numpy()})[0]
            np.testing.assert_allclose(output, torch_output, rtol=1e-3, atol=1e-4)
            deltas.append(float(np.max(np.abs(output-torch_output))))
            matches.append(bool(output.argmax(1)[0] == torch_output.argmax(1)[0]))
            torch_predictions.append(reject(softmax(torch_output)[0], data.classes, calibration["threshold"])[0])
            onnx_predictions.append(reject(softmax(output)[0], data.classes, calibration["threshold"])[0])
    if not all(matches) or torch_predictions != onnx_predictions:
        raise RuntimeError("PyTorch/ONNX predictions differ")
    save_json(args.output / "classes.json", checkpoint["classes"])
    save_json(args.output / "preprocessing.json", PREPROCESS)
    shutil.copyfile(args.threshold, args.output / "threshold.json")
    report = {"opset": 17, "ir_version": graph.ir_version, "input_shape": [1, 3, 224, 224],
              "checker": "passed", "samples_compared": len(data), "comparison_split": "validation",
              "max_absolute_logit_difference": max(deltas), "mean_max_logit_difference": float(np.mean(deltas)),
              "argmax_agreement": float(np.mean(matches)), "rejection_decisions_match": True,
              "model_size_bytes": path.stat().st_size, "onnxruntime": ort.__version__, "torch": torch.__version__,
              "checkpoint_sha256": checkpoint_hash, "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    save_json(args.output / "export_validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
