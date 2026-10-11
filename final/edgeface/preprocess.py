from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

PREPROCESS = {"size": [224, 224], "color": "RGB", "scale": 255.0,
              "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
              "resize": "PIL_BILINEAR", "layout": "NCHW", "dtype": "float32"}


def bundle_version(directory: Path) -> str:
    import hashlib
    files = ["face_classifier.onnx", "classes.json", "preprocessing.json", "threshold.json"]
    return hashlib.sha256(b"".join((directory / name).read_bytes() for name in files)).hexdigest()[:12]


def tensor_from_rgb(image: Image.Image | np.ndarray, config: dict | None = None) -> np.ndarray:
    cfg = config or PREPROCESS
    if cfg != PREPROCESS:
        raise ValueError("Unsupported preprocessing config; training and inference must match")
    im = image if isinstance(image, Image.Image) else Image.fromarray(image)
    im = im.convert("RGB").resize(tuple(cfg["size"]), Image.Resampling.BILINEAR)
    array = np.asarray(im, dtype=np.float32) / cfg["scale"]
    array = (array - np.asarray(cfg["mean"], np.float32)) / np.asarray(cfg["std"], np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None], dtype=np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(z)):
        raise ValueError("Model returned non-finite logits")
    z = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


def reject(probabilities: np.ndarray, classes: list[str], threshold: float) -> tuple[str, float]:
    probs = np.asarray(probabilities).reshape(-1)
    if len(probs) != len(classes) or not np.isfinite(probs).all():
        raise ValueError("Invalid classifier output")
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    label = classes[idx]
    return ("Unknown" if label == "Unknown" or confidence < threshold else label), confidence


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
