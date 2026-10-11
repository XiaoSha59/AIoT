from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from edgeface.preprocess import reject, save_json, softmax
from training.data import FaceDataset
from training.model import load_checkpoint


def collect_predictions(checkpoint: Path, data: Path, split: str):
    torch.set_num_threads(4)
    model, metadata = load_checkpoint(checkpoint)
    dataset = FaceDataset(data, split)
    manifest_hash = hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest()
    if metadata["manifest_sha256"] != manifest_hash or metadata["classes"] != dataset.classes:
        raise ValueError("Dataset/checkpoint provenance mismatch")
    arrays, labels = [], []
    with torch.inference_mode():
        for images, targets in DataLoader(dataset, batch_size=16, shuffle=False):
            arrays.append(model(images).numpy())
            labels.extend(targets.numpy().tolist())
    return softmax(np.concatenate(arrays)), np.asarray(labels), dataset, metadata


def metrics(labels: np.ndarray, prediction: np.ndarray, classes: list[str]) -> dict:
    unknown = classes.index("Unknown")
    known_mask, unknown_mask = labels != unknown, labels == unknown
    if not known_mask.any() or not unknown_mask.any():
        raise ValueError("Evaluation needs both known and unknown examples")
    known_acc = float(np.mean(labels[known_mask] == prediction[known_mask]))
    rejection = float(np.mean(prediction[unknown_mask] == unknown))
    return {"accuracy": float(accuracy_score(labels, prediction)), "known_member_accuracy": known_acc,
            "unknown_rejection_rate": rejection, "false_acceptance_rate": 1-rejection,
            "known_false_rejection_rate": float(np.mean(prediction[known_mask] == unknown)),
            "balanced_known_unknown_accuracy": (known_acc+rejection)/2,
            "known_samples": int(known_mask.sum()), "unknown_samples": int(unknown_mask.sum()),
            "per_class": classification_report(labels, prediction, labels=list(range(len(classes))),
                                                target_names=classes, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(labels, prediction, labels=list(range(len(classes)))).tolist()}


def predictions(probabilities: np.ndarray, classes: list[str], threshold: float) -> np.ndarray:
    return np.asarray([classes.index(reject(p, classes, threshold)[0]) for p in probabilities])


def calibrate(probabilities: np.ndarray, labels: np.ndarray, classes: list[str], max_far: float = 0.05) -> dict:
    unknown = classes.index("Unknown")
    # Consider acceptance scores only where the network predicts a known member.
    scores = probabilities.max(axis=1)[probabilities.argmax(axis=1) != unknown]
    # A threshold below 1/K never rejects a softmax argmax. Use the chance
    # floor so the selected midpoint can provide actual uncertainty rejection.
    chance_floor = 1 / len(classes)
    points = sorted(set([chance_floor, 1.0, *map(float, scores)]))
    candidates = [chance_floor, 1.0, *[(a+b)/2 for a, b in zip(points[:-1], points[1:])]]
    trials = []
    for threshold in candidates:
        result = metrics(labels, predictions(probabilities, classes, threshold), classes)
        trials.append({"threshold": threshold, **{k: result[k] for k in ["known_member_accuracy", "false_acceptance_rate", "balanced_known_unknown_accuracy"]}})
    feasible = [r for r in trials if r["false_acceptance_rate"] <= max_far]
    chosen = max(feasible or trials, key=lambda r: (r["balanced_known_unknown_accuracy"], r["known_member_accuracy"], r["threshold"]))
    return {"source": "validation", "threshold": chosen["threshold"], "max_validation_far": max_far,
            "selected_validation_metrics": chosen, "validation_samples": len(labels),
            "selection_rule": "Maximize balanced known/unknown accuracy subject to validation FAR; ties prefer known accuracy then conservative midpoint",
            "chance_floor": chance_floor, "candidates": trials,
            "limitation": "Small validation set; empirical FAR does not establish real-world FAR"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["calibrate", "test"])
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mvp/best_model.pth"))
    parser.add_argument("--data", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--output", type=Path, default=Path("runs/mvp/evaluation"))
    parser.add_argument("--threshold", type=Path, default=Path("runs/mvp/threshold.json"))
    parser.add_argument("--max-far", type=float, default=0.05)
    args = parser.parse_args()
    split = "val" if args.action == "calibrate" else "test"
    probabilities, labels, dataset, metadata = collect_predictions(args.checkpoint, args.data, split)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.action == "calibrate":
        calibration = calibrate(probabilities, labels, dataset.classes, args.max_far)
        calibration["checkpoint_sha256"] = checkpoint_hash
        calibration["manifest_sha256"] = metadata["manifest_sha256"]
        save_json(args.threshold, calibration)
        print(json.dumps({k: v for k, v in calibration.items() if k != "candidates"}, indent=2))
        return
    threshold = json.loads(args.threshold.read_text())
    if threshold["source"] != "validation" or threshold["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Calibrate this checkpoint on validation data first")
    pred = predictions(probabilities, dataset.classes, threshold["threshold"])
    report = metrics(labels, pred, dataset.classes)
    report.update(threshold=threshold["threshold"], evaluation_split="test", classes=dataset.classes,
        independent_known_test=dataset.manifest["independent_known_test"],
        unknown_identity_disjoint=dataset.manifest["unknown_identity_disjoint"],
        checkpoint_sha256=checkpoint_hash, manifest_sha256=metadata["manifest_sha256"],
        limitation=("Known-member test frames share source recordings with training; web Unknown differs from Pi camera"
                    if not dataset.manifest["independent_known_test"] else "Web Unknown differs from Pi camera"))
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "metrics.json", report)
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "true", "predicted", "confidence", "correct"])
        writer.writeheader()
        for row, label, guess, probs in zip(dataset.rows, labels, pred, probabilities):
            writer.writerow({"file": row["file"], "true": dataset.classes[label], "predicted": dataset.classes[guess],
                             "confidence": float(probs.max()), "correct": bool(label == guess)})
    matrix = np.asarray(report["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(len(dataset.classes)), yticks=range(len(dataset.classes)),
           xticklabels=dataset.classes, yticklabels=dataset.classes, xlabel="Predicted", ylabel="Actual",
           title="Test · independent known recording" if dataset.manifest["independent_known_test"] else "MVP test · shared known recordings")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(args.output / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    # Local diagnostic examples only, never sent automatically through Telegram.
    wrong = np.flatnonzero(labels != pred).tolist()
    correct = np.flatnonzero(labels == pred).tolist()
    indices = (wrong[:4] + correct[:4])[:8]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, indices):
        with Image.open(args.data / dataset.rows[index]["file"]) as im:
            ax.imshow(im)
        ax.set_title(f"{dataset.classes[labels[index]]} → {dataset.classes[pred[index]]}\n{probabilities[index].max():.3f}")
    fig.tight_layout()
    fig.savefig(args.output / "prediction_examples.png", dpi=120)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k not in {"per_class", "confusion_matrix"}}, indent=2))


if __name__ == "__main__":
    main()
