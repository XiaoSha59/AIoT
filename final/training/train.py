from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from edgeface.preprocess import PREPROCESS, save_json
from training.data import FaceDataset
from training.model import build_model, load_checkpoint, seed_everything


def epoch(model, loader, loss_fn, device, optimizer=None) -> tuple[float, float]:
    model.train(optimizer is not None)
    # Frozen backbone BatchNorm statistics stay fixed; small face datasets otherwise destabilize them.
    if optimizer is not None:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            total += len(labels)
    return loss_sum / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Real pretrained MobileNetV3-Small head training and partial backbone fine-tuning")
    parser.add_argument("--data", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--output", type=Path, default=Path("runs/mvp"))
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--fine-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--init-checkpoint", type=Path, help="Continue from a local verified checkpoint; replay old training data to reduce forgetting")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--fine-lr", type=float, default=1e-4)
    args = parser.parse_args()
    os.umask(0o077)
    seed_everything(args.seed)
    torch.set_num_threads(4)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; run outside restricted sandbox or choose CPU")
    train = FaceDataset(args.data, "train", augment=True)
    val = FaceDataset(args.data, "val")
    if not len(train) or not len(val):
        raise ValueError("Training/validation data missing")
    if args.output.exists() and (args.output / "best_model.pth").exists():
        raise ValueError("Run output already exists; choose a new --output for reproducibility")
    args.output.mkdir(parents=True, exist_ok=True)
    loader_args = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device == "cuda"}
    train_loader = DataLoader(train, shuffle=True, **loader_args)
    val_loader = DataLoader(val, shuffle=False, **loader_args)
    counts = Counter(row["person"] for row in train.rows)
    weights = torch.tensor([len(train)/(len(train.classes)*counts[name]) for name in train.classes], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    initial = None
    initial_hash = None
    if args.init_checkpoint:
        model, initial = load_checkpoint(args.init_checkpoint, device)
        if initial["classes"] != train.classes or initial["preprocessing"] != PREPROCESS:
            raise ValueError("Initialization checkpoint class order/preprocessing mismatch")
        initial_hash = hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest()
    else:
        model = build_model(len(train.classes), pretrained=True).to(device)
    starting_head = model.classifier[-1].weight.detach().cpu().clone()
    backbone_initial = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if name.startswith("features.10")}
    manifest_hash = hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest()
    initial_val_loss, initial_val_accuracy = epoch(model, val_loader, loss_fn, device)
    if initial is not None:
        # Same weights as the previous model, evaluated against the new manifest.
        # This checkpoint is a comparison artifact, never a new training result.
        initial_eval = {**initial, "manifest_sha256": manifest_hash,
                        "evaluation_protocols": train.manifest["evaluation_protocols"],
                        "source_checkpoint_sha256": initial_hash, "stage": "initial_comparison"}
        torch.save(initial_eval, args.output / "initial_model.pth")
    history = []
    global_best = float("inf")
    started = time.perf_counter()
    for stage, epochs, lr in (("head", args.head_epochs, args.head_lr), ("fine", args.fine_epochs, args.fine_lr)):
        if epochs <= 0:
            continue
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("classifier") or (stage == "fine" and name.startswith(("features.10", "features.11", "features.12")))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        stage_best, stale = float("inf"), 0
        for number in range(epochs):
            tr_loss, tr_acc = epoch(model, train_loader, loss_fn, device, optimizer)
            va_loss, va_acc = epoch(model, val_loader, loss_fn, device)
            row = {"epoch": len(history)+1, "stage": stage, "train_loss": tr_loss, "train_accuracy": tr_acc,
                   "val_loss": va_loss, "val_accuracy": va_acc}
            history.append(row)
            print(json.dumps(row), flush=True)
            checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "classes": train.classes, "preprocessing": PREPROCESS, "seed": args.seed,
                          "pretrained_weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1", "stage": stage,
                          "epoch": row["epoch"], "val_loss": va_loss, "manifest_sha256": manifest_hash,
                          "evaluation_protocols": train.manifest["evaluation_protocols"]}
            if initial_hash:
                checkpoint["init_checkpoint_sha256"] = initial_hash
            if va_loss < stage_best:
                stage_best, stale = va_loss, 0
                torch.save(checkpoint, args.output / f"best_{stage}.pth")
            else:
                stale += 1
            if va_loss < global_best:
                global_best = va_loss
                torch.save(checkpoint, args.output / "best_model.pth")
            save_json(args.output / "history.json", history)
            if stale >= args.patience:
                break
        best = torch.load(args.output / f"best_{stage}.pth", map_location=device, weights_only=False)
        model.load_state_dict(best["state_dict"])
    final = torch.load(args.output / "best_model.pth", map_location="cpu", weights_only=False)
    head_delta = float((final["state_dict"]["classifier.3.weight"] - starting_head).abs().max())
    backbone_delta = max(float((final["state_dict"][k] - v).abs().max()) for k, v in backbone_initial.items())
    if head_delta <= 0:
        raise RuntimeError("Classifier weights did not change; training invalid")
    metadata = {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                "torch": torch.__version__, "elapsed_s": time.perf_counter()-started, "seed": args.seed,
                "classifier_max_weight_update": head_delta, "last_backbone_max_weight_update": backbone_delta,
                "best_stage": final["stage"], "best_epoch": final["epoch"], "manifest_sha256": manifest_hash,
                "initial_checkpoint_sha256": initial_hash,
                "initial_validation_loss": initial_val_loss, "initial_validation_accuracy": initial_val_accuracy,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    save_json(args.output / "training_run.json", metadata)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        for split in ["train", "val"]:
            ax.plot([r["epoch"] for r in history], [r[f"{split}_{metric}"] for r in history], label=split)
        ax.set(xlabel="Epoch", ylabel=metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "training_curves.png", dpi=150)
    plt.close(fig)
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
