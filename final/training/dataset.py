from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from edgeface.preprocess import save_json
from edgeface.protocol import safe_name
from raspberry_pi.collection import image_hash


def assign_groups(groups: list[str], rng: random.Random) -> dict[str, str]:
    if len(groups) < 3:
        raise ValueError("Need at least three independent sessions (known) or identities (Unknown)")
    groups = sorted(groups)
    rng.shuffle(groups)
    n_test = max(1, round(len(groups) * 0.15))
    n_val = max(1, round(len(groups) * 0.15))
    return {g: "test" if i < n_test else "val" if i < n_test+n_val else "train" for i, g in enumerate(groups)}


def scan(raw: Path, delete_corrupt: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    corrupt: list[str] = []
    exact: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    for meta_path in sorted(raw.glob("*/*/session.json")):
        meta = json.loads(meta_path.read_text())
        person = safe_name(meta["person"])
        if person != meta_path.parent.parent.name:
            raise ValueError(f"Label/path mismatch: {meta_path}")
        if meta.get("source") != "public_lfw" and meta.get("consent") is not True:
            raise ValueError(f"Missing participant consent: {meta_path}")
        identity = str(meta.get("identity", person))
        for sample in meta.get("samples", []):
            path = meta_path.parent / sample["file"]
            if path.parent != meta_path.parent or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError("Unsafe sample path")
            try:
                with Image.open(path) as image:
                    image.verify()
                bgr = cv2.imread(str(path))
                if bgr is None or min(bgr.shape[:2]) < 20:
                    raise ValueError("Undecodable or too small")
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                if sample.get("sha256") != sha:
                    raise ValueError("Sample checksum changed")
            except (OSError, ValueError, SyntaxError):
                corrupt.append(str(path))
                if delete_corrupt and path.exists():
                    path.unlink()
                continue
            if sha in exact:
                duplicates.append({"file": str(path), "original": exact[sha]})
                continue
            exact[sha] = str(path)
            rows.append({"raw_path": str(path), "person": person, "identity": identity,
                         "session_id": meta["session_id"], "source_recording": meta.get("source_recording", meta["session_id"]),
                         "source": meta.get("source", "pi_camera"), "sha256": sha,
                         "dhash": image_hash(bgr), "forced_split": meta.get("forced_split"),
                         "video_timestamp_s": sample.get("video_timestamp_s"),
                         "evaluation_protocol": meta.get("evaluation_protocol", "session_split")})
    return rows, {"valid_images": len(rows), "corrupt": corrupt, "exact_duplicates_excluded": duplicates,
                  "class_distribution": dict(Counter(row["person"] for row in rows))}


def split_dataset(dataset: Path, seed: int, allow_temporal: bool, allow_multi_temporal: bool = False) -> dict[str, Any]:
    rows, audit = scan(dataset / "raw")
    if not rows:
        raise ValueError("No valid real samples found")
    classes = sorted({row["person"] for row in rows})
    if "Unknown" not in classes or len(classes) < 2:
        raise ValueError("Need known members and an explicit Unknown class")
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[row["person"]].append(row)
    for name, members in by_class.items():
        forced = [r for r in members if r["forced_split"]]
        if forced:
            recordings = {r["source_recording"] for r in members}
            if name == "Unknown" or len(forced) != len(members):
                raise ValueError("Do not mix temporal data with independent sessions or Unknown identities")
            if len(recordings) > 1 and not allow_multi_temporal:
                raise ValueError("Multiple temporal videos require --allow-multi-video-temporal; test is not independent")
            if len(recordings) == 1 and not (allow_temporal or allow_multi_temporal):
                raise ValueError("Temporal MVP data requires --allow-single-video-temporal")
            for row in members:
                if row["forced_split"] not in {"train", "val", "test"}:
                    raise ValueError("Invalid forced split")
                row["split"] = row["forced_split"]
                if len(recordings) > 1:
                    row["evaluation_protocol"] = "multi_video_purged_temporal"
        else:
            group_key = "identity" if name == "Unknown" else "source_recording"
            assignments = assign_groups(list({r[group_key] for r in members}), rng)
            for row in members:
                row["split"] = assignments[row[group_key]]
    # Remove near-identical crops crossing split boundaries. Training keeps priority.
    chosen: list[dict[str, Any]] = []
    fingerprints: dict[str, list[tuple[int, str]]] = defaultdict(list)
    near_duplicates = []
    for split in ("train", "val", "test"):
        for row in rows:
            if row["split"] != split:
                continue
            if any(previous_split != split and (row["dhash"] ^ previous).bit_count() <= 2
                   for previous, previous_split in fingerprints[row["person"]]):
                near_duplicates.append(row["raw_path"])
                continue
            fingerprints[row["person"]].append((row["dhash"], split))
            chosen.append(row)
    distribution = {split: dict(Counter(r["person"] for r in chosen if r["split"] == split)) for split in ("train", "val", "test")}
    for split, counts in distribution.items():
        if any(counts.get(c, 0) < 3 for c in classes):
            raise ValueError(f"Insufficient {split} samples after filtering: {counts}; collect more real data")
    output = dataset / "processed"
    if output.exists():
        raise ValueError("processed/ exists; use a new dataset path or archive the previous split before rebuilding")
    staging = dataset / (".processed-" + uuid.uuid4().hex)
    for row in chosen:
        relative = Path(row["split"]) / row["person"] / f"{row['session_id']}_{Path(row['raw_path']).name}"
        dest = staging / relative
        dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(row["raw_path"], dest)
        dest.chmod(0o600)
        row["file"] = str(relative)
        row.pop("dhash")
    protocols = sorted({r["evaluation_protocol"] for r in chosen})
    known_train_val = {(r["person"], r["source_recording"]) for r in chosen
                       if r["person"] != "Unknown" and r["split"] in {"train", "val"}}
    known_test = {(r["person"], r["source_recording"]) for r in chosen
                  if r["person"] != "Unknown" and r["split"] == "test"}
    manifest = {"seed": seed, "classes": classes, "distribution": distribution,
                "evaluation_protocols": protocols, "independent_known_test": not bool(known_train_val & known_test),
                "unknown_identity_disjoint": True, "samples": chosen}
    save_json(staging / "manifest.json", manifest)
    staging.rename(output)
    audit["near_duplicates_cross_split_excluded"] = near_duplicates
    save_json(dataset / "metadata" / "audit.json", audit)
    save_json(dataset / "metadata" / "split_summary.json", {k: v for k, v in manifest.items() if k != "samples"})
    return {k: v for k, v in manifest.items() if k != "samples"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["audit", "split"])
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delete-corrupt", action="store_true", help="Explicitly delete only corrupt raw images")
    parser.add_argument("--allow-single-video-temporal", action="store_true")
    parser.add_argument("--allow-multi-video-temporal", action="store_true", help="Explicit MVP fallback; known test shares source recordings with training")
    args = parser.parse_args()
    if args.action == "audit":
        _, report = scan(args.dataset / "raw", args.delete_corrupt)
        save_json(args.dataset / "metadata" / "audit.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(split_dataset(args.dataset, args.seed, args.allow_single_video_temporal, args.allow_multi_video_temporal), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
