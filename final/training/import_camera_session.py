"""Import a consented Pi session with explicit temporal gaps and provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2

from edgeface.preprocess import save_json
from edgeface.protocol import safe_name
from training.import_video import temporal_split


def import_session(source: Path, dataset: Path, swap_red_blue: bool = False) -> dict:
    meta = json.loads((source / "session.json").read_text())
    if meta.get("consent") is not True or meta.get("source") != "pi_camera":
        raise ValueError("Expected a consented Pi camera session")
    person = safe_name(meta["person"])
    if person == "Unknown":
        raise ValueError("Unknown requires identity-disjoint splitting")
    samples = meta["samples"]
    times = [datetime.fromisoformat(s["timestamp"]).timestamp() for s in samples]
    if len(samples) < 20 or times != sorted(times) or times[-1] <= times[0]:
        raise ValueError("Need enough timestamped samples from a chronological session")
    # Check every source before creating derivatives. Original files stay untouched.
    for sample in samples:
        path = source / sample["file"]
        if path.parent != source or hashlib.sha256(path.read_bytes()).hexdigest() != sample["sha256"]:
            raise ValueError("Source path/checksum mismatch")
    counts: Counter = Counter()
    sessions = {}
    for split in ("train", "val", "test"):
        sid = uuid.uuid4().hex
        dest = dataset / "raw" / person / sid
        dest.mkdir(parents=True, mode=0o700)
        sessions[split] = (dest, {**meta, "session_id": sid,
            "source_recording": meta["session_id"], "source_session": str(source),
            "source_metadata_sha256": hashlib.sha256((source / "session.json").read_bytes()).hexdigest(),
            "forced_split": split, "evaluation_protocol": "camera_purged_temporal",
            "color_correction": "swap_red_blue_for_legacy_MJPEG_RGB_as_BGR" if swap_red_blue else "none",
            "limitation": "Chronological blocks of one capture; not independent sessions",
            "samples": []})
    for sample, timestamp in zip(samples, times):
        split = temporal_split((timestamp-times[0])/(times[-1]-times[0]))
        if split is None:
            counts["purged"] += 1
            continue
        dest, derived = sessions[split]
        image = cv2.imread(str(source / sample["file"]))
        if image is None:
            raise ValueError("Cannot decode source image")
        if swap_red_blue:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        filename = Path(sample["file"]).stem + ".png"
        target = dest / filename
        if not cv2.imwrite(str(target), image):
            raise OSError("Could not write derived image")
        target.chmod(0o600)
        derived["samples"].append({**sample, "file": filename,
            "source_sha256": sample["sha256"],
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        counts[split] += 1
    for dest, derived in sessions.values():
        derived["captured"] = len(derived["samples"])
        save_json(dest / "session.json", derived)
        (dest / "session.json").chmod(0o600)
    report = {"source_session": str(source), "source_images": len(samples),
              "swap_red_blue": swap_red_blue, "counts": dict(counts)}
    save_json(dataset / "metadata" / "camera_import.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--swap-red-blue", action="store_true",
                        help="Only for legacy Picamera2 MJPEG captures saved as BGR")
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(import_session(args.source, args.dataset, args.swap_red_blue), indent=2))


if __name__ == "__main__":
    main()
