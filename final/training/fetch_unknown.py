from __future__ import annotations

import argparse
import hashlib
import random
import tarfile
import urllib.request
import uuid
from pathlib import Path

import cv2

from edgeface.preprocess import save_json
from edgeface.protocol import now_iso
from raspberry_pi.detection import FaceDetector, crop_face

# Archive and SHA256 from scikit-learn's official LFW loader.
URL = "https://ndownloader.figshare.com/files/5976015"
SHA256 = "b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a"


def main() -> None:
    parser = argparse.ArgumentParser(description="Import public LFW photos as Unknown, retaining identity provenance")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--identities", type=int, default=60)
    parser.add_argument("--per-identity", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detector", type=Path, default=Path("models/face_detection_yunet_2023mar.onnx"))
    args = parser.parse_args()
    cache = args.dataset / "downloads"
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    archive = cache / "lfw-funneled.tgz"
    if not archive.exists():
        temp = archive.with_suffix(".part")
        request = urllib.request.Request(URL, headers={"User-Agent": "EdgeFace-coursework/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response, temp.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if hashlib.sha256(temp.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError("LFW archive checksum mismatch")
        temp.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("LFW archive checksum mismatch")
    detector = FaceDetector(str(args.detector))
    rng = random.Random(args.seed)
    total = 0
    selected: list[str] = []
    with tarfile.open(archive, "r:gz") as tar:
        groups: dict[str, list[tarfile.TarInfo]] = {}
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if member.isfile() and len(parts) == 3 and parts[0] == "lfw_funneled" and parts[-1].endswith(".jpg"):
                groups.setdefault(parts[1], []).append(member)
        identities = [k for k, v in groups.items() if len(v) >= 2]
        rng.shuffle(identities)
        for identity in identities:
            if len(selected) >= args.identities:
                break
            members = groups[identity]
            rng.shuffle(members)
            samples = []
            session_id = uuid.uuid4().hex
            directory = args.dataset / "raw" / "Unknown" / session_id
            # Never extract arbitrary archive paths. Decode only selected JPEG bytes.
            for member in members[:args.per_identity * 3]:
                import numpy as np
                source = tar.extractfile(member)
                if source is None:
                    continue
                frame = cv2.imdecode(np.frombuffer(source.read(), np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                faces = detector.detect(frame)
                if len(faces) != 1:
                    continue
                crop = crop_face(frame, faces[0])
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                filename = f"{len(samples)+1:05d}.jpg"
                output = directory / filename
                if not cv2.imwrite(str(output), crop):
                    raise OSError("Could not save Unknown face")
                output.chmod(0o600)
                samples.append({"file": filename, "original": member.name,
                                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "timestamp": now_iso()})
                if len(samples) >= args.per_identity:
                    break
            if samples:
                anonymous_id = "lfw-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
                save_json(directory / "session.json", {"session_id": session_id, "person": "Unknown",
                    "identity": anonymous_id, "source": "public_lfw", "source_url": URL,
                    "archive_sha256": SHA256, "source_recording": anonymous_id,
                    "consent": "not_verified_public_research_dataset", "started_at": now_iso(),
                    "samples": samples, "captured": len(samples), "event": "completed"})
                (directory / "session.json").chmod(0o600)
                selected.append(anonymous_id)
                total += len(samples)
    save_json(args.dataset / "metadata" / "unknown_source.json", {"dataset": "LFW funneled", "url": URL,
        "sha256": SHA256, "identities": selected, "images": total, "seed": args.seed,
        "usage": "Local educational experiment; do not redistribute images or claim verified participant consent",
        "reference": "https://people.cs.umass.edu/~elm/papers/lfw.pdf",
        "limitations": "Web-photo domain differs from live Pi camera; verify on consenting unseen participants"})
    print(f"Imported {total} real public face crops from {len(selected)} identities as Unknown.")


if __name__ == "__main__":
    main()
