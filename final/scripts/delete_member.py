"""Explicit local biometric deletion with dry-run default. Run on laptop AND Pi."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from edgeface.protocol import safe_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--confirm-name", help="Must exactly match name to delete; omit for dry run")
    args = parser.parse_args()
    name = safe_name(args.name)
    root = args.dataset.resolve()
    candidates = [root / "raw" / name, root / "videos" / name]
    candidates += [root / "processed" / split / name for split in ["train", "val", "test"]]
    # Include archived imports/splits bearing this exact class directory name.
    for path in (root / "archive").rglob(name) if (root / "archive").exists() else []:
        if path.is_dir() and path not in candidates:
            candidates.append(path)
    for path in candidates:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Refusing symlink/outside-dataset deletion")
    existing = [path for path in candidates if path.exists()]
    print(json.dumps({"member": name, "directories": [str(p) for p in existing],
                      "mode": "delete" if args.confirm_name == name else "dry_run"}, indent=2))
    if args.confirm_name != name:
        return
    for path in existing:
        shutil.rmtree(path)
    manifest = root / "processed" / "manifest.json"
    if manifest.exists():
        # Invalidate split; a checkpoint tied to its hash must no longer train/evaluate.
        manifest.unlink()
    for path in (root / "metadata").glob("*.json") if (root / "metadata").exists() else []:
        if name in path.read_text():
            path.unlink()
    print("Biometric samples removed locally; split invalidated. Remove other copies/backups and retrain/redeploy without this member.")
    print("Existing model weights are derived data and are NOT erased by deleting image samples.")


if __name__ == "__main__":
    main()
