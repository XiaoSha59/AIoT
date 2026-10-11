import random

import pytest
from PIL import Image

from training.dataset import assign_groups, split_dataset
from training.import_video import temporal_split


def test_unknown_identities_and_sessions_are_assigned_to_only_one_split():
    assignment = assign_groups([f"identity-{i}" for i in range(30)], random.Random(42))
    partitions = {split: {k for k, v in assignment.items() if v == split} for split in ["train", "val", "test"]}
    assert all(partitions.values())
    assert not partitions["train"] & partitions["test"]
    assert not partitions["train"] & partitions["val"]
    assert not partitions["val"] & partitions["test"]


def test_single_video_temporal_split_has_purge_gaps():
    assert temporal_split(0.49) == "train"
    assert temporal_split(0.55) is None
    assert temporal_split(0.65) == "val"
    assert temporal_split(0.80) is None
    assert temporal_split(0.90) == "test"


@pytest.fixture
def temporal_dataset(tmp_path, monkeypatch):
    # Synthetic file fixtures exercise split policy only, never model metrics.
    rows = []
    rng = random.Random(14)
    for person in ["son", "Unknown"]:
        groups = ["video1", "video2"] if person == "son" else ["identity1", "identity2", "identity3"]
        for group in groups:
            splits = ["train", "val", "test"] if person == "son" else [None]
            for forced in splits:
                for index in range(3):
                    path = tmp_path / f"{person}_{group}_{forced}_{index}.jpg"
                    Image.new("RGB", (24, 24), (index*70, 100, 50)).save(path)
                    rows.append({"person": person, "identity": group, "source_recording": group,
                                 "session_id": group, "raw_path": str(path), "forced_split": forced,
                                 "evaluation_protocol": "single_video_purged_temporal" if forced else "session_split",
                                 "dhash": rng.getrandbits(64)})
    monkeypatch.setattr("training.dataset.scan", lambda *args: ([dict(r) for r in rows], {}))
    return tmp_path / "dataset"


def test_second_temporal_recording_requires_explicit_opt_in(temporal_dataset):
    with pytest.raises(ValueError, match="allow-multi-video-temporal"):
        split_dataset(temporal_dataset, 42, True)
    assert not (temporal_dataset / "processed").exists()


def test_multi_video_temporal_cannot_claim_independent_test(temporal_dataset):
    import json
    summary = split_dataset(temporal_dataset, 42, False, True)
    assert summary["independent_known_test"] is False
    assert "multi_video_purged_temporal" in summary["evaluation_protocols"]
    manifest = json.loads((temporal_dataset / "processed" / "manifest.json").read_text())
    known = [r for r in manifest["samples"] if r["person"] == "son"]
    assert all(r["split"] == r["forced_split"] for r in known)
    unknown = [r for r in manifest["samples"] if r["person"] == "Unknown"]
    for identity in {r["identity"] for r in unknown}:
        assert len({r["split"] for r in unknown if r["identity"] == identity}) == 1
