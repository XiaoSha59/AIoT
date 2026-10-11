from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from edgeface.preprocess import tensor_from_rgb


class FaceDataset(Dataset):
    def __init__(self, root: Path, split: str, augment: bool = False):
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.classes = self.manifest["classes"]
        self.rows = [r for r in self.manifest["samples"] if r["split"] == split]
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        ]) if augment else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(self.root / row["file"]) as source:
            image = source.convert("RGB")
            if self.augment:
                image = self.augment(image)
            tensor = torch.from_numpy(tensor_from_rgb(image)[0])
        return tensor, self.classes.index(row["person"])
