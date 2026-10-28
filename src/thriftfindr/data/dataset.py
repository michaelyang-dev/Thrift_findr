"""PyTorch dataset over a manifest."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from thriftfindr.data.transforms import corrupt


def label_maps(df: pd.DataFrame, tasks) -> dict[str, list[str]]:
    """Sorted class names per task, from the whole manifest so indices agree across splits."""
    return {t: sorted(df[t].unique()) for t in tasks}


class ManifestDataset(Dataset):
    """Yields (image tensor, {task: class index}). Optionally applies a fixed test-time corruption,
    seeded by image id so every model sees the same corrupted images whatever the row order."""

    def __init__(self, df: pd.DataFrame, root: Path, transform, classes: dict[str, list[str]] | None = None,
                 corruption: tuple[str, int] | None = None, corrupt_size: int = 256):
        self.paths = [Path(root) / p for p in df.path]
        self.ids = list(df.image_id)
        self.transform = transform
        self.targets = {t: torch.as_tensor(df[t].map({c: i for i, c in enumerate(names)}).to_numpy())
                        for t, names in (classes or {}).items()}
        self.corruption = corruption
        self.corrupt_size = corrupt_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.corruption is not None:
            # Corrupt the centre square at a common resolution, so a severity means the same thing for
            # every dataset and aspect ratio; the eval transform's centre crop keeps the same framing.
            s = self.corrupt_size
            scale = s / min(img.size)
            img = img.resize((max(s, round(img.width * scale)), max(s, round(img.height * scale))), Image.BICUBIC)
            left, top = (img.width - s) // 2, (img.height - s) // 2
            img = img.crop((left, top, left + s, top + s))
            img = corrupt(img, *self.corruption, rng=np.random.default_rng(zlib.crc32(self.ids[i].encode())))
        return self.transform(img), {t: y[i] for t, y in self.targets.items()}
