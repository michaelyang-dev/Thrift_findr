"""Extract and cache frozen-backbone features for every image in a manifest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from thriftfindr import paths
from thriftfindr.data import manifest
from thriftfindr.data.dataset import ManifestDataset
from thriftfindr.data.transforms import eval_transform
from thriftfindr.models.backbones import BACKBONES, default_device, load_encoder


def features_path(dataset: str, backbone: str) -> Path:
    return paths.FEATURES_DIR / dataset / f"{backbone}.npz"


@torch.inference_mode()
def extract(dataset: str, backbone: str, batch_size: int = 128, num_workers: int = 6,
            overwrite: bool = False) -> Path:
    out = features_path(dataset, backbone)
    if out.exists() and not overwrite:
        return out
    df = manifest.load(dataset)
    ds = ManifestDataset(df, paths.dataset_dir(dataset), eval_transform(BACKBONES[backbone]))
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    device = default_device()
    encoder = load_encoder(backbone).to(device).eval()
    feats = [encoder(x.to(device)).float().cpu().numpy()
             for x, _ in tqdm(loader, desc=f"{dataset}/{backbone}", unit="batch")]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, features=np.concatenate(feats).astype(np.float32), image_id=df.image_id.to_numpy())
    return out


def load(dataset: str, backbone: str, df: pd.DataFrame) -> np.ndarray:
    """Cached features in the row order of df (checked by image_id)."""
    data = np.load(features_path(dataset, backbone), allow_pickle=True)
    index = pd.Series(np.arange(len(data["image_id"])), index=data["image_id"])
    missing = set(df.image_id) - set(index.index)
    if missing:
        raise KeyError(f"{len(missing)} images have no cached {backbone} features; re-run extraction")
    return data["features"][index.loc[df.image_id].to_numpy()]
