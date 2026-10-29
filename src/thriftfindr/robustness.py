"""Test-set accuracy under fixed corruptions (blur, glare, rotation, perspective, JPEG, low light).

Compares fine-tuned checkpoints with the frozen-feature linear probe on identical corrupted images
(each test image's corruption is seeded by its row position).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from thriftfindr import baselines, features, paths
from thriftfindr.data import manifest, registry
from thriftfindr.data.dataset import ManifestDataset, label_maps
from thriftfindr.data.transforms import CORRUPTIONS, SEVERITIES, eval_transform
from thriftfindr.finetune import load_checkpoint, predict
from thriftfindr.models.backbones import BACKBONES, default_device, load_encoder


def _conditions(corruptions, severities):
    yield ("clean", 0)
    for kind in corruptions:
        for sev in severities:
            yield (kind, sev)


def _accuracies(scores: dict, df, classes) -> dict:
    return {t: float((scores[t].argmax(1) == df[t].map({c: i for i, c in enumerate(n)}).to_numpy()).mean())
            for t, n in classes.items()}


def _loader(test, dataset, backbone, classes, condition, batch_size, num_workers):
    corruption = None if condition[0] == "clean" else condition
    ds = ManifestDataset(test, paths.dataset_dir(dataset), eval_transform(BACKBONES[backbone]), classes,
                         corruption=corruption)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)


def _nest(flat: dict) -> dict:
    out: dict = {}
    for (kind, sev), acc in flat.items():
        out.setdefault(kind, {})[str(sev)] = acc
    return out


def evaluate_checkpoint(checkpoint, corruptions=CORRUPTIONS, severities=SEVERITIES,
                        batch_size: int = 128, num_workers: int = 6) -> dict:
    device = default_device()
    model, ckpt = load_checkpoint(checkpoint, device)
    cfg, classes = ckpt["config"], ckpt["classes"]
    columns = {**classes, **ckpt.get("inputs", {})}  # the loader must also yield the model's inputs
    df = manifest.load(cfg["dataset"])
    test = df[df.split == "test"].reset_index(drop=True)
    flat = {}
    for cond in tqdm(list(_conditions(corruptions, severities)), desc=f"robustness {cfg['backbone']}_{cfg['aug']}"):
        loader = _loader(test, cfg["dataset"], cfg["backbone"], columns, cond, batch_size, num_workers)
        flat[cond] = _accuracies(predict(model, loader, device), test, classes)
    return _nest(flat)


@torch.inference_mode()
def evaluate_frozen_probe(dataset: str, backbone: str, probe_C: dict[str, float], corruptions=CORRUPTIONS,
                          severities=SEVERITIES, batch_size: int = 128, num_workers: int = 6) -> dict:
    """Fit the linear probe on clean cached train features, then score corrupted test images."""
    spec = registry.get(dataset)
    df = manifest.load(dataset)
    classes = label_maps(df, spec.tasks)
    train, test = df[df.split == "train"], df[df.split == "test"].reset_index(drop=True)
    X_train = features.load(dataset, backbone, train)
    probes = {}
    for t, names in classes.items():
        y = train[t].map({c: i for i, c in enumerate(names)}).to_numpy()
        probes[t] = baselines.fit_linear_probe(X_train, y, X_train[:1], y[:1], Cs=(probe_C[t],))
    device = default_device()
    encoder = load_encoder(backbone).to(device).eval()
    flat = {}
    for cond in tqdm(list(_conditions(corruptions, severities)), desc=f"robustness {backbone}_frozen_probe"):
        loader = _loader(test, dataset, backbone, classes, cond, batch_size, num_workers)
        X = np.concatenate([encoder(x.to(device)).float().cpu().numpy() for x, _ in loader])
        flat[cond] = _accuracies({t: baselines.probe_scores(p, X) for t, p in probes.items()}, test, classes)
    return _nest(flat)
