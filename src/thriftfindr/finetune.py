"""Fine-tune a pretrained backbone with one linear head per task (multi-task cross-entropy).

Schedule: the first `warmup_epochs` train only the heads on a frozen backbone, then everything is
unfrozen with a 50x smaller backbone learning rate (linear probe then fine-tune, which Kumar et al.
2022 show distorts pretrained features less than fine-tuning from a random head). The epoch with
the best mean validation macro-F1 is kept and evaluated once on the test split.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from thriftfindr import evaluation, paths
from thriftfindr.data import manifest, registry
from thriftfindr.data.dataset import ManifestDataset, label_maps
from thriftfindr.data.transforms import eval_transform, seed_worker, train_transform
from thriftfindr.models.backbones import BACKBONES, default_device, load_encoder


@dataclass
class FinetuneConfig:
    dataset: str
    backbone: str = "dinov2_vits14"
    aug: str = "strong"  # none | light | strong (see data/transforms.py)
    epochs: int = 8
    warmup_epochs: int = 1
    batch_size: int = 64
    lr_head: float = 1e-3
    lr_backbone: float = 2e-5
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    probe_init: bool = False  # start each head from the fitted linear probe on cached frozen features
    num_workers: int = 6
    seed: int = 0
    tag: str = ""

    @property
    def run_name(self) -> str:
        """backbone_aug plus every setting that differs from the defaults, so variants never share files."""
        default = FinetuneConfig(dataset=self.dataset)
        parts = [self.backbone, self.aug]
        parts += [f"{f}{getattr(self, f)}" for f in ("epochs", "warmup_epochs", "batch_size", "lr_head",
                                                     "lr_backbone", "weight_decay", "label_smoothing", "seed")
                  if getattr(self, f) != getattr(default, f)]
        if self.probe_init:
            parts.append("probeinit")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


class MultiTaskModel(nn.Module):
    def __init__(self, backbone: str, n_classes: dict[str, int], dropout: float = 0.1):
        super().__init__()
        self.encoder = load_encoder(backbone)
        dim = BACKBONES[backbone].feat_dim
        self.heads = nn.ModuleDict({t: nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n))
                                    for t, n in n_classes.items()})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.encoder(x)
        return {t: head(f) for t, head in self.heads.items()}


def balanced_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """n / (K * count_c), as sklearn's class_weight="balanced" used by the linear probe."""
    counts = np.bincount(y, minlength=n_classes).clip(min=1)
    return torch.tensor(len(y) / (n_classes * counts), dtype=torch.float32)


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device) -> dict[str, np.ndarray]:
    model.eval()
    scores: dict[str, list] = {}
    for x, _ in loader:
        for t, logits in model(x.to(device)).items():
            scores.setdefault(t, []).append(logits.softmax(-1).float().cpu().numpy())
    return {t: np.concatenate(v) for t, v in scores.items()}


def evaluate_scores(scores, df: pd.DataFrame, classes) -> dict:
    out = {}
    for t, names in classes.items():
        y = df[t].map({c: i for i, c in enumerate(names)}).to_numpy()
        out[t] = evaluation.classification_metrics(y, scores[t].argmax(1), len(names), scores[t],
                                                   df.group_id.to_numpy())
    return out


def init_heads_from_probe(model: MultiTaskModel, cfg: FinetuneConfig, train_df: pd.DataFrame, classes) -> None:
    """Copy the linear probe (fit on cached frozen features with the C chosen in run_baselines.py)
    into each head, folding the probe's feature standardization into the weights."""
    from thriftfindr import baselines, features

    base = json.loads((paths.RESULTS_DIR / cfg.dataset / "baselines.json").read_text())
    X = features.load(cfg.dataset, cfg.backbone, train_df)
    for t, names in classes.items():
        y = train_df[t].map({c: i for i, c in enumerate(names)}).to_numpy()
        C = base["results"][cfg.backbone][t]["linear_probe"]["param"]
        scaler, clf = baselines.fit_linear_probe(X, y, X[:1], y[:1], Cs=(C,)).model
        W = clf.coef_ / scaler.scale_
        b = clf.intercept_ - (clf.coef_ * scaler.mean_ / scaler.scale_).sum(axis=1)
        if W.shape[0] == 1:  # binary probe has one logit z for class 1; logits [0, z] give the same softmax
            W, b = np.vstack([np.zeros_like(W), W]), np.concatenate([[0.0], b])
        linear = model.heads[t][-1]
        assert W.shape == tuple(linear.weight.shape), (W.shape, linear.weight.shape)
        linear.weight.data.copy_(torch.as_tensor(W, dtype=torch.float32))
        linear.bias.data.copy_(torch.as_tensor(b, dtype=torch.float32))


def _set_backbone_trainable(model: MultiTaskModel, trainable: bool) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = trainable


def train(cfg: FinetuneConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    spec, bspec = registry.get(cfg.dataset), BACKBONES[cfg.backbone]
    df = manifest.load(cfg.dataset)
    classes = label_maps(df, spec.tasks)
    root = paths.dataset_dir(cfg.dataset)
    parts = {s: df[df.split == s].reset_index(drop=True) for s in manifest.SPLITS}

    train_ds = ManifestDataset(parts["train"], root, train_transform(bspec, cfg.aug, spec.hflip), classes)
    loaders = {"train": DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                                   num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0,
                                   worker_init_fn=seed_worker,
                                   generator=torch.Generator().manual_seed(cfg.seed))}
    for s in ("val", "test"):
        loaders[s] = DataLoader(ManifestDataset(parts[s], root, eval_transform(bspec), classes),
                                batch_size=2 * cfg.batch_size, num_workers=cfg.num_workers)

    device = default_device()
    model = MultiTaskModel(cfg.backbone, {t: len(c) for t, c in classes.items()})
    if cfg.probe_init:
        init_heads_from_probe(model, cfg, parts["train"], classes)
    model.to(device)
    criteria = {t: nn.CrossEntropyLoss(weight=balanced_weights(train_ds.targets[t].numpy(), len(c)).to(device),
                                       label_smoothing=cfg.label_smoothing) for t, c in classes.items()}
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg.lr_backbone},
        {"params": model.heads.parameters(), "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)
    steps_per_epoch = len(loaders["train"])
    total, warm = cfg.epochs * steps_per_epoch, steps_per_epoch  # cosine decay after one epoch of linear warmup
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    history, best = [], {"score": -1.0}
    step = 0
    for epoch in range(cfg.epochs):
        frozen = epoch < cfg.warmup_epochs
        _set_backbone_trainable(model, not frozen)
        model.train()
        if frozen:
            model.encoder.eval()  # keep BatchNorm statistics fixed while the backbone is frozen
        t0, losses = time.time(), []
        for x, ys in loaders["train"]:
            scale = min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))
            for g, lr in zip(optimizer.param_groups, base_lrs):
                g["lr"] = lr * scale
            out = model(x.to(device))
            loss = sum(criteria[t](out[t], ys[t].to(device)) for t in classes)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            step += 1
        val = evaluate_scores(predict(model, loaders["val"], device), parts["val"], classes)
        score = float(np.mean([val[t]["macro_f1"] for t in classes]))
        history.append({"epoch": epoch + 1, "frozen_backbone": frozen, "train_loss": float(np.mean(losses)),
                        "val_mean_macro_f1": score, "seconds": round(time.time() - t0, 1),
                        **{f"val_{t}_acc": val[t]["accuracy"] for t in classes}})
        print(json.dumps(history[-1]), flush=True)
        if score > best["score"]:
            best = {"score": score, "epoch": epoch + 1,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    ckpt = paths.CHECKPOINTS_DIR / cfg.dataset / f"{cfg.run_name}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(cfg), "classes": classes, "state_dict": best["state"]}, ckpt)

    scores = {s: predict(model, loaders[s], device) for s in ("val", "test")}
    results = {"config": asdict(cfg), "classes": classes, "best_epoch": best["epoch"], "history": history,
               "val": evaluate_scores(scores["val"], parts["val"], classes),
               "test": evaluate_scores(scores["test"], parts["test"], classes), "checkpoint": str(ckpt.relative_to(paths.OUTPUTS_DIR))}
    if {"brand", "era"} <= set(classes):
        te = parts["test"]
        y = {t: te[t].map({c: i for i, c in enumerate(classes[t])}).to_numpy() for t in ("brand", "era")}
        b_ok = scores["test"]["brand"].argmax(1) == y["brand"]
        e_ok = scores["test"]["era"].argmax(1) == y["era"]
        results["test"]["joint"] = {
            "both_correct": float((b_ok & e_ok).mean()),
            "era_given_brand_correct": float(e_ok[b_ok].mean()),
            "era_accuracy_by_brand": evaluation.era_accuracy_by_brand(y["brand"], y["era"], e_ok, classes["brand"]),
        }

    out_dir = paths.RESULTS_DIR / cfg.dataset
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame({"image_id": parts["test"].image_id})
    for t, names in classes.items():
        pred[f"{t}_true"] = parts["test"][t]
        pred[f"{t}_pred"] = np.asarray(names)[scores["test"][t].argmax(1)]
        pred[f"{t}_conf"] = scores["test"][t].max(1)
    pred.to_csv(out_dir / "predictions" / f"finetune_{cfg.run_name}.csv", index=False)
    (out_dir / f"finetune_{cfg.run_name}.json").write_text(json.dumps(results, indent=2))
    return results
