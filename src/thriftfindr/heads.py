"""Train only the task heads (and the FiLM input conditioning) on cached frozen-backbone features.

The same MultiTaskModel as fine-tuning, with the encoder replaced by the cached features, so a run takes
minutes instead of hours and the checkpoint works with scripts/predict.py (which re-attaches the
pretrained encoder). No augmentation: every image has one cached feature vector.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from thriftfindr import features, paths
from thriftfindr.data import manifest, registry
from thriftfindr.data.dataset import label_maps
from thriftfindr.finetune import MultiTaskModel, balanced_weights, evaluate_scores, selection_score
from thriftfindr.models.backbones import BACKBONES


@dataclass
class HeadsConfig:
    dataset: str
    backbone: str = "clip_vitb16"
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    regress_weight: float = 1.0
    ignore_inputs: bool = False  # ablation: image only, without the dataset's input columns (e.g. category)
    seed: int = 0
    tag: str = ""
    frozen_backbone: bool = True  # marks the checkpoint for load_checkpoint; not a setting

    @property
    def run_name(self) -> str:
        default = HeadsConfig(dataset=self.dataset)
        parts = ["heads", self.backbone]
        parts += [f"{f}{getattr(self, f)}" for f in ("epochs", "batch_size", "lr", "weight_decay", "label_smoothing",
                                                     "regress_weight", "seed")
                  if getattr(self, f) != getattr(default, f)]
        if self.ignore_inputs:
            parts.append("noinputs")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def _tensors(df: pd.DataFrame, X: np.ndarray, columns: dict, regress) -> tuple[torch.Tensor, dict]:
    ys = {t: torch.as_tensor(df[t].map({c: i for i, c in enumerate(names)}).to_numpy().copy())
          for t, names in columns.items()}
    for t in regress:
        v = pd.to_numeric(df[t], errors="coerce").to_numpy(dtype=float)
        ys[t] = torch.as_tensor(np.log(np.where(v > 0, v, np.nan)), dtype=torch.float32)
    return torch.as_tensor(X), ys


@torch.inference_mode()
def _predict(model: MultiTaskModel, X, ys, stats, batch_size: int = 1024) -> dict[str, np.ndarray]:
    model.eval()
    scores: dict[str, list] = {}
    for s in range(0, len(X), batch_size):
        out = model(X[s:s + batch_size], {c: ys[c][s:s + batch_size] for c in model.inputs})
        for t, o in out.items():
            v = np.exp(stats[t]["mean"] + stats[t]["std"] * o.numpy()) if t in model.regress else o.softmax(-1).numpy()
            scores.setdefault(t, []).append(v)
    return {t: np.concatenate(v) for t, v in scores.items()}


def train(cfg: HeadsConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    spec = registry.get(cfg.dataset)
    df = manifest.load(cfg.dataset)
    classes = label_maps(df, spec.tasks)
    inputs = {} if cfg.ignore_inputs else label_maps(df, spec.inputs)
    regress = spec.regress
    parts = {s: df[df.split == s].reset_index(drop=True) for s in manifest.SPLITS}
    data = {s: _tensors(p, features.load(cfg.dataset, cfg.backbone, p), {**classes, **inputs}, regress)
            for s, p in parts.items()}
    X_train, y_train = data["train"]
    stats = {}
    for t in regress:
        y = y_train[t].numpy()
        y = y[np.isfinite(y)]
        stats[t] = {"mean": float(y.mean()), "std": float(y.std())}

    model = MultiTaskModel(cfg.backbone, {t: len(c) for t, c in classes.items()}, regress,
                           {c: len(v) for c, v in inputs.items()}, encoder=nn.Identity())
    criteria = {t: nn.CrossEntropyLoss(weight=balanced_weights(y_train[t].numpy(), len(c)),
                                       label_smoothing=cfg.label_smoothing) for t, c in classes.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(X_train) / cfg.batch_size)
    total = cfg.epochs * steps_per_epoch

    history, best, step = [], {"score": -1.0}, 0
    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for idx in torch.randperm(len(X_train)).split(cfg.batch_size):
            for g in optimizer.param_groups:
                g["lr"] = cfg.lr * 0.5 * (1 + math.cos(math.pi * step / total))
            ys = {t: v[idx] for t, v in y_train.items()}
            out = model(X_train[idx], {c: ys[c] for c in inputs})
            loss = sum(criteria[t](out[t], ys[t]) for t in classes)
            for t in regress:
                ok = torch.isfinite(ys[t])
                z = (ys[t][ok] - stats[t]["mean"]) / stats[t]["std"]
                loss = loss + cfg.regress_weight * nn.functional.smooth_l1_loss(out[t][ok], z)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1
        val = evaluate_scores(_predict(model, *data["val"], stats), parts["val"], classes, regress)
        score = selection_score(val, classes, regress)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_score": score,
                        **{f"val_{t}_acc": val[t]["accuracy"] for t in classes},
                        **{f"val_{t}_mdape_pct": val[t]["mdape_pct"] for t in regress}})
        if score > best["score"]:
            best = {"score": score, "epoch": epoch + 1,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    ckpt = paths.CHECKPOINTS_DIR / cfg.dataset / f"{cfg.run_name}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(cfg), "classes": classes, "inputs": inputs, "regress": stats,
                "state_dict": best["state"]}, ckpt)

    scores = {s: _predict(model, *data[s], stats) for s in ("val", "test")}
    results = {"config": asdict(cfg), "classes": classes, "inputs": inputs, "regress": stats,
               "manifest_fingerprint": manifest.fingerprint(df, (*spec.tasks, *spec.inputs, *spec.regress)),
               "best_epoch": best["epoch"], "history": history,
               "val": evaluate_scores(scores["val"], parts["val"], classes, regress),
               "test": evaluate_scores(scores["test"], parts["test"], classes, regress),
               "checkpoint": str(ckpt.relative_to(paths.OUTPUTS_DIR))}

    out_dir = paths.RESULTS_DIR / cfg.dataset
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame({"image_id": parts["test"].image_id})
    for c in inputs:
        pred[c] = parts["test"][c]
    for t, names in classes.items():
        pred[f"{t}_true"] = parts["test"][t]
        pred[f"{t}_pred"] = np.asarray(names)[scores["test"][t].argmax(1)]
        pred[f"{t}_conf"] = scores["test"][t].max(1)
    for t in regress:
        pred[f"{t}_true"] = parts["test"][t]
        pred[f"{t}_pred"] = scores["test"][t].round(2)
    pred.to_csv(out_dir / "predictions" / f"{cfg.run_name}.csv", index=False)
    (out_dir / f"{cfg.run_name}.json").write_text(json.dumps(results, indent=2))
    return results
