"""Shortcut check: can low-level photo statistics alone predict the label?

If an 8x8 thumbnail, or colour statistics of the image border (mostly studio background),
predicts era far above chance, the era signal is partly photography style rather than garment design.

python scripts/analyze_shortcuts.py --dataset fashiongen
"""

import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from thriftfindr import baselines, evaluation, paths
from thriftfindr.data import manifest, registry
from thriftfindr.data.dataset import label_maps

BORDER = 16  # pixels, at 256x256


def photo_stats(path) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(path).convert("RGB").resize((256, 256), Image.BILINEAR)
    thumb = np.asarray(img.resize((8, 8), Image.BOX), dtype=np.float32).ravel() / 255.0
    a = np.asarray(img, dtype=np.float32) / 255.0
    mask = np.ones(a.shape[:2], bool)
    mask[BORDER:-BORDER, BORDER:-BORDER] = False
    border = a[mask]
    hist = np.concatenate([np.histogram(border[:, c], bins=16, range=(0, 1))[0] / len(border) for c in range(3)])
    return thumb, np.concatenate([border.mean(0), border.std(0), hist])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    spec = registry.get(args.dataset)
    df = manifest.load(args.dataset)
    root = paths.dataset_dir(args.dataset)
    stats = [photo_stats(root / p) for p in tqdm(df.path, desc="photo stats")]
    feats = {"thumbnail_8x8": np.stack([s[0] for s in stats]), "border_colour": np.stack([s[1] for s in stats])}
    classes = label_maps(df, spec.tasks)
    m = {s: (df.split == s).to_numpy() for s in manifest.SPLITS}

    out = {"chance": {}, "features": {}}
    for task, names in classes.items():
        y = df[task].map({c: i for i, c in enumerate(names)}).to_numpy()
        majority = np.bincount(y[m["train"]]).argmax()
        out["chance"][task] = {"majority_class_test_acc": float((y[m["test"]] == majority).mean()),
                               "uniform": 1.0 / len(names)}
        for fname, X in feats.items():
            # Unweighted probe, so its accuracy is comparable with the majority-class accuracy.
            fitted = baselines.fit_linear_probe(X[m["train"]], y[m["train"]], X[m["val"]], y[m["val"]],
                                                class_weight=None)
            scores = baselines.probe_scores(fitted, X[m["test"]])
            r = evaluation.classification_metrics(y[m["test"]], scores.argmax(1), len(names), scores,
                                                  df.group_id.to_numpy()[m["test"]])
            out["features"].setdefault(fname, {})[task] = {
                k: r[k] for k in ("accuracy", "macro_f1", "balanced_accuracy", "accuracy_ci95")}
            print(f"{task:6s} {fname:14s} test acc {r['accuracy']:.3f}  macro-F1 {r['macro_f1']:.3f}  "
                  f"(majority {out['chance'][task]['majority_class_test_acc']:.3f})")

    if "era" in classes and "season" in df:
        pose = df.image_id.str.rsplit("_", n=1).str[-1]
        out["pose_share_by_era"] = pd.crosstab(df.era, pose, normalize="index").round(3).to_dict(orient="index")
    path = paths.RESULTS_DIR / args.dataset / "shortcuts.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
