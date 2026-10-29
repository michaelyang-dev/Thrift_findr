"""Non-image reference points on the Fashion-Gen test split, fit on train: what the category alone (or
nothing) gets you. python scripts/reference_points.py -> outputs/results/fashiongen/reference_points.json

Also prints the manifest fingerprint, so a manifest built on another machine can be checked against
the one the results were computed on.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from thriftfindr import evaluation, paths
from thriftfindr.data import manifest, registry


def main() -> None:
    spec = registry.get("fashiongen")
    df = manifest.load("fashiongen")
    tr, te = df[df.split == "train"], df[df.split == "test"]
    out = {"manifest_fingerprint": manifest.fingerprint(df, (*spec.tasks, *spec.inputs, *spec.regress))}
    for t in spec.tasks:
        by_cat = tr.groupby("category")[t].agg(lambda s: s.mode()[0])
        for name, pred in (("majority", pd.Series(tr[t].mode()[0], index=te.index)),
                           ("majority_given_category", te.category.map(by_cat))):
            out[f"{t}/{name}"] = {"accuracy": float((pred == te[t]).mean()),
                                  "macro_f1": float(f1_score(te[t], pred, average="macro"))}
    key = lambda d: d.brand + "|" + d.category
    preds = {"global_median": np.full(len(te), tr.price.median()),
             "median_by_category": te.category.map(tr.groupby("category").price.median()).to_numpy(),
             # uses the true brand: a reference point for a perfect brand classifier, not a baseline
             "median_by_true_brand_and_category":
                 key(te).map(tr.price.groupby(key(tr)).median()).fillna(tr.price.median()).to_numpy()}
    for name, pred in preds.items():
        m = evaluation.price_metrics(te.price.to_numpy(), pred)
        out[f"price/{name}"] = {k: m[k] for k in ("mdape_pct", "within_25pct")}
    path = paths.RESULTS_DIR / "fashiongen" / "reference_points.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
