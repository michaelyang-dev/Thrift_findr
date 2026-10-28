"""Frozen-feature baselines (linear probe + kNN) for every task of a dataset, plus price estimation.

python scripts/run_baselines.py --dataset fashiongen
Writes outputs/results/<dataset>/baselines.json and test-set predictions per backbone and method.
"""

import argparse
import json

import numpy as np
import pandas as pd

from thriftfindr import baselines, evaluation, features, paths, price
from thriftfindr.data import manifest, registry
from thriftfindr.data.dataset import label_maps
from thriftfindr.models.backbones import BACKBONES


def run_task(X, y, n_classes, masks, groups):
    """Fit both baselines on train, select on val, score val and test. Returns metrics and test scores."""
    tr, va, te = masks["train"], masks["val"], masks["test"]
    results, test_scores = {}, {}
    for fitted in (baselines.fit_linear_probe(X[tr], y[tr], X[va], y[va]),
                   baselines.fit_knn(X[tr], y[tr], X[va], y[va], n_classes)):
        res = {"param": fitted.param}
        for split, m in (("val", va), ("test", te)):
            scores = baselines.predict_scores(fitted, X[m])
            # kNN votes tie at zero for most classes, so its top-3 would depend on class order: skip it.
            top3_scores = scores if fitted.method == "linear_probe" else None
            res[split] = evaluation.classification_metrics(y[m], scores.argmax(1), n_classes, top3_scores, groups[m])
            if split == "test":
                test_scores[fitted.method] = scores
        results[fitted.method] = res
    return results, test_scores


def joint_metrics(y_brand, y_era, pred_brand, pred_era, brand_names) -> dict:
    b_ok, e_ok = pred_brand == y_brand, pred_era == y_era
    return {
        "both_correct": float((b_ok & e_ok).mean()),
        "era_given_brand_correct": float(e_ok[b_ok].mean()),
        "era_accuracy_by_brand": evaluation.era_accuracy_by_brand(y_brand, y_era, e_ok, brand_names),
    }


def run_price(df, X, masks, price_col, brand_pred_test=None) -> dict:
    p = df[price_col].to_numpy(dtype=float)
    ok = np.isfinite(p) & (p > 0)
    tr, va, te = (masks[s] & ok for s in ("train", "val", "test"))
    out = {}
    k = price.fit_knn_price(X[tr], p[tr], X[va], p[va])
    out["knn"] = {"param": k, "test": evaluation.price_metrics(p[te], price.knn_price(X[tr], p[tr], X[te], k))}
    alpha, ridge = price.fit_ridge_price(X[tr], p[tr], X[va], p[va])
    out["ridge"] = {"param": alpha, "test": evaluation.price_metrics(p[te], ridge(X[te]))}
    if brand_pred_test is not None:
        pred = brand_pred_test[ok[masks["test"]]]
        out["predicted_brand_median"] = {
            "test": evaluation.price_metrics(p[te], price.median_lookup(df.brand[tr], p[tr], pred))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=list(BACKBONES))
    args = parser.parse_args()

    spec = registry.get(args.dataset)
    df = manifest.load(args.dataset)
    classes = label_maps(df, spec.tasks)
    y = {t: df[t].map({c: i for i, c in enumerate(names)}).to_numpy() for t, names in classes.items()}
    masks = {s: (df.split == s).to_numpy() for s in manifest.SPLITS}
    groups = df.group_id.to_numpy()
    out_dir = paths.RESULTS_DIR / args.dataset
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    # Merge with earlier runs so backbones can be evaluated one at a time, but only if they were computed
    # on the same manifest (same images, splits and labels).
    out_path = out_dir / "baselines.json"
    fingerprint = manifest.fingerprint(df, spec.tasks)
    report = json.loads(out_path.read_text()) if out_path.exists() else {}
    if report.get("manifest_fingerprint") != fingerprint:
        if report:
            print("manifest changed since the last run: discarding earlier baseline results")
        report = {"results": {}, "price": {}}
    report.update(dataset=args.dataset, classes=classes, manifest_fingerprint=fingerprint)
    for backbone in args.backbones:
        X = features.load(args.dataset, backbone, df)
        res, test_scores = {}, {}
        for task, names in classes.items():
            res[task], test_scores[task] = run_task(X, y[task], len(names), masks, groups)
        methods = list(test_scores[spec.tasks[0]])
        if {"brand", "era"} <= set(spec.tasks):
            te = masks["test"]
            res["joint"] = {m: joint_metrics(y["brand"][te], y["era"][te], test_scores["brand"][m].argmax(1),
                                             test_scores["era"][m].argmax(1), classes["brand"]) for m in methods}
        report["results"][backbone] = res

        for m in methods:
            pred = pd.DataFrame({"image_id": df.image_id[masks["test"]].to_numpy()})
            for task, names in classes.items():
                pred[f"{task}_true"] = df[task][masks["test"]].to_numpy()
                pred[f"{task}_pred"] = np.asarray(names)[test_scores[task][m].argmax(1)]
                pred[f"{task}_conf"] = test_scores[task][m].max(1)
            pred.to_csv(out_dir / "predictions" / f"{backbone}_{m}.csv", index=False)

        if spec.price_col:
            brand_pred = (np.asarray(classes["brand"])[test_scores["brand"]["linear_probe"].argmax(1)]
                          if "brand" in classes else None)
            report["price"][backbone] = run_price(df, X, masks, spec.price_col, brand_pred)

    if spec.price_col:
        p = df[spec.price_col].to_numpy(dtype=float)
        ok = np.isfinite(p) & (p > 0)
        tr, te = masks["train"] & ok, masks["test"] & ok
        report["price"]["no_image"] = {
            "global_median": {"test": evaluation.price_metrics(p[te], np.full(te.sum(), np.median(p[tr])))},
            "true_brand_median": {"test": evaluation.price_metrics(
                p[te], price.median_lookup(df.brand[tr], p[tr], df.brand[te]))},
        }

    out_path.write_text(json.dumps(report, indent=2))
    print(f"{'backbone':14s} {'task':6s} {'method':13s} {'param':>7s} {'val acc':>8s} {'test acc':>9s} "
          f"{'95% CI':>15s} {'test F1':>8s}")
    for backbone, res in report["results"].items():
        for task in spec.tasks:
            for m, r in res[task].items():
                ci = r["test"]["accuracy_ci95"]
                print(f"{backbone:14s} {task:6s} {m:13s} {r['param']:>7g} {r['val']['accuracy']:8.3f} "
                      f"{r['test']['accuracy']:9.3f} [{ci[0]:.3f}, {ci[1]:.3f}] {r['test']['macro_f1']:8.3f}")
        for m, j in res.get("joint", {}).items():
            print(f"{backbone:14s} joint  {m:13s} both correct {j['both_correct']:.3f}; "
                  f"era | brand correct {j['era_given_brand_correct']:.3f}")
    for backbone, pr in report["price"].items():
        for m, r in pr.items():
            t = r["test"]
            print(f"price {backbone:14s} {m:24s} MdAPE {t['mdape_pct']:6.1f}%  MAE ${t['mae']:7.0f}  "
                  f"within 25% {t['within_25pct']:.3f}  Spearman {t['spearman']:.3f}")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
