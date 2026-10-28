"""OCR brand matching on label photos, alone and fused with the CLIP linear probe.

python scripts/run_ocr.py --dataset secondhand_labels
The match threshold is chosen on the validation split; results go to outputs/results/<dataset>/ocr.json.
"""

import argparse
import json

import numpy as np
import pandas as pd

from thriftfindr import baselines, evaluation, features, ocr, paths
from thriftfindr.data import manifest
from thriftfindr.data.dataset import label_maps

THRESHOLDS = (0.6, 0.7, 0.8, 0.9, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="secondhand_labels")
    parser.add_argument("--backbone", default="clip_vitb16")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    df = manifest.load(args.dataset)
    brands = label_maps(df, ["brand"])["brand"]
    y = df.brand.map({b: i for i, b in enumerate(brands)}).to_numpy()
    m = {s: (df.split == s).to_numpy() for s in manifest.SPLITS}
    root = paths.dataset_dir(args.dataset)
    out_dir = paths.RESULTS_DIR / args.dataset

    # Image model: the linear probe with the C chosen in run_baselines.py.
    base = json.loads((out_dir / "baselines.json").read_text())
    C = base["results"][args.backbone]["brand"]["linear_probe"]["param"]
    X = features.load(args.dataset, args.backbone, df)
    probe = baselines.fit_linear_probe(X[m["train"]], y[m["train"]], X[m["val"]], y[m["val"]], Cs=(C,))
    image_pred = {s: baselines.probe_scores(probe, X[m[s]]).argmax(1) for s in ("val", "test")}

    texts = {}
    for s in ("val", "test"):
        rows = df[m[s]]
        texts[s] = ocr.read_labels([root / p for p in rows.path], out_dir / f"ocr_text_{s}.json", gpu=not args.cpu)

    def run(split: str, threshold: float):
        rows = df[m[split]]
        ocr_pred = np.full(len(rows), -1)
        for i, p in enumerate(rows.path):
            b, _ = ocr.match(texts[split][p.split("/")[-1]], brands, threshold)
            ocr_pred[i] = -1 if b is None else brands.index(b)
        fused = np.where(ocr_pred >= 0, ocr_pred, image_pred[split])
        return ocr_pred, fused

    best_t = max(THRESHOLDS, key=lambda t: (run("val", t)[1] == y[m["val"]]).mean())
    ocr_pred, fused = run("test", best_t)
    yt, groups = y[m["test"]], df.group_id.to_numpy()[m["test"]]
    covered = ocr_pred >= 0
    result = {
        "threshold": best_t,
        "ocr_coverage": float(covered.mean()),
        "ocr_precision": float((ocr_pred[covered] == yt[covered]).mean()) if covered.any() else None,
        "ocr_only_accuracy": float((ocr_pred == yt).mean()),
        "image_only": evaluation.classification_metrics(yt, image_pred["test"], len(brands), groups=groups),
        "ocr_then_image": evaluation.classification_metrics(yt, fused, len(brands), groups=groups),
        "image_accuracy_where_ocr_confident": float((image_pred["test"][covered] == yt[covered]).mean()),
        "image_accuracy_where_ocr_abstains": float((image_pred["test"][~covered] == yt[~covered]).mean()),
    }
    # Same fusion with each fine-tuned model's saved test predictions (threshold still chosen on val).
    test_ids = df.image_id[m["test"]].to_numpy()
    for f in sorted((out_dir / "predictions").glob("finetune_*.csv")):
        pred = pd.read_csv(f, dtype={"image_id": str}).set_index("image_id").loc[test_ids, "brand_pred"]
        ft_pred = pred.map({b: i for i, b in enumerate(brands)}).to_numpy()
        fused_ft = np.where(covered, ocr_pred, ft_pred)
        result.setdefault("ocr_then_finetuned", {})[f.stem.removeprefix("finetune_")] = {
            "image_only": evaluation.classification_metrics(yt, ft_pred, len(brands), groups=groups),
            "ocr_then_image": evaluation.classification_metrics(yt, fused_ft, len(brands), groups=groups)}
    (out_dir / "ocr.json").write_text(json.dumps(result, indent=2))
    print(f"threshold {best_t}: OCR covers {result['ocr_coverage']:.3f} of test images with precision "
          f"{result['ocr_precision']:.3f}; OCR-only acc {result['ocr_only_accuracy']:.3f}")
    print(f"image only ({args.backbone} probe) acc {result['image_only']['accuracy']:.3f}; "
          f"OCR-then-image acc {result['ocr_then_image']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
