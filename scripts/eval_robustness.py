"""Accuracy under phone-photo corruptions for fine-tuned checkpoints and frozen linear probes.

python scripts/eval_robustness.py --dataset fashiongen \
    --checkpoints outputs/checkpoints/fashiongen/dinov2_vits14_strong.pt outputs/checkpoints/fashiongen/dinov2_vits14_light.pt \
    --frozen-backbones dinov2_vits14
Results are merged into outputs/results/<dataset>/robustness.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from thriftfindr import paths, robustness
from thriftfindr.data import manifest, registry
from thriftfindr.data.transforms import CORRUPTIONS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--frozen-backbones", nargs="*", default=[])
    args = parser.parse_args()

    # Merge with earlier runs only if they used the same manifest (same test images and labels).
    out_path = paths.RESULTS_DIR / args.dataset / "robustness.json"
    fp = manifest.fingerprint(manifest.load(args.dataset), registry.get(args.dataset).tasks)
    saved = json.loads(out_path.read_text()) if out_path.exists() else {}
    results = saved.get("models", {}) if saved.get("manifest_fingerprint") == fp else {}
    for ckpt in args.checkpoints:
        results[f"finetune_{Path(ckpt).stem}"] = robustness.evaluate_checkpoint(ckpt)
    if args.frozen_backbones:
        base = json.loads((paths.RESULTS_DIR / args.dataset / "baselines.json").read_text())
        for backbone in args.frozen_backbones:
            Cs = {t: r["linear_probe"]["param"] for t, r in base["results"][backbone].items() if t != "joint"}
            results[f"frozen_probe_{backbone}"] = robustness.evaluate_frozen_probe(args.dataset, backbone, Cs)
    out_path.write_text(json.dumps({"manifest_fingerprint": fp, "models": results}, indent=2))

    tasks = list(next(iter(results.values()))["clean"]["0"])
    for task in tasks:
        print(f"\n{task} accuracy (mean over severities 1-3)")
        print(f"{'model':40s} {'clean':>6s} " + " ".join(f"{c:>11s}" for c in CORRUPTIONS))
        for name, r in results.items():
            row = [np.mean([r[c][s][task] for s in r[c]]) for c in CORRUPTIONS]
            print(f"{name:40s} {r['clean']['0'][task]:6.3f} " + " ".join(f"{v:11.3f}" for v in row))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
