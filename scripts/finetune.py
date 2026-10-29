"""Fine-tune a backbone: python scripts/finetune.py --dataset fashiongen --backbone dinov2_vits14 --aug strong

Every FinetuneConfig field is a flag (e.g. --epochs 8 --lr-backbone 3e-5 --tag seed1).
"""

import argparse
from dataclasses import fields

from thriftfindr.data.transforms import AUG_STRENGTHS
from thriftfindr.finetune import FinetuneConfig, train
from thriftfindr.models.backbones import BACKBONES

CHOICES = {"aug": AUG_STRENGTHS, "backbone": tuple(BACKBONES)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    for f in fields(FinetuneConfig):
        flag = f"--{f.name.replace('_', '-')}"
        if f.name == "dataset":
            continue
        if isinstance(f.default, bool):
            parser.add_argument(flag, action="store_true")
        else:
            parser.add_argument(flag, type=type(f.default), default=f.default, choices=CHOICES.get(f.name))
    cfg = FinetuneConfig(**vars(parser.parse_args()))

    results = train(cfg)
    print(f"\n{cfg.run_name}: best epoch {results['best_epoch']}")
    for task in results["classes"]:
        r = results["test"][task]
        ci = r["accuracy_ci95"]
        print(f"  {task:8s} test acc {r['accuracy']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  macro-F1 {r['macro_f1']:.3f}")
    for col in results["regress"]:
        r = results["test"][col]
        print(f"  {col:8s} test median abs. error {r['mdape_pct']:.1f}%  within 25% {r['within_25pct']:.3f}")
    if "joint" in results["test"]:
        j = results["test"]["joint"]
        print(f"  joint  both correct {j['both_correct']:.3f}; era | brand correct {j['era_given_brand_correct']:.3f}")


if __name__ == "__main__":
    main()
