"""Extract and cache frozen-backbone features: python scripts/extract_features.py --dataset fashiongen"""

import argparse

from thriftfindr import features
from thriftfindr.models.backbones import BACKBONES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=list(BACKBONES))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for backbone in args.backbones:
        print(features.extract(args.dataset, backbone, batch_size=args.batch_size, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
