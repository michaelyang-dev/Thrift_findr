"""Build the Fashion-Gen toy dataset into data/processed/fashiongen/ (downloads ~16 GB on first run)."""

import argparse
import json

from thriftfindr.data import fashiongen, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-images", type=int, default=50, help="drop (brand, era) classes smaller than this")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = fashiongen.build(min_images=args.min_images, seed=args.seed)
    path = manifest.save(df, fashiongen.NAME)
    print(json.dumps(manifest.summarize(df), indent=2))
    print("validation:", manifest.validate(df) or "OK")
    print("manifest:", path)


if __name__ == "__main__":
    main()
