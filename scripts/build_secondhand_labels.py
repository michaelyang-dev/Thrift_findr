"""Build the secondhand_labels toy dataset: brand-label photos from the Second-Hand Fashion dataset.

Reads only the needed byte ranges of the Hugging Face parquet repack (~6 GB of its 26.6 GB by default).
The metadata scan is cached in data/raw/secondhand_labels/ and images already on disk are reused,
so re-runs are cheap.
"""

import argparse
import json

from thriftfindr.data import manifest, secondhand


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-gb", type=float, default=6.0, help="budget for downloaded label-photo chunks")
    parser.add_argument("--min-images", type=int, default=100, help="drop brands with fewer usable label photos")
    parser.add_argument("--max-per-brand", type=int, default=300, help="seeded random cap per brand")
    parser.add_argument("--max-classes", type=int, default=15)
    parser.add_argument("--max-side", type=int, default=1024, help="resize so the longer side is at most this")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, secondhand.MAX_WORKERS + 1),
                        help="parallel HTTP range requests")
    args = parser.parse_args()

    df = secondhand.build(max_gb=args.max_gb, min_images=args.min_images, max_per_brand=args.max_per_brand,
                          max_classes=args.max_classes, max_side=args.max_side, seed=args.seed,
                          workers=args.workers)
    print(f"wrote {manifest.save(df, secondhand.NAME)}")
    print(json.dumps(manifest.summarize(df), indent=2))
    problems = manifest.validate(df)
    print("validate:", problems)
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
