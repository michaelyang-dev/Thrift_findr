"""Build the europeana_runway toy dataset: Catwalk Pictures runway photos of archive designers x decade.

Searches are cached in data/raw/europeana_runway/search/ and images already on disk are reused,
so re-runs are cheap. Set EUROPEANA_API_KEY to use your own key instead of the public demo key.
"""

import argparse
import json

from thriftfindr.data import europeana, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cap", type=int, default=400, help="max images per (brand, era) class")
    parser.add_argument("--per-show", type=int, default=25, help="max images per runway show")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, europeana.MAX_WORKERS + 1),
                        help="parallel image downloads")
    parser.add_argument("--refresh", action="store_true", help="re-run the API searches instead of using the cache")
    args = parser.parse_args()

    df = europeana.build(cap=args.cap, per_show=args.per_show, seed=args.seed, workers=args.workers,
                         refresh=args.refresh)
    print(f"wrote {manifest.save(df, europeana.NAME)}")
    print(json.dumps(manifest.summarize(df), indent=2))
    problems = manifest.validate(df)
    print("validate:", problems)
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
