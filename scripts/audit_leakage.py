"""Leakage audit: do val/test images have exact or near-duplicate copies in the training split?

python scripts/audit_leakage.py --dataset fashiongen
Counts val/test images with (a) a byte-identical file, (b) a near-duplicate thumbnail (the criterion
used to merge groups), or (c) a CLIP ViT-B/16 neighbour with cosine >= 0.99 in train.
Writes outputs/results/<dataset>/leakage_audit.json.
"""

import argparse
import hashlib
import json

import numpy as np

from thriftfindr import features, paths
from thriftfindr.data import manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--clip-threshold", type=float, default=0.99)
    args = parser.parse_args()

    df = manifest.load(args.dataset)
    root = paths.dataset_dir(args.dataset)
    split = df.split.to_numpy()
    train = split == "train"

    md5 = np.array([hashlib.md5((root / p).read_bytes()).hexdigest() for p in df.path])
    train_md5 = set(md5[train])
    near = manifest.duplicate_pairs(manifest.thumbnails([root / p for p in df.path]))
    near_hit = np.zeros(len(df), bool)
    for i, j in near:
        if train[i] != train[j]:
            near_hit[j if train[i] else i] = True

    X = features.load(args.dataset, "clip_vitb16", df)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    clip_hit = np.zeros(len(df), bool)
    held = np.flatnonzero(~train)
    for s in range(0, len(held), 2048):
        idx = held[s:s + 2048]
        clip_hit[idx] = (X[idx] @ X[train].T).max(axis=1) >= args.clip_threshold

    out = {}
    for s in ("val", "test"):
        m = split == s
        out[s] = {"n": int(m.sum()),
                  "exact_copy_in_train": int(sum(h in train_md5 for h in md5[m])),
                  "near_duplicate_in_train": int(near_hit[m].sum()),
                  f"clip_cos_ge_{args.clip_threshold}_in_train": int(clip_hit[m].sum())}
    path = paths.RESULTS_DIR / args.dataset / "leakage_audit.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
