"""Manifest format shared by every dataset builder and by the training / evaluation code.

A manifest is a CSV with one row per image, stored at data/processed/<dataset>/manifest.csv.
Required columns:

    image_id   unique id within the dataset
    path       image path relative to the dataset directory (data/processed/<dataset>/)
    dataset    dataset name
    brand      canonical brand / house label ("" if the dataset has no brand task)
    era        era label, e.g. "2016" or "2000s" ("" if the dataset has no era task)
    group_id   grouping key for leakage-free splits (product, garment, or runway show):
               all images of one group must land in the same split
    split      train / val / test

Optional columns used when present: subline, year, season, price, price_band, category,
source_url, license.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

from thriftfindr import paths

REQUIRED_COLUMNS = ["image_id", "path", "dataset", "brand", "era", "group_id", "split"]
OPTIONAL_COLUMNS = ["subline", "year", "season", "price", "price_band", "category", "source_url", "license"]
SPLITS = ("train", "val", "test")
_STR_COLUMNS = {c: str for c in ["image_id", "path", "dataset", "brand", "era", "group_id", "split",
                                 "subline", "season", "price_band", "category", "source_url", "license"]}


def joint_label(df: pd.DataFrame, cols) -> pd.Series:
    """Join several label columns into one string key, e.g. brand|era."""
    return df[list(cols)].astype(str).agg("|".join, axis=1)


def assign_splits(df: pd.DataFrame, group_col: str = "group_id", strata_cols=("brand", "era"),
                  val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0) -> pd.Series:
    """Assign train/val/test so every group lands in exactly one split.

    Groups are split with stratified group k-fold: test is one of round(1/test_frac) folds and val one
    of round((1-test_frac)/val_frac) folds of the rest, i.e. 5:1:1 (71/14/14%) for the defaults.
    Classes with fewer groups than folds would often get no val or test group that way, so their groups
    are dealt out directly (one to test, one to val, the rest to train); with drop_rare's
    min_groups >= 3 every class then appears in all three splits.
    """
    strata = joint_label(df, strata_cols).to_numpy()
    groups = df[group_col].astype(str).to_numpy()
    split = np.full(len(df), "train", dtype=object)
    n_test_folds = max(2, round(1 / test_frac))

    rng = np.random.default_rng(seed)
    dealt: dict[str, str] = {}
    for _, gs in pd.Series(groups).groupby(strata).unique().items():
        if len(gs) < n_test_folds:
            for k, g in enumerate(rng.permutation(sorted(gs))):
                dealt.setdefault(g, ("test", "val")[k] if k < 2 else "train")
    is_dealt = np.isin(groups, list(dealt))
    split[is_dealt] = [dealt[g] for g in groups[is_dealt]]
    rest = np.flatnonzero(~is_dealt)

    sgkf = StratifiedGroupKFold(n_splits=n_test_folds, shuffle=True, random_state=seed)
    keep, test_idx = next(sgkf.split(np.zeros(len(rest)), strata[rest], groups[rest]))
    split[rest[test_idx]] = "test"
    sgkf = StratifiedGroupKFold(n_splits=max(2, round((1 - test_frac) / val_frac)), shuffle=True,
                                random_state=seed + 1)
    _, val_idx = next(sgkf.split(np.zeros(len(keep)), strata[rest[keep]], groups[rest[keep]]))
    split[rest[keep[val_idx]]] = "val"
    return pd.Series(split, index=df.index, name="split")


def thumbnails(paths_: list[Path], size: int = 32) -> np.ndarray:
    """size x size grayscale thumbnail of every image, flattened (uint8)."""
    return np.stack([np.asarray(Image.open(p).convert("L").resize((size, size), Image.BILINEAR)).ravel()
                     for p in paths_])


def duplicate_pairs(thumbs: np.ndarray, min_corr: float = 0.99, max_mean_diff: float = 3.0,
                    chunk: int = 2048) -> np.ndarray:
    """Index pairs (i < j) of exact or near-duplicate images. Thumbnails correlated at >= min_corr are
    candidates; a pair is kept if the thumbnails also differ by at most max_mean_diff grey levels.
    On Fashion-Gen the defaults catch 99.6% of cross-product pairs with CLIP cosine >= 0.99 while the
    largest merged group stays at 6 products (looser settings chain look-alike studio shots)."""
    t = thumbs.astype(np.float32)
    t -= t.mean(axis=1, keepdims=True)
    t /= np.linalg.norm(t, axis=1, keepdims=True).clip(min=1e-6)
    found = []
    for s in range(0, len(t), chunk):
        i, j = np.nonzero(t[s:s + chunk] @ t.T >= min_corr)
        i += s
        i, j = i[i < j], j[i < j]
        diff = np.abs(thumbs[i].astype(np.int16) - thumbs[j].astype(np.int16)).mean(axis=1)
        found.append(np.stack([i, j], axis=1)[diff <= max_mean_diff])
    return np.concatenate(found) if found else np.empty((0, 2), dtype=int)


def merge_duplicate_groups(df: pd.DataFrame, root: Path, **kwargs) -> pd.Series:
    """group_id after merging every group that shares an exact or near-duplicate image with another,
    so the same photo filed under two products can never straddle splits. A merged group is named
    after its smallest member id."""
    pairs = duplicate_pairs(thumbnails([Path(root) / p for p in df.path]), **kwargs)
    codes, names = pd.factorize(df.group_id.astype(str))
    a, b = codes[pairs[:, 0]], codes[pairs[:, 1]]
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(names), len(names)))
    _, comp = connected_components(graph, directed=False)
    merged = pd.Series(names).groupby(comp).transform("min").to_numpy()
    return pd.Series(merged[codes], index=df.index, name="group_id")


def drop_rare(df: pd.DataFrame, cols, min_images: int = 50, min_groups: int = 3) -> pd.DataFrame:
    """Drop classes (joint values of cols) with too few images or groups to train and test on."""
    key = joint_label(df, cols)
    stats = df.groupby(key).agg(n=("image_id", "size"), g=("group_id", "nunique"))
    keep = stats.index[(stats.n >= min_images) & (stats.g >= min_groups)]
    return df[key.isin(keep)].copy()


def validate(df: pd.DataFrame, root: Path | None = None, check_files: bool = True) -> list[str]:
    """Return a list of problems; an empty list means the manifest is usable."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    problems = []
    if df.image_id.duplicated().any():
        problems.append(f"{df.image_id.duplicated().sum()} duplicate image_id values")
    bad = set(df.split) - set(SPLITS)
    if bad:
        problems.append(f"unknown split values: {sorted(bad)}")
    for s in SPLITS:
        if not (df.split == s).any():
            problems.append(f"empty split: {s}")
    leaked = df.groupby("group_id").split.nunique()
    if (leaked > 1).any():
        problems.append(f"{int((leaked > 1).sum())} groups appear in more than one split")
    if check_files:
        root = Path(root) if root is not None else paths.dataset_dir(df.dataset.iloc[0])
        absent = [p for p in df.path if not (root / p).is_file()]
        if absent:
            problems.append(f"{len(absent)} image files missing, e.g. {absent[:3]}")
    return problems


def summarize(df: pd.DataFrame) -> dict:
    out = {
        "n_images": int(len(df)),
        "n_groups": int(df.group_id.nunique()),
        "splits": {k: int(v) for k, v in df.split.value_counts().items()},
    }
    for col in ("brand", "era"):
        if col in df and (df[col] != "").any():
            out[f"{col}_by_split"] = {str(k): {s: int(n) for s, n in v.items()}
                                      for k, v in pd.crosstab(df[col], df.split).to_dict(orient="index").items()}
    if "brand" in df and "era" in df and (df.brand != "").any() and (df.era != "").any():
        out["brand_by_era"] = {str(k): {e: int(n) for e, n in v.items()}
                               for k, v in pd.crosstab(df.brand, df.era).to_dict(orient="index").items()}
    return out


def fingerprint(df: pd.DataFrame, tasks=()) -> str:
    """Short hash of image ids, splits and task labels: results computed on another manifest won't match."""
    import hashlib

    cols = ["image_id", "split", *tasks]
    return hashlib.sha1(df[cols].sort_values("image_id").to_csv(index=False).encode()).hexdigest()[:12]


def save(df: pd.DataFrame, name: str) -> Path:
    """Write the manifest (required columns first) and a stats.json summary next to it."""
    path = paths.manifest_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = REQUIRED_COLUMNS + [c for c in df.columns if c not in REQUIRED_COLUMNS]
    df[cols].to_csv(path, index=False)
    (path.parent / "stats.json").write_text(json.dumps(summarize(df), indent=2))
    return path


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(paths.manifest_path(name), dtype=_STR_COLUMNS, keep_default_na=False)
    for col in ("price", "year"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
