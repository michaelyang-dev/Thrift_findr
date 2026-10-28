"""Fashion-Gen toy dataset (Rostamzadeh et al. 2018): SSENSE catalog photos of archive houses.

Reads the 256x256 HDF5 files of the Hugging Face mirror hieupth/fashiongen, keeps clothing from
the target houses, and writes one JPEG per image plus a manifest.
era = season year (SS2014 ... SS2018); price = SSENSE retail list price (MSRP, USD), not resale.
"""

from __future__ import annotations

import html
import itertools
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from thriftfindr import paths
from thriftfindr.data import manifest

NAME = "fashiongen"
HF_REPO = "hieupth/fashiongen"
FILES = {"train": "fashiongen_256_256_train.h5", "validation": "fashiongen_256_256_validation.h5"}
LICENSE = "Fashion-Gen, non-commercial research use (Rostamzadeh et al. 2018); unofficial HF mirror"

# Fashion-Gen brand string -> (house, sub-line)
BRAND_MAP = {
    "Comme des Garçons": ("Comme des Garcons", "Comme des Garcons"),
    "Comme des Garçons Comme des Garçons": ("Comme des Garcons", "Comme des Garcons Comme des Garcons"),
    "Comme des Garçons Girl": ("Comme des Garcons", "Girl"),
    "Comme des Garçons Homme Plus": ("Comme des Garcons", "Homme Plus"),
    "Comme des Garçons Play": ("Comme des Garcons", "Play"),
    "Comme des Garçons Shirt": ("Comme des Garcons", "Shirt"),
    "Tricot Comme des Garçons": ("Comme des Garcons", "Tricot"),
    "Maison Margiela": ("Maison Margiela", "Maison Margiela"),
    "MM6 Maison Margiela": ("Maison Margiela", "MM6"),
    "Undercover": ("Undercover", "Undercover"),
    "John Undercover": ("Undercover", "John Undercover"),
    "Yohji Yamamoto": ("Yohji Yamamoto", "Yohji Yamamoto"),
    "Junya Watanabe": ("Junya Watanabe", "Junya Watanabe"),
    "Junya Watanabe Man Pink": ("Junya Watanabe", "Man Pink"),
    "Issey Miyake": ("Issey Miyake", "Issey Miyake"),
    "Issey Miyake Men": ("Issey Miyake", "Men"),
    "Homme Plisse Issey Miyake": ("Issey Miyake", "Homme Plisse"),
    "Pleats Please Issey Miyake": ("Issey Miyake", "Pleats Please"),
    "Helmut Lang": ("Helmut Lang", "Helmut Lang"),
    "Raf Simons": ("Raf Simons", "Raf Simons"),
    "Raf Simons/ Sterling Ruby": ("Raf Simons", "Sterling Ruby"),
    "Prada": ("Prada", "Prada"),
}


def download(raw_dir: Path | None = None) -> dict[str, Path]:
    """Fetch the two HDF5 files (~16 GB) unless they are already present."""
    from huggingface_hub import hf_hub_download

    raw_dir = Path(raw_dir or paths.RAW_DIR / NAME)
    out = {}
    for split, fname in FILES.items():
        if not (raw_dir / fname).exists():
            hf_hub_download(HF_REPO, fname, repo_type="dataset", local_dir=raw_dir)
        out[split] = raw_dir / fname
    return out


def _strings(ds) -> np.ndarray:
    return np.array([html.unescape(v[0].decode("latin-1")).strip() for v in ds[:]])


def _read_metadata(path: Path, source_split: str) -> pd.DataFrame:
    # Always open HDF5 files in a with-block: h5py segfaults at interpreter exit on unclosed files.
    with h5py.File(path, "r") as f:
        raw_brand = _strings(f["input_brand"])
        dept = _strings(f["input_department"])
        rows = np.flatnonzero(np.isin(raw_brand, list(BRAND_MAP)) & (dept == "CLOTHING"))
        cols = {k: _strings(f[f"input_{k}"])[rows]
                for k in ("season", "category", "subcategory", "gender", "pose", "name")}
        pid = f["input_productID"][:, 0][rows]
        msrp = f["input_msrpUSD"][:, 0][rows].astype(float)

    house, subline = zip(*(BRAND_MAP[b] for b in raw_brand[rows]))
    pose_num = [p.rsplit("_", 1)[-1] for p in cols["pose"]]
    return pd.DataFrame({
        "image_id": [f"{p}_{n}" for p, n in zip(pid, pose_num)],
        "path": [f"images/{p}_{n}.jpg" for p, n in zip(pid, pose_num)],
        "dataset": NAME,
        "brand": house,
        "era": [s[2:] for s in cols["season"]],  # "FW2016" -> "2016"
        "group_id": pid.astype(str),
        "subline": subline,
        "year": [int(s[2:]) for s in cols["season"]],
        "season": cols["season"],
        "price": np.where(msrp > 0, msrp, np.nan),
        "category": cols["category"],
        "subcategory": cols["subcategory"],
        "gender": cols["gender"],
        "name": cols["name"],
        "license": LICENSE,
        "source_split": source_split,
        "h5_row": rows,
    })


def _extract_images(path: Path, df: pd.DataFrame, out_dir: Path) -> None:
    """Decode the selected rows, reading whole 8-row HDF5 chunks so each chunk is decompressed once."""
    with h5py.File(path, "r") as f:
        images = f["input_image"]
        step = images.chunks[0] if images.chunks else 64
        order = df.sort_values("h5_row")
        blocks = itertools.groupby(zip(order.h5_row, order.path), key=lambda rp: rp[0] // step)
        for b, members in tqdm(blocks, desc=f"decode {path.name}", unit="chunk"):
            block = images[b * step:(b + 1) * step]
            for row, rel in members:
                Image.fromarray(block[row - b * step]).save(out_dir / rel, quality=95)


def build(raw_dir: Path | None = None, min_images: int = 50, min_groups: int = 3, seed: int = 0) -> pd.DataFrame:
    files = download(raw_dir)
    meta = {s: _read_metadata(p, s) for s, p in files.items()}
    df = pd.concat(meta.values(), ignore_index=True)
    # The train file lists a few products twice (28 rows, pixel-identical); keep the first copy.
    df = df.drop_duplicates("image_id")
    df = manifest.drop_rare(df, ("brand", "era"), min_images, min_groups)

    out_dir = paths.dataset_dir(NAME)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    for s, p in files.items():
        _extract_images(p, df[df.source_split == s], out_dir)

    # SSENSE reuses one outfit photo for every product in the look (poses 4-6), so products that share
    # an identical or near-identical photo are merged into one group before splitting.
    df["product_id"] = df.group_id
    df["group_id"] = manifest.merge_duplicate_groups(df, out_dir)
    df["split"] = manifest.assign_splits(df, strata_cols=("brand", "era"), seed=seed)
    return df.drop(columns="h5_row").reset_index(drop=True)
