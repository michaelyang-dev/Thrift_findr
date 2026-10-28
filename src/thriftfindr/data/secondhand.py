"""Brand-label photos from the Clothing Dataset for Second-Hand Fashion (Zenodo 13788681, CC BY 4.0).

Reads the Hugging Face parquet repack (one row per donated garment, JPEGs inline, ~26.6 GB) without a bulk
download:

1. scan: per row group, fetch only the small metadata columns and the data page of the image_brand chunk.
   That chunk is dictionary-encoded (the JPEGs sit in its dictionary page), so its data page is ~100 bytes
   of definition levels and indices; the definition levels say, per garment, whether a label photo exists.
   Cached in data/raw/.
2. select: a seeded random subset of row groups (100 garments, ~34 MB of label photos each) within a byte
   budget; inside it, rank brands by usable label photos, keep those with >= min_images, cap each brand
   with a seeded random sample.
3. download: read only the image_brand chunk of row groups holding selected garments and save each photo
   resized to <= max_side px as images/<garment_id>.jpg.
"""

from __future__ import annotations

import hashlib
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image, ImageOps

from thriftfindr import paths
from thriftfindr.data import manifest

NAME = "secondhand_labels"
REPO_ID = "chibifire/zenodo-second-hand-fashion-v3"
REVISION = "1975649e0a779890a74018a995a8fb0732249caf"  # pinned: the selection depends on the row-group layout
LICENSE = "CC BY 4.0"
SOURCE_URL = "https://doi.org/10.5281/zenodo.13788681"
MAX_WORKERS = 4  # parallel HTTP range requests
IMAGE_COLUMN = "image_brand"
META_COLUMNS = ["garment_id", "brand", "category", "price", "station"]
PLACEHOLDER_BRANDS = {"notinthelist", "missing", "notapplicable"}  # brand_key of the non-brand values
CATEGORY_FIXES = {"ladys": "ladies", "women": "ladies"}
PRICE_FIXES = {"150-200": "150-250"}  # one garment uses a band that lies inside the standard 150-250


def brand_key(brand: pd.Series) -> pd.Series:
    """Case-, accent- and punctuation-insensitive key, so 'KappAhl' and 'Kappahl' compare equal."""
    return brand.str.normalize("NFKD").str.lower().str.replace(r"[^a-z0-9]", "", regex=True)


def normalise_brands(brand: pd.Series) -> pd.Series:
    """Map each brand's spelling variants onto its most frequent spelling."""
    key = brand_key(brand)
    counts = pd.DataFrame({"key": key, "brand": brand}).value_counts().reset_index()
    spelling = (counts.sort_values(["key", "count", "brand"], ascending=[True, False, True])
                .drop_duplicates("key").set_index("key").brand)
    return key.map(spelling)


def normalise_categories(category: pd.Series) -> pd.Series:
    key = category.str.strip().str.lower()
    return key.replace(CATEGORY_FIXES).str.capitalize()


# --- reading label-photo presence from the parquet pages (format spec: github.com/apache/parquet-format)

def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7


def _thrift_value(buf: bytes, pos: int, ctype: int):
    """Decode one Thrift compact-protocol value; returns (value, next position)."""
    if ctype in (1, 2):  # bool, stored in the field header itself
        return ctype == 1, pos
    if ctype == 3:
        return buf[pos], pos + 1
    if ctype in (4, 5, 6):  # zigzag varint i16 / i32 / i64
        n, pos = _varint(buf, pos)
        return (n >> 1) ^ -(n & 1), pos
    if ctype == 7:
        return buf[pos:pos + 8], pos + 8
    if ctype == 8:
        n, pos = _varint(buf, pos)
        return buf[pos:pos + n], pos + n
    if ctype in (9, 10):  # list / set
        size, etype = buf[pos] >> 4, buf[pos] & 0x0F
        pos += 1
        if size == 15:
            size, pos = _varint(buf, pos)
        items = []
        for _ in range(size):
            if etype in (1, 2):  # bools inside containers take one byte each
                item, pos = buf[pos] == 1, pos + 1
            else:
                item, pos = _thrift_value(buf, pos, etype)
            items.append(item)
        return items, pos
    if ctype == 12:
        return _thrift_struct(buf, pos)
    raise ValueError(f"unsupported Thrift compact type {ctype}")


def _thrift_struct(buf: bytes, pos: int = 0) -> tuple[dict, int]:
    """Decode a Thrift compact-protocol struct into {field id: value}."""
    fields, fid = {}, 0
    while buf[pos] != 0:  # 0 is the stop field
        delta, ctype = buf[pos] >> 4, buf[pos] & 0x0F
        pos += 1
        if delta:
            fid += delta
        else:
            n, pos = _varint(buf, pos)
            fid = (n >> 1) ^ -(n & 1)
        fields[fid], pos = _thrift_value(buf, pos, ctype)
    return fields, pos + 1


def _rle_bits(buf: bytes, count: int) -> np.ndarray:
    """Decode `count` 1-bit values stored with Parquet's RLE / bit-packed hybrid encoding."""
    runs, pos, total = [], 0, 0
    while total < count:
        header, pos = _varint(buf, pos)
        if header & 1:  # bit-packed: (header >> 1) groups of 8 values, one byte per group at bit width 1
            run = np.unpackbits(np.frombuffer(buf, np.uint8, header >> 1, pos), bitorder="little")
            pos += header >> 1
        else:  # RLE: one value (a single byte at bit width 1) repeated header >> 1 times
            run = np.full(header >> 1, buf[pos], np.uint8)
            pos += 1
        runs.append(run)
        total += len(run)
    return np.concatenate(runs)[:count].astype(bool)


def _photo_mask(pages: bytes, compression: str) -> np.ndarray:
    """Per-row presence of a flat optional column, decoded from the bytes of its data pages."""
    masks, pos = [], 0
    while pos < len(pages):
        header, pos = _thrift_struct(pages, pos)
        body, pos = pages[pos:pos + header[3]], pos + header[3]
        if header[1] == 0:  # DATA_PAGE: whole body compressed, [u32 length][definition levels][values]
            if header[5][3] != 3:
                raise ValueError("definition levels are not RLE-encoded")
            raw = body if compression == "UNCOMPRESSED" else \
                pa.Codec(compression).decompress(body, decompressed_size=header[2]).to_pybytes()
            n, levels = header[5][1], raw[4:4 + int.from_bytes(raw[:4], "little")]
        elif header[1] == 3:  # DATA_PAGE_V2: uncompressed [repetition levels][definition levels] up front
            v2 = header[8]
            n, levels = v2[1], body[v2[6]:v2[6] + v2[5]]
        else:
            raise ValueError(f"unexpected parquet page type {header[1]}")
        masks.append(_rle_bits(levels, n))
    return np.concatenate(masks)


# --- scan

def _chunk_span(col) -> tuple[int, int]:
    start = col.dictionary_page_offset if col.has_dictionary_page else col.data_page_offset
    return start, start + col.total_compressed_size


def _columns(row_group) -> dict:
    return {row_group.column(j).path_in_schema: row_group.column(j) for j in range(row_group.num_columns)}


def _merge_spans(spans: list[tuple[int, int]], max_gap: int = 1 << 16) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _scan_file(fs: HfFileSystem, path: str) -> pd.DataFrame:
    """Metadata columns plus label-photo presence for every garment in one parquet file."""
    with fs.open(path, "rb", cache_type="none") as f:  # "none": fetch exactly the bytes asked for
        md = pq.ParquetFile(f).metadata
        groups = [i for i in range(md.num_row_groups) if md.row_group(i).num_rows > 0]
        spans, pages = [], []
        for i in groups:
            cols = _columns(md.row_group(i))
            spans += [_chunk_span(cols[c]) for c in META_COLUMNS]
            img = cols[IMAGE_COLUMN]
            pages.append((img.data_page_offset, _chunk_span(img)[1], img.compression))
        parts = {}
        for start, end in _merge_spans(spans + [(s, e) for s, e, _ in pages]):
            f.seek(start)
            parts[(start, end)] = f.read(end - start)
    if not groups:
        return pd.DataFrame()

    def cut(start: int, end: int) -> bytes:
        (s0, _), data = next(item for item in parts.items() if item[0][0] <= start and end <= item[0][1])
        return data[start - s0:end - s0]

    # pyarrow decodes the metadata columns from the fetched ranges; strict mode forbids any other read
    with fs.open(path, "rb", cache_type="parts", cache_options={"data": dict(parts), "strict": True}) as f:
        df = pq.ParquetFile(f, metadata=md).read_row_groups(groups, columns=META_COLUMNS).to_pandas()
    sizes = [md.row_group(i).num_rows for i in groups]
    df["has_photo"] = np.concatenate([_photo_mask(cut(s, e), c) for s, e, c in pages])
    df["file"] = path.rsplit("/", 1)[-1]
    df["row_group"] = np.repeat(groups, sizes)
    df["row"] = np.concatenate([np.arange(n) for n in sizes])
    df["chunk_bytes"] = np.repeat([_columns(md.row_group(i))[IMAGE_COLUMN].total_compressed_size
                                   for i in groups], sizes)
    return df


def load_garments(fs: HfFileSystem, revision: str = REVISION, workers: int = 4) -> pd.DataFrame:
    """One row per garment: metadata, has_photo, and where its label photo lives (file, row_group, row)."""
    cache = paths.RAW_DIR / NAME / f"garments-{revision[:12]}.parquet"
    if cache.is_file():
        print(f"using cached scan {cache}")
        return pd.read_parquet(cache)
    root = f"datasets/{REPO_ID}@{revision}/data"
    names = sorted(p.rsplit("/", 1)[-1] for p in fs.glob(f"{root}/*.parquet"))
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        frames = list(pool.map(lambda name: _scan_file(fs, f"{root}/{name}"), names))
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    print(f"scanned {len(df)} garments in {len(names)} files ({int(df.has_photo.sum())} with a label photo)"
          f" in {time.time() - t0:.0f}s")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


# --- select and download

def select(garments: pd.DataFrame, max_bytes: float, min_images: int = 100, max_per_brand: int = 300,
           max_classes: int = 15, seed: int = 0) -> pd.DataFrame:
    """Seeded row-group subset within max_bytes, then the top brands by usable label photos inside it."""
    groups = (garments[["file", "row_group", "chunk_bytes"]].drop_duplicates()
              .sort_values(["file", "row_group"]).sample(frac=1, random_state=seed))
    groups = groups[groups.chunk_bytes.cumsum() <= max_bytes]
    usable = garments[garments.has_photo & ~brand_key(garments.brand).isin(PLACEHOLDER_BRANDS)]
    pool = usable.merge(groups[["file", "row_group"]], on=["file", "row_group"])
    counts = pool.brand.value_counts().sort_index().sort_values(ascending=False, kind="stable")
    classes = counts[counts >= min_images].index[:max_classes]
    pool = pool[pool.brand.isin(classes)].sample(frac=1, random_state=seed)
    return pool[pool.groupby("brand").cumcount() < max_per_brand].sort_values("garment_id")


def _read_labels(fs: HfFileSystem, path: str, md, row_group: int, retries: int = 4) -> pa.ChunkedArray:
    for attempt in range(retries):
        try:
            with fs.open(path, "rb", cache_type="none") as f:
                return pq.ParquetFile(f, metadata=md).read_row_group(row_group, columns=[IMAGE_COLUMN]).column(0)
        except Exception as err:  # network errors surface from inside pyarrow with varying types
            if attempt == retries - 1:
                raise
            wait = 5 * 2 ** attempt
            print(f"  retry {path.rsplit('/', 1)[-1]} row group {row_group} in {wait}s ({err})")
            time.sleep(wait)


def _save_label(data: bytes, out: Path, max_side: int) -> None:
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    tmp = out.with_suffix(".part")  # write-then-rename, so an interrupted run never leaves half a JPEG
    im.save(tmp, format="JPEG", quality=92)
    tmp.replace(out)


def _download_file(fs: HfFileSystem, path: str, rows: pd.DataFrame, image_dir: Path, max_side: int) -> int:
    """Save the selected label photos held in one parquet file; returns the bytes fetched."""
    todo = rows[[not (image_dir / f"{g}.jpg").is_file() for g in rows.garment_id]]
    if todo.empty:
        return 0
    with fs.open(path, "rb", cache_type="none") as f:
        md = pq.ParquetFile(f).metadata
    fetched = 0
    for row_group, sel in todo.groupby("row_group"):
        labels = _read_labels(fs, path, md, int(row_group))
        fetched += int(sel.chunk_bytes.iloc[0])
        for garment_id, row in zip(sel.garment_id, sel.row):
            data = labels[int(row)].as_py()
            if data is None:
                print(f"  garment {garment_id}: no label photo after all, skipped")
                continue
            try:
                _save_label(data, image_dir / f"{garment_id}.jpg", max_side)
            except OSError as err:  # undecodable JPEG
                print(f"  garment {garment_id}: {err}, skipped")
    return fetched


def download(fs: HfFileSystem, picked: pd.DataFrame, image_dir: Path, revision: str = REVISION,
             max_side: int = 1024, workers: int = 4) -> int:
    """Fetch the image_brand chunks holding the picked garments; returns total bytes fetched."""
    image_dir.mkdir(parents=True, exist_ok=True)
    root = f"datasets/{REPO_ID}@{revision}/data"
    by_file = list(picked.groupby("file"))
    total, t0 = 0, time.time()
    with ThreadPoolExecutor(workers) as pool:
        jobs = [pool.submit(_download_file, fs, f"{root}/{name}", rows, image_dir, max_side)
                for name, rows in by_file]
        for done, ((name, rows), job) in enumerate(zip(by_file, jobs), 1):
            total += job.result()
            print(f"[{done}/{len(jobs)}] {name}: {len(rows)} photos; {total / 1e9:.2f} GB fetched, "
                  f"{time.time() - t0:.0f}s")
    return total


def build(max_gb: float = 6.0, min_images: int = 100, max_per_brand: int = 300, max_classes: int = 15,
          max_side: int = 1024, seed: int = 0, workers: int = 4, revision: str = REVISION) -> pd.DataFrame:
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be 1-{MAX_WORKERS} (polite concurrency)")
    fs = HfFileSystem(token=False)  # public dataset: never send credentials
    garments = load_garments(fs, revision, workers)
    garments["brand"] = normalise_brands(garments.brand)
    picked = select(garments, max_gb * 1e9, min_images, max_per_brand, max_classes, seed)
    print(f"selected {len(picked)} label photos of {picked.brand.nunique()} brands from "
          f"{picked.groupby(['file', 'row_group']).ngroups} row groups")

    image_dir = paths.dataset_dir(NAME) / "images"
    fetched = download(fs, picked, image_dir, revision, max_side, workers)
    print(f"fetched {fetched / 1e9:.2f} GB of label-photo chunks")

    ids = picked.garment_id.astype(str)
    df = pd.DataFrame({
        "image_id": ids,
        "path": "images/" + ids + ".jpg",
        "dataset": NAME,
        "brand": picked.brand,
        "era": "",
        "group_id": ids,
        "price_band": picked.price.replace(PRICE_FIXES),
        "category": normalise_categories(picked.category),
        "station": picked.station,
        "source_url": SOURCE_URL,
        "license": LICENSE,
    }).reset_index(drop=True)
    df = df[[(paths.dataset_dir(NAME) / p).is_file() for p in df.path]]
    # a label photo stored under two garment ids would straddle splits; keep the first copy
    digest = [hashlib.md5((paths.dataset_dir(NAME) / p).read_bytes()).hexdigest() for p in df.path]
    dupes = pd.Series(digest, index=df.index).duplicated()
    if dupes.any():
        print(f"dropped {int(dupes.sum())} byte-identical duplicate label photos")
    df = df[~dupes].reset_index(drop=True)
    df["split"] = manifest.assign_splits(df, strata_cols=("brand",))
    return df
