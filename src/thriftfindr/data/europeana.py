"""Europeana Fashion catwalk photos (data provider "Catwalk Pictures"): archive designers x decade.

Records come from the Europeana Search API and are labelled from titles such as
"Undercover, Autumn-Winter 2003, Womenswear": era is the decade of the collection year, subline the
sub-label plus show line ("Homme Plus Menswear", "Womenswear"), and group_id the runway show
"<brand>|<season>|<year>|<subline>". The photos are In Copyright: research use only.
"""

from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from thriftfindr import paths
from thriftfindr.data import manifest

NAME = "europeana_runway"
API_URL = "https://api.europeana.eu/record/v2/search.json"
PROVIDER = "Catwalk Pictures"
# Paul Van Riel's 1970s-80s slide scans would let a decade classifier learn the photo medium.
EXCLUDED_PROVIDERS = ("Paul Van Riel",)
PROFILE = "minimal"  # already has id, title, year, rights and image URLs; ~1/13 the size of "standard"
RECORD_FIELDS = ("id", "title", "year", "dataProvider", "dcCreator", "rights", "edmIsShownBy", "edmPreview")
USER_AGENT = "thriftfindr/0.1 (university course project; research use)"
MAX_WORKERS = 4
ERAS = ("1990s", "2000s", "2010s")
COLUMNS = ["image_id", "path", "dataset", "brand", "era", "group_id", "split",
           "subline", "year", "season", "source_url", "license"]

DESIGNER_QUERIES = {
    "Comme des Garcons": 'who:"Comme des Garçons" OR who:"Comme des Garcons" OR who:"Rei Kawakubo"',
    "Maison Margiela": 'who:"Margiela"',
    "Undercover": 'who:"Undercover" OR who:"Jun Takahashi"',
    "Yohji Yamamoto": 'who:"Yohji Yamamoto"',
    "Issey Miyake": 'who:"Issey Miyake" OR who:"Miyake Design Studio"',
    "Junya Watanabe": 'who:"Junya Watanabe"',
    "Helmut Lang": 'who:"Helmut Lang"',
    "Raf Simons": 'who:"Raf Simons"',
    "Jean Paul Gaultier": 'who:"Jean Paul Gaultier" OR who:"Jean-Paul Gaultier"',
    "Vivienne Westwood": 'who:"Vivienne Westwood"',
}
# How each house is written in titles (accents stripped); whatever else the house part says is the sub-line.
HOUSE_NAMES = {
    "Comme des Garcons": r"Comme des Garcons",
    "Maison Margiela": r"(?:Maison )?(?:Martin )?Margiela",
    "Undercover": r"Undercover",
    "Yohji Yamamoto": r"Yohji Yamamoto",
    "Issey Miyake": r"Issey Miyake",
    "Junya Watanabe": r"Junya Watanabe",
    "Helmut Lang": r"Helmut Lang",
    "Raf Simons": r"Raf Simons",
    "Jean Paul Gaultier": r"Jean[- ]Paul Gaultier",
    "Vivienne Westwood": r"Vivienne Westwood",
}
SUBLINE_ALIASES = {"Comme": "Comme des Garcons Comme des Garcons",
                   "Comme des Garcons": "Comme des Garcons Comme des Garcons"}
# Not a new collection of the house: a retrospective of older pieces, and a collaboration for another label.
EXCLUDED_SUBLINES = ("(Retro 30)", "La Perla")
# Presentation format or venue, not a sub-line: "Raf Simons (Static)" is the same collection as its show.
_NOT_SUBLINE = re.compile(r"\((?:Show|Static)\)|\bParis\b|\bby\b", re.I)
_SEASON = re.compile(r"\b(Spring|Autumn|Fall)[- ]?(?:Summer|Winter)\s+((?:19|20)\d{2})\b", re.I)
_LINE = re.compile(r"\b(Haute Couture|Couture|Womenswear|Menswear)\b", re.I)

_local = threading.local()


def _session() -> requests.Session:
    """One session per thread, retrying throttled or failed GETs with exponential backoff."""
    if not hasattr(_local, "session"):
        retry = Retry(total=6, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset({"GET"}), respect_retry_after_header=True)
        s = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://", HTTPAdapter(max_retries=retry))
        _local.session = s
    return _local.session


def api_key() -> str:
    return os.environ.get("EUROPEANA_API_KEY", "api2demo")


def search(query: str, rows: int = 100) -> list[dict]:
    """All Catwalk Pictures image records matching query, via cursor pagination."""
    params = {"wskey": api_key(), "query": query, "qf": ["TYPE:IMAGE", f'DATA_PROVIDER:"{PROVIDER}"'],
              "rows": rows, "profile": PROFILE, "cursor": "*"}
    items = []
    while True:
        r = _session().get(API_URL, params=params, timeout=(10, 60))
        r.raise_for_status()
        page = r.json()
        if not page.get("success", False):
            raise RuntimeError(f"Europeana search failed: {page.get('error', page)}")
        # Keeping only the needed fields also drops "link"/"guid", which embed the API key.
        items += [{k: it[k] for k in RECORD_FIELDS if k in it} for it in page.get("items", [])]
        if not page.get("items") or not page.get("nextCursor"):
            return items
        params["cursor"] = page["nextCursor"]


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def fetch_records(cache_dir: Path | None = None, refresh: bool = False) -> dict[str, list[dict]]:
    """Search results per designer, cached as JSON lines under data/raw/europeana_runway/search/."""
    cache_dir = Path(cache_dir) if cache_dir is not None else paths.RAW_DIR / NAME / "search"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for brand, query in DESIGNER_QUERIES.items():
        path = cache_dir / f"{_slug(brand)}.jsonl"
        if refresh or not path.is_file():
            items = search(query)
            tmp = path.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
            tmp.replace(path)
        out[brand] = [json.loads(line) for line in path.read_text().splitlines() if line]
    return out


def _ascii(text: str) -> str:
    """Strip accents and odd spaces: 'Comme des Garçons\\xa0Womenswear' -> 'Comme des Garcons Womenswear'."""
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip()


def parse_title(title: str) -> dict:
    """Split a title into house part, season ("SS"/"AW"), collection year and line.

    Handles both "House Sub, Spring-Summer 2016, Womenswear" and "House Womenswear Spring-Summer 2013".
    """
    text = _ascii(title)
    season, line = _SEASON.search(text), _LINE.search(text)
    cut = min((m.start() for m in (season, line) if m), default=len(text))
    line_name = line.group(1).title() if line else ""
    return {"house": text[:cut].strip(" ,"),
            "season": ("SS" if season.group(1).lower() == "spring" else "AW") if season else "",
            "year": int(season.group(2)) if season else None,
            "line": "Couture" if line_name.endswith("Couture") else line_name}


def sub_label(house: str, brand: str) -> str | None:
    """What a title's house part adds to the brand ("Homme Plus", "MM6", "Tao"); None if it is another house."""
    m = re.search(HOUSE_NAMES[brand], house, re.I)
    if m is None:
        return None
    rest = _NOT_SUBLINE.sub(" ", f"{house[:m.start()]} {house[m.end():]}")
    rest = re.sub(r"\s+", " ", rest).strip()
    return SUBLINE_ALIASES.get(rest, rest)


def _first_year(values: list[str] | None) -> int | None:
    for v in values or []:
        if re.fullmatch(r"(?:19|20)\d{2}", str(v).strip()):
            return int(v)
    return None


def records_to_frame(records: dict[str, list[dict]]) -> pd.DataFrame:
    """One labelled row per dated record of the queried house; also keeps the image URLs to download."""
    rows = []
    for brand, items in records.items():
        for it in items:
            if any(p in EXCLUDED_PROVIDERS for p in it.get("dataProvider", [])):
                continue
            t = parse_title((it.get("title") or [""])[0])
            sub = sub_label(t["house"], brand)
            year = t["year"] or _first_year(it.get("year"))
            if sub is None or sub in EXCLUDED_SUBLINES or year is None:
                continue
            subline = " ".join(filter(None, [sub, t["line"]]))
            image_id = re.sub(r"[^A-Za-z0-9._-]+", "_", it["id"]).strip("_")
            rows.append({
                "image_id": image_id,
                "path": f"images/{image_id}.jpg",
                "dataset": NAME,
                "brand": brand,
                "era": f"{year // 10 * 10}s",
                "group_id": f"{brand}|{t['season']}|{year}|{subline}",
                "subline": subline,
                "year": year,
                "season": t["season"],
                "source_url": f"https://www.europeana.eu/item{it['id']}",
                "license": (it.get("rights") or [""])[0],
                "image_url": (it.get("edmIsShownBy") or [""])[0],
                "preview_url": (it.get("edmPreview") or [""])[0],
            })
    df = pd.DataFrame(rows).drop_duplicates("image_id")
    return df[df.era.isin(ERAS)].reset_index(drop=True)


def sample_shows(df: pd.DataFrame, cap: int = 400, per_show: int = 25, seed: int = 0) -> pd.DataFrame:
    """Up to cap images per (brand, era), dealt round-robin over its shows with at most per_show per show."""
    rng = np.random.default_rng(seed)
    df = df.sort_values("image_id").sample(frac=1, random_state=rng)
    rank = df.groupby("group_id").cumcount()  # position of the image in its show's random order
    shows = np.sort(df.group_id.unique())
    show_order = df.group_id.map(dict(zip(rng.permutation(shows), range(len(shows)))))
    df = df.assign(_rank=rank, _show=show_order)[rank < per_show].sort_values(["_rank", "_show"])
    return df.groupby(["brand", "era"]).head(cap).drop(columns=["_rank", "_show"])


def _save_image(candidates: list[tuple[str, str]], dest: Path, max_side: int, quality: int) -> str:
    """Save the first (source, url) candidate that decodes as an image; returns its source ("" if none)."""
    if dest.is_file():
        return "cached"
    for source, url in candidates:
        try:
            r = _session().get(url, timeout=(10, 60))
            r.raise_for_status()
            im = ImageOps.exif_transpose(Image.open(BytesIO(r.content))).convert("RGB")
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            tmp = dest.with_name(dest.name + ".part")
            im.save(tmp, "JPEG", quality=quality)
            tmp.replace(dest)
            return source
        except (requests.RequestException, OSError, Image.DecompressionBombError):
            continue
    return ""


def download_images(df: pd.DataFrame, root: Path, workers: int = MAX_WORKERS, max_side: int = 1024,
                    quality: int = 92) -> pd.Series:
    """Download each row's full image (thumbnail as fallback) to root/<path>; returns the source used per row."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    jobs = [([s for s in (("full", row.image_url), ("preview", row.preview_url)) if s[1]], root / row.path)
            for row in df.itertuples()]
    with ThreadPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as pool:  # polite to the image host
        sources = list(tqdm(pool.map(lambda job: _save_image(*job, max_side, quality), jobs),
                            total=len(jobs), desc="images"))
    return pd.Series(sources, index=df.index, name="source")


def build(cap: int = 400, per_show: int = 25, seed: int = 0, workers: int = MAX_WORKERS,
          max_side: int = 1024, quality: int = 92, refresh: bool = False) -> pd.DataFrame:
    """Fetch, label, sample, download and split the europeana_runway manifest."""
    records = fetch_records(refresh=refresh)
    df = records_to_frame(records)
    print(f"{sum(map(len, records.values()))} records -> {len(df)} labelled images in {df.group_id.nunique()} shows")
    df = sample_shows(df, cap=cap, per_show=per_show, seed=seed)
    df = manifest.drop_rare(df, ("brand", "era"), min_images=50, min_groups=3)
    source = download_images(df, paths.dataset_dir(NAME), workers=workers, max_side=max_side, quality=quality)
    print("image sources:", source.value_counts().to_dict())
    df = manifest.drop_rare(df[source != ""], ("brand", "era"), min_images=50, min_groups=3)
    df = df.sort_values(["brand", "era", "group_id", "image_id"]).reset_index(drop=True)
    # Merge shows that share an identical or near-identical photo (e.g. a look filed under two shows).
    df["group_id"] = manifest.merge_duplicate_groups(df, paths.dataset_dir(NAME))
    df["split"] = manifest.assign_splits(df, strata_cols=("brand", "era"))
    return df[COLUMNS].reset_index(drop=True)
