"""Read label text with EasyOCR (CRAFT detection + CRNN recognition) and match it to brand names.

An early version of the weeks 6-7 OCR module: it measures how far plain OCR plus fuzzy string
matching gets on brand-label close-ups, and how it combines with the image classifier
(trust OCR when it matches a brand confidently, otherwise fall back to the image model).
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from tqdm import tqdm

SHORT = 4  # normalized names this short (e.g. "HM", "ZARA", "ONLY") must equal a whole OCR token exactly
MAX_JOIN = 3  # also try runs of up to 3 adjacent tokens within a line ("GINA TRICOT", "LAGER 157")


def normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def partial_ratio(needle: str, haystack: str) -> float:
    """Best SequenceMatcher ratio between needle and any equally long window of haystack."""
    if not needle or not haystack:
        return 0.0
    if len(haystack) <= len(needle):
        return SequenceMatcher(None, needle, haystack).ratio()
    return max(SequenceMatcher(None, needle, haystack[i:i + len(needle)]).ratio()
               for i in range(len(haystack) - len(needle) + 1))


def candidates(lines: list[str]) -> set[str]:
    """Normalized OCR tokens and runs of adjacent tokens, never crossing a line boundary."""
    out = set()
    for line in lines:
        tokens = [normalize(t) for t in line.split()]
        tokens = [t for t in tokens if t]
        for i in range(len(tokens)):
            for j in range(i + 1, min(i + MAX_JOIN, len(tokens)) + 1):
                out.add("".join(tokens[i:j]))
    return out


def brand_scores(lines: list[str], brands: list[str]) -> dict[str, float]:
    cands = candidates(lines)
    scores = {}
    for b in brands:
        nb = normalize(b)
        if len(nb) <= SHORT:
            scores[b] = float(nb in cands)
        else:
            scores[b] = max((partial_ratio(nb, c) for c in cands), default=0.0)
    return scores


def match(lines: list[str], brands: list[str], threshold: float) -> tuple[str | None, float]:
    """Best-matching brand if its score clears threshold and no other brand ties it."""
    scores = brand_scores(lines, brands)
    ranked = sorted(scores.values(), reverse=True)
    best = max(scores, key=scores.get)
    tied = len(ranked) > 1 and ranked[1] == ranked[0]
    return (best, scores[best]) if scores[best] >= threshold and not tied else (None, scores[best])


def read_labels(image_paths: list[Path], cache: Path, gpu: bool = True) -> dict[str, list[str]]:
    """OCR every image once (results cached as JSON keyed by file name)."""
    done = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [p for p in image_paths if p.name not in done]
    if todo:
        import os

        import certifi
        import easyocr

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())  # model download via urllib, as in backbones.py
        reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
        for i, p in enumerate(tqdm(todo, desc="OCR")):
            # Label photos are often upside down or sideways: let EasyOCR try each box rotated too.
            done[p.name] = [text for _, text, _ in reader.readtext(str(p), rotation_info=[90, 180, 270])]
            if i % 50 == 0:
                cache.write_text(json.dumps(done))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(done))
    return {p.name: done[p.name] for p in image_paths}
