"""Which prediction tasks each processed dataset supports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    tasks: tuple[str, ...]  # manifest label columns to predict
    description: str
    hflip: bool = True  # mirrored text is unrealistic, so label datasets disable horizontal flips
    price_col: str | None = None


DATASETS = {
    "fashiongen": DatasetSpec(
        "fashiongen", ("brand", "era"),
        "Fashion-Gen (SSENSE catalog photos): 9 archive houses, era = season year 2014-2018, MSRP in USD",
        hflip=True, price_col="price"),
    "secondhand_labels": DatasetSpec(
        "secondhand_labels", ("brand",),
        "Clothing Dataset for Second-Hand Fashion (CC BY 4.0): brand-label close-ups of donated garments",
        hflip=False),
    "europeana_runway": DatasetSpec(
        "europeana_runway", ("brand", "era"),
        "Europeana Fashion catwalk photos: archive designers x decade (1990s-2010s)",
        hflip=True),
}


def get(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[name]
