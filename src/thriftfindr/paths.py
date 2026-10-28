"""Filesystem layout. Override the data/output roots with THRIFTFINDR_DATA / THRIFTFINDR_OUTPUTS."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("THRIFTFINDR_DATA", REPO_ROOT / "data"))
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
OUTPUTS_DIR = Path(os.environ.get("THRIFTFINDR_OUTPUTS", REPO_ROOT / "outputs"))
FEATURES_DIR = OUTPUTS_DIR / "features"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
RESULTS_DIR = OUTPUTS_DIR / "results"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORT_DIR = REPO_ROOT / "reports" / "midterm"


def dataset_dir(name: str) -> Path:
    """Directory holding a processed dataset: images/ plus manifest.csv."""
    return PROCESSED_DIR / name


def manifest_path(name: str) -> Path:
    return dataset_dir(name) / "manifest.csv"
