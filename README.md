# Thrift Findr
Predict the **brand**, **era**, and **market value** of second-hand archive garments (Comme des
Garçons, Maison Margiela, Undercover, ...) from a photo, ideally of the garment's label.
Course project by Michael Yang (msyang9) and Alex Ryan (apryan3).

The midterm report is in [reports/midterm/](reports/midterm/); build `midterm.pdf` with step 8 below. The
PDF and its sample-image figure are not committed because they contain dataset images.

## Status (midterm)

Automated collection from resale sites is not possible (eBay's Buy APIs are partner-only and its
2025 license restricts AI use; Grailed and Depop block automated access), so the full pipeline is
developed and evaluated on three public **proxy datasets** while we collect our own label photos:

| Dataset | What it covers | Labels | License |
|---|---|---|---|
| Fashion-Gen (SSENSE catalog) | 9 archive houses | brand, season year 2014–18, retail price | non-commercial research |
| Clothing Dataset for Second-Hand Fashion | real brand-label close-ups from donated garments | brand (high-street) | CC BY 4.0 |
| Europeana Fashion catwalk photos | archive designers across decades | brand, decade 1990s–2010s | In Copyright (research only) |

Images are never committed or redistributed; every builder downloads from the original source.

Midterm results (test splits, CLIP ViT-B/16; details and confidence intervals in the report):

| Task | Frozen + linear probe | Fine-tuned | Reference point |
|---|---|---|---|
| Fashion-Gen brand (9 houses) | 56.9% | 65.3% | majority class 28.9% |
| Fashion-Gen era (5 season years) | 69.7% | 81.3% | majority class 31.7% |
| Label-photo brand (11 brands) | 78.0% | 84.6%, 87.0% with OCR | DINOv2 35.2%, ResNet-50 27.1% |
| Runway brand / decade | 63.7% / 81.8% | – | decade majority 45.3% |
| Retail price, median abs. % error | 31.2% (kNN retrieval) | – | brand-median lookup 41.7% |

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt   # exact versions used: requirements-lock.txt
.venv/bin/pip install -e .
```

Runs on Apple-silicon GPUs (PyTorch MPS), CUDA, or CPU. Data and outputs default to `data/` and
`outputs/` (override with `THRIFTFINDR_DATA` / `THRIFTFINDR_OUTPUTS`).

## Pipeline

```bash
# 1. Build datasets -> data/processed/<name>/{images/, manifest.csv, stats.json}
.venv/bin/python scripts/build_fashiongen.py          # downloads ~16 GB of HDF5 once
.venv/bin/python scripts/build_secondhand_labels.py   # reads only the needed parquet columns
.venv/bin/python scripts/build_europeana_runway.py    # Europeana Search API (demo key or EUROPEANA_API_KEY)

# 2. Frozen features (CLIP ViT-B/16, DINOv2 ViT-S/14, ResNet-50) -> outputs/features/
.venv/bin/python scripts/extract_features.py --dataset fashiongen

# 3. Baselines: linear probe + kNN for every task, plus price estimation -> outputs/results/<dataset>/baselines.json
.venv/bin/python scripts/run_baselines.py --dataset fashiongen

# 4. Fine-tune the best backbone (multi-task heads, phone-photo augmentation, 8 epochs by default)
.venv/bin/python scripts/finetune.py --dataset fashiongen --backbone clip_vitb16 --aug strong
.venv/bin/python scripts/finetune.py --dataset secondhand_labels --backbone clip_vitb16 --aug strong
.venv/bin/python scripts/finetune.py --dataset secondhand_labels --backbone clip_vitb16 --aug light   # ablation

# 5. Robustness to blur / glare / rotation / perspective / JPEG / low light
.venv/bin/python scripts/eval_robustness.py --dataset secondhand_labels \
    --checkpoints outputs/checkpoints/secondhand_labels/clip_vitb16_strong.pt outputs/checkpoints/secondhand_labels/clip_vitb16_light.pt \
    --frozen-backbones clip_vitb16
.venv/bin/python scripts/eval_robustness.py --dataset fashiongen \
    --checkpoints outputs/checkpoints/fashiongen/clip_vitb16_strong.pt --frozen-backbones clip_vitb16

# 6. Checks: low-level photo-style shortcuts, and train/test leakage (exact, near-duplicate, CLIP cos >= 0.99)
.venv/bin/python scripts/analyze_shortcuts.py --dataset fashiongen
.venv/bin/python scripts/audit_leakage.py --dataset fashiongen

# 7. OCR on label photos (EasyOCR + brand-name matching), alone and fused with the image models
.venv/bin/python scripts/run_ocr.py --dataset secondhand_labels

# 8. Report tables and figures (from the result files only), then the PDF
.venv/bin/python scripts/make_report_assets.py
cd reports/midterm && latexmk -pdf midterm.tex
```

## Layout

```
src/thriftfindr/
  data/manifest.py     manifest schema, grouped + stratified splits, validation
  data/registry.py     tasks per dataset
  data/fashiongen.py   data/secondhand.py   data/europeana.py   dataset builders
  data/transforms.py   phone-photo augmentations and test-time corruptions
  data/dataset.py      PyTorch dataset over a manifest
  models/backbones.py  CLIP / DINOv2 / ResNet-50 encoders
  features.py          feature extraction + cache
  baselines.py         linear probe and kNN
  price.py             retrieval-based price estimation
  finetune.py          multi-task fine-tuning (linear probe, then fine-tune)
  robustness.py        corruption benchmark
  evaluation.py        metrics with product-level bootstrap intervals
  ocr.py               EasyOCR label reading + brand-name matching
  viz.py               report figures
scripts/               command-line entry points (above)
configs/brand_eras.yaml  draft label-based era definitions for the real data (to verify with garments)
reports/midterm/       LaTeX report, generated tables/figures, verified bibliography
```

Every split is grouped (all photos of one product, garment or runway show stay in one split; groups
that share an identical or near-identical photo are merged) and stratified by class, 5:1:1. Classes
with fewer than 50 images or 3 groups are dropped, and every kept class appears in all three splits.
