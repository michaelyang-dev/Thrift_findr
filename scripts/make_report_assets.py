"""Generate the midterm report's LaTeX tables and figures from outputs/results.

python scripts/make_report_assets.py
Writes reports/midterm/tables/*.tex and reports/midterm/figures/*.{pdf,png}. Every number in the
report tables comes from these result files; nothing is typed by hand.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from thriftfindr import paths, viz
from thriftfindr.data import manifest
from thriftfindr.data.transforms import CORRUPTIONS

TABLES = paths.REPORT_DIR / "tables"
FIGURES = paths.REPORT_DIR / "figures"
BACKBONE_NAMES = {"clip_vitb16": "CLIP ViT-B/16", "dinov2_vits14": "DINOv2 ViT-S/14", "resnet50": "ResNet-50"}
SHORT_NAMES = {"clip_vitb16": "CLIP", "dinov2_vits14": "DINOv2", "resnet50": "ResNet-50"}  # single-column tables
CORRUPTION_NAMES = {c: ("JPEG" if c == "jpeg" else c.capitalize()) for c in CORRUPTIONS}
BIB_DROP = ("url", "doi", "eprint", "archiveprefix", "primaryclass", "editor", "address", "publisher",
            "series", "month", "location", "isbn", "issn", "note", "biburl", "bibsource", "timestamp")
METHOD_NAMES = {"linear_probe": "linear probe", "knn": "$k$-NN"}
DATASET_NAMES = {"fashiongen": "Fashion-Gen", "secondhand_labels": "Second-hand labels",
                 "europeana_runway": "Europeana runway"}


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def acc_ci(r: dict) -> str:
    lo, hi = r["accuracy_ci95"]
    return f"{pct(r['accuracy'])} \\ci{{{pct(lo)}}}{{{pct(hi)}}}"


def finetune_runs(dataset: str) -> dict[str, dict]:
    runs = {}
    for f in sorted((paths.RESULTS_DIR / dataset).glob("finetune_*.json")):
        runs[f.stem.removeprefix("finetune_")] = json.loads(f.read_text())
    return runs


def ft_label(run: dict, with_backbone: bool = True, short: bool = False) -> str:
    cfg = run["config"]
    extras = [f"{cfg['aug']} aug."]
    if cfg.get("probe_init"):
        extras.append("probe init")
    if cfg.get("seed", 0):
        extras.append(f"seed {cfg['seed']}")
    if cfg.get("tag"):
        extras.append(cfg["tag"])
    names = SHORT_NAMES if short else BACKBONE_NAMES
    name = f"{names[cfg['backbone']]} fine-tuned" if with_backbone else "fine-tuned"
    return f"{name} ({', '.join(extras)})"


def write(name: str, body: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{name}.tex").write_text(body)


def table_datasets() -> None:
    rows = []
    era_desc = {"fashiongen": "years 2014--18", "secondhand_labels": "---",
                "europeana_runway": "decades 1990s--2010s"}
    group_desc = {"fashiongen": "product groups", "secondhand_labels": "garments", "europeana_runway": "shows"}
    lic = {"fashiongen": "research only", "secondhand_labels": "CC BY 4.0", "europeana_runway": "In Copyright"}
    for ds in DATASET_NAMES:
        stats = load_json(paths.dataset_dir(ds) / "stats.json")
        if stats is None:
            continue
        sp = stats["splits"]
        n_brand = len(stats.get("brand_by_split", {}))
        n_era = len(stats.get("era_by_split", {}))
        groups = f"{stats['n_groups']:,} {group_desc[ds]}"
        if ds == "fashiongen":  # groups merge products that share an identical or near-identical photo
            groups = f"{stats['n_groups']:,} ({manifest.load(ds).product_id.nunique():,} products)"
        rows.append(f"{DATASET_NAMES[ds]} & {stats['n_images']:,} & {groups} & "
                    f"{n_brand} & {n_era or '---'} ({era_desc[ds]}) & "
                    f"{sp['train']:,}/{sp['val']:,}/{sp['test']:,} & {lic[ds]} \\\\")
    write("datasets", "\\begin{table*}[t]\\centering\\footnotesize\n"
          "\\caption{Proxy datasets used to validate the pipeline. Splits are grouped by product (merged when "
          "products share a photo), garment or runway show, and stratified by class.}\\label{tab:datasets}\n"
          "\\begin{tabular}{lrlcllr}\\toprule\n"
          "Dataset & Images & Split groups & Brands & Eras & Train/val/test & License \\\\\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")


def table_fashiongen() -> None:
    base = load_json(paths.RESULTS_DIR / "fashiongen" / "baselines.json")
    shortcuts = load_json(paths.RESULTS_DIR / "fashiongen" / "shortcuts.json")
    if base is None:
        return
    lines = []
    if shortcuts:
        ch = shortcuts["chance"]
        lines.append(f"Majority class & {pct(ch['brand']['majority_class_test_acc'])} & --- & "
                     f"{pct(ch['era']['majority_class_test_acc'])} & --- & --- \\\\")
        t = shortcuts["features"]["thumbnail_8x8"]
        lines.append(f"8$\\times$8 thumbnail, linear probe & {pct(t['brand']['accuracy'])} & {pct(t['brand']['macro_f1'])} & "
                     f"{pct(t['era']['accuracy'])} & {pct(t['era']['macro_f1'])} & --- \\\\\\midrule")
    for backbone, res in base["results"].items():
        for m in ("linear_probe", "knn"):
            b, e, j = res["brand"][m]["test"], res["era"][m]["test"], res["joint"][m]
            lines.append(f"{BACKBONE_NAMES[backbone]}, {METHOD_NAMES[m]} & {acc_ci(b)} & {pct(b['macro_f1'])} & "
                         f"{acc_ci(e)} & {pct(e['macro_f1'])} & {pct(j['both_correct'])} \\\\")
    runs = finetune_runs("fashiongen")
    if runs:
        lines[-1] += "\\midrule"
    for run in runs.values():
        b, e, j = run["test"]["brand"], run["test"]["era"], run["test"]["joint"]
        lines.append(f"{ft_label(run)} & {acc_ci(b)} & {pct(b['macro_f1'])} & {acc_ci(e)} & "
                     f"{pct(e['macro_f1'])} & {pct(j['both_correct'])} \\\\")
    write("fashiongen", "\\begin{table*}[t]\\centering\\small\n"
          "\\caption{Fashion-Gen test results (\\%; 9 brands, 5 era years; 95\\% product-bootstrap intervals in "
          "brackets). ``Both'' = brand and era both correct.}\\label{tab:fashiongen}\n"
          "\\begin{tabular}{lccccc}\\toprule\n"
          "Model & Brand acc. & Brand F1 & Era acc. & Era F1 & Both \\\\\\midrule\n"
          + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")


def table_price() -> None:
    base = load_json(paths.RESULTS_DIR / "fashiongen" / "baselines.json")
    if base is None or not base.get("price"):
        return
    pr = base["price"]
    rows = [("Global median", pr["no_image"]["global_median"]),
            ("Median of true brand", pr["no_image"]["true_brand_median"])]
    if "clip_vitb16" in pr:
        rows += [("Median of predicted brand (CLIP)", pr["clip_vitb16"]["predicted_brand_median"]),
                 ("Ridge on log price (CLIP)", pr["clip_vitb16"]["ridge"])]
    rows += [(f"$k$-NN retrieval ({SHORT_NAMES[b]}, $k$={pr[b]['knn']['param']})", pr[b]["knn"])
             for b in reversed(list(BACKBONE_NAMES)) if b in pr]
    body = "\n".join(f"{n} & {r['test']['mdape_pct']:.1f} & {pct(r['test']['within_25pct'])} & "
                     f"{r['test']['spearman']:.2f} \\\\" if np.isfinite(r['test']['spearman']) else
                     f"{n} & {r['test']['mdape_pct']:.1f} & {pct(r['test']['within_25pct'])} & --- \\\\"
                     for n, r in rows)
    write("price", "\\begin{table}[t]\\centering\\small\n"
          "\\caption{Price (retail MSRP) estimation on the Fashion-Gen test split. MdAPE = median absolute "
          "percentage error (lower is better); $\\rho$ = Spearman rank correlation.}\\label{tab:price}\n"
          "\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lccc}\\toprule\n"
          "Method & MdAPE & $\\leq$25\\% & $\\rho$ \\\\\\midrule\n" + body + "\n\\bottomrule\\end{tabular}\\end{table}\n")


def table_other() -> None:
    lines = []
    for ds, tasks in (("secondhand_labels", ("brand",)), ("europeana_runway", ("brand", "era"))):
        base = load_json(paths.RESULTS_DIR / ds / "baselines.json")
        if base is None:
            continue
        ncls = {t: len(base["classes"][t]) for t in tasks}
        head = ", ".join(f"{ncls[t]} {t}s" if t == "brand" else f"{ncls[t]} eras" for t in tasks)
        lines.append(f"\\multicolumn{{3}}{{l}}{{\\emph{{{DATASET_NAMES[ds]}}} ({head})}} \\\\")
        shortcuts = load_json(paths.RESULTS_DIR / ds / "shortcuts.json")
        if shortcuts:
            ch, th = shortcuts["chance"], shortcuts["features"]["thumbnail_8x8"]
            era_ch = pct(ch["era"]["majority_class_test_acc"]) if "era" in tasks else "---"
            era_th = pct(th["era"]["accuracy"]) if "era" in tasks else "---"
            lines.append(f"\\quad Majority class & {pct(ch['brand']['majority_class_test_acc'])} & {era_ch} \\\\")
            lines.append(f"\\quad 8$\\times$8 thumbnail probe & {pct(th['brand']['accuracy'])} & {era_th} \\\\")
        for backbone, res in base["results"].items():
            r = res["brand"]["linear_probe"]["test"]
            era = res["era"]["linear_probe"]["test"] if "era" in tasks else None
            lines.append(f"\\quad {SHORT_NAMES[backbone]} probe & {acc_ci(r)} & "
                         f"{acc_ci(era) if era else '---'} \\\\")
        for run in finetune_runs(ds).values():
            r = run["test"]["brand"]
            era = run["test"].get("era")
            lines.append(f"\\quad {ft_label(run, short=True)} & {acc_ci(r)} & {acc_ci(era) if era else '---'} \\\\")
        ocr = load_json(paths.RESULTS_DIR / ds / "ocr.json")
        if ocr:
            lines.append(f"\\quad OCR alone (answers {pct(ocr['ocr_coverage'])}\\%) & "
                         f"{pct(ocr['ocr_only_accuracy'])} & --- \\\\")
            lines.append(f"\\quad OCR, else CLIP probe & {acc_ci(ocr['ocr_then_image'])} & --- \\\\")
            runs = finetune_runs(ds)
            for name, fused in ocr.get("ocr_then_finetuned", {}).items():
                if name in runs:
                    lines.append(f"\\quad OCR, else {ft_label(runs[name], with_backbone=False)} & "
                                 f"{acc_ci(fused['ocr_then_image'])} & --- \\\\")
    if lines:
        write("other", "\\begin{table}[t]\\centering\\footnotesize\n"
              "\\caption{Test accuracy (\\%) on the label-photo and runway proxies.}\\label{tab:other}\n"
              "\\setlength{\\tabcolsep}{2pt}\\begin{tabular}{@{}lcc@{}}\\toprule\n"
              "Model & Brand & Era \\\\\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\\end{table}\n")


def robustness_assets(dataset: str) -> None:
    saved = load_json(paths.RESULTS_DIR / dataset / "robustness.json")
    if not saved:
        return
    rob = saved["models"]
    runs = finetune_runs(dataset)
    order = [k for k in rob if k.startswith("frozen_probe")] + sorted(
        [k for k in rob if k.startswith("finetune")], key=lambda k: ("strong" in k, k))
    label = {k: "frozen + linear probe" if k.startswith("frozen") else
             ft_label(runs[k.removeprefix("finetune_")], with_backbone=False) for k in order}
    backbones = sorted({k.removeprefix("frozen_probe_") for k in order if k.startswith("frozen")} |
                       {runs[k.removeprefix("finetune_")]["config"]["backbone"] for k in order if k.startswith("finetune")})
    name = DATASET_NAMES[dataset]
    viz.apply_style()
    fig = viz.robustness(rob, [(k, label[k]) for k in order], "brand", CORRUPTIONS)  # the caption names it
    viz.save(fig, f"robustness_{dataset}", FIGURES)
    # One column per model, one row per corruption (averaged over severities 1-3): fits a single column.
    per = {k: {c: np.mean([rob[k][c][s]["brand"] for s in rob[k][c]]) for c in CORRUPTIONS} for k in order}
    rows = [f"Clean & " + " & ".join(pct(rob[k]["clean"]["0"]["brand"]) for k in order) + " \\\\\\midrule"]
    rows += [f"{CORRUPTION_NAMES[c]} & " + " & ".join(pct(per[k][c]) for k in order) + " \\\\" for c in CORRUPTIONS]
    rows.append("\\midrule Mean of corruptions & " + " & ".join(pct(np.mean(list(per[k].values()))) for k in order)
                + " \\\\")
    heads = {k: "Frozen probe" if k.startswith("frozen") else
             "FT, " + runs[k.removeprefix("finetune_")]["config"]["aug"] for k in order}
    write(f"robustness_{dataset}", "\\begin{table}[t]\\centering\\small\n"
          f"\\caption{{{name} brand accuracy (\\%) under test-time corruptions, averaged over severities 1--3 "
          f"({', '.join(BACKBONE_NAMES[b] for b in backbones)}; FT = fine-tuned with light or strong "
          f"augmentation).}}\\label{{tab:robustness_{dataset}}}\n"
          f"\\begin{{tabular}}{{l{'c' * len(order)}}}\\toprule\n"
          "Condition & " + " & ".join(heads[k] for k in order) + " \\\\\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")


def confusion_figure() -> None:
    runs = finetune_runs("fashiongen")
    # Pick the run to plot on validation, never on test.
    best = max(runs.values(), key=lambda r: np.mean([r["val"][t]["macro_f1"] for t in r["classes"]]), default=None)
    if best is None:
        return
    short = {"Comme des Garcons": "CDG", "Maison Margiela": "Margiela", "Yohji Yamamoto": "Yohji",
             "Junya Watanabe": "Junya", "Issey Miyake": "Issey", "Helmut Lang": "H. Lang", "Raf Simons": "Raf"}
    viz.apply_style()
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1.5, 1]})
    viz.confusion(a1, best["test"]["brand"]["confusion"], [short.get(c, c) for c in best["classes"]["brand"]],
                  "Brand (row-normalised %)")
    viz.confusion(a2, best["test"]["era"]["confusion"], best["classes"]["era"], "Era / season year")
    fig.tight_layout()
    viz.save(fig, "confusion_fashiongen", FIGURES)


def samples_figure() -> None:
    rows = []
    n = 4  # images per row, sized for one column of the report
    picks = {"fashiongen": ("brand", ["Comme des Garcons", "Maison Margiela", "Undercover", "Yohji Yamamoto"]),
             "secondhand_labels": ("brand", None), "europeana_runway": ("era", None)}
    short = {"fashiongen": "Fashion-\nGen", "secondhand_labels": "Second-\nhand\nlabels", "europeana_runway": "Europeana\nrunway"}
    for ds, (col, values) in picks.items():
        if not paths.manifest_path(ds).exists():
            continue
        df = manifest.load(ds)
        test = df[df.split == "test"]
        values = values or sorted(test[col].unique())
        per = max(1, -(-n // len(values)))
        by_value = [list(test[test[col] == v].drop_duplicates("group_id").sample(per, random_state=0).path)
                    for v in values]
        chosen = [p for batch in zip(*by_value) for p in batch][:n]  # interleave so every class shows up
        rows.append((short[ds], [paths.dataset_dir(ds) / p for p in chosen]))
    if rows:
        viz.apply_style()
        viz.save(viz.image_grid(rows, cell=0.68), "samples", FIGURES)


def price_figure() -> None:
    df = manifest.load("fashiongen")
    base = load_json(paths.RESULTS_DIR / "fashiongen" / "baselines.json")
    if base is None or "clip_vitb16" not in base.get("price", {}):
        return
    from thriftfindr import features, price

    X = features.load("fashiongen", "clip_vitb16", df)
    p = df.price.to_numpy(dtype=float)
    ok = np.isfinite(p) & (p > 0)
    tr, te = (df.split == "train").to_numpy() & ok, (df.split == "test").to_numpy() & ok
    k = base["price"]["clip_vitb16"]["knn"]["param"]
    pred = price.knn_price(X[tr], p[tr], X[te], k)
    viz.apply_style()
    viz.save(viz.price_scatter(p[te], pred, f"k-NN retrieval price (CLIP, k={k})"), "price_scatter", FIGURES)


def compact_bibliography() -> None:
    """refs.bib = refs_verified.bib without URLs, DOIs, editors, publishers etc. (shorter reference list);
    @misc entries keep their URL and note, since those are the reference."""
    src = (paths.REPORT_DIR / "refs_verified.bib").read_text()
    out = []
    for entry in re.split(r"\n(?=@)", src.strip()):
        if not entry.lower().startswith("@misc"):
            for field in BIB_DROP:
                while (m := re.search(rf"\n\s*{field}\s*=\s*", entry, flags=re.I)):
                    i, depth = m.end(), 0
                    if entry[i] == "{":
                        for j in range(i, len(entry)):
                            depth += {"{": 1, "}": -1}.get(entry[j], 0)
                            if depth == 0:
                                break
                    else:  # quoted or bare value
                        j = entry.index(",", i) - 1 if "," in entry[i:] else len(entry) - 2
                    end = j + 1
                    if end < len(entry) and entry[end] == ",":
                        end += 1
                    entry = entry[:m.start()] + entry[end:]
        out.append(entry)
    (paths.REPORT_DIR / "refs.bib").write_text("\n\n".join(out) + "\n")


def main() -> None:
    compact_bibliography()
    table_datasets()
    table_fashiongen()
    table_price()
    table_other()
    for ds in ("secondhand_labels", "fashiongen"):
        robustness_assets(ds)
    confusion_figure()
    samples_figure()
    price_figure()
    print("tables:", sorted(p.name for p in TABLES.glob("*.tex")))
    print("figures:", sorted(p.name for p in FIGURES.glob("*.pdf")))


if __name__ == "__main__":
    main()
