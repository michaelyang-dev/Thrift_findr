"""Run a fine-tuned checkpoint on any images: python scripts/predict.py CHECKPOINT IMAGE_OR_DIR... --out samples/predictions

Models with inputs (Fashion-Gen: the garment category) need them per image, either one value for all
images (--set category="JACKETS & COATS") or a CSV with a file-name column `file` and one column per
input (--inputs-csv). Writes predictions.csv (top class and confidence per task, price in USD) and,
with --annotate, a copy of every image with the predictions written underneath.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from thriftfindr.data.transforms import eval_transform
from thriftfindr.finetune import load_checkpoint
from thriftfindr.models.backbones import BACKBONES, default_device

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def image_paths(args) -> list[Path]:
    out = []
    for p in map(Path, args):
        out += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_SUFFIXES) if p.is_dir() else [p]
    return out


def annotate(img: Image.Image, lines: list[str], path: Path) -> None:
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    w = max(img.width, max(int(ImageDraw.Draw(img).textlength(s, font=font)) for s in lines) + 20)
    canvas = Image.new("RGB", (w, img.height + 10 + 26 * len(lines)), "white")
    canvas.paste(img, ((w - img.width) // 2, 0))
    d = ImageDraw.Draw(canvas)
    for k, s in enumerate(lines):
        d.text((10, img.height + 6 + 26 * k), s, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("images", nargs="+", help="image files or directories")
    parser.add_argument("--out", type=Path, default=Path("samples/predictions"))
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="COL=VALUE", help="input value for every image")
    parser.add_argument("--inputs-csv", type=Path, help="per-image inputs: column `file` (name) plus input columns")
    args = parser.parse_args()

    device = default_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    classes, stats, vocab = ckpt["classes"], ckpt.get("regress", {}), ckpt.get("inputs", {})
    fixed = dict(s.split("=", 1) for s in args.set)
    per_file = pd.read_csv(args.inputs_csv, dtype=str).set_index("file") if args.inputs_csv else None
    transform = eval_transform(BACKBONES[ckpt["config"]["backbone"]])
    rows = []
    for path in image_paths(args.images):
        given = {**fixed, **(per_file.loc[path.name].to_dict() if per_file is not None and path.name in per_file.index else {})}
        missing = [c for c in vocab if given.get(c) not in vocab[c]]
        if missing:
            raise SystemExit(f"{path.name}: give a valid {missing[0]!r}, one of {vocab[missing[0]]}")
        img = Image.open(path).convert("RGB")
        inputs = {c: torch.tensor([vocab[c].index(given[c])], device=device) for c in vocab}
        out = model(transform(img).unsqueeze(0).to(device), inputs)
        row, lines = {"file": str(path), **{c: given[c] for c in vocab}}, [f"{c}: {given[c]} (given)" for c in vocab]
        for t, names in classes.items():
            p = out[t].softmax(-1)[0].float().cpu()
            row[f"{t}_pred"], row[f"{t}_conf"] = names[int(p.argmax())], round(float(p.max()), 4)
            lines.append(f"{t}: {row[f'{t}_pred']} ({100 * row[f'{t}_conf']:.0f}%)")
        for t, s in stats.items():
            row[f"{t}_pred"] = round(float(torch.exp(s["mean"] + s["std"] * out[t][0].float().cpu())), 2)
            lines.append(f"{t}: ${row[f'{t}_pred']:,.0f}")
        rows.append(row)
        print(path.name, " | ".join(lines))
        if args.annotate:
            annotate(img, lines, args.out / path.name)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
