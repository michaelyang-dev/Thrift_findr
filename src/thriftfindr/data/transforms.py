"""Training augmentations and test-time corruptions for phone photos of labels.

The augmentations simulate what goes wrong when a label is photographed in a store: blur,
glare from overhead lights or a flash, tilted framing (rotation), off-axis shots (perspective),
dim lighting and heavy JPEG compression. The same effects, at fixed severities, are used as
test-time corruptions to measure robustness (in the spirit of Hendrycks & Dietterich 2019).
"""

from __future__ import annotations

import io
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from thriftfindr.models.backbones import BackboneSpec

FILL = (124, 116, 104)  # ImageNet mean colour for pixels exposed by rotation / perspective
CORRUPTIONS = ("blur", "glare", "rotate", "perspective", "jpeg", "dark")
SEVERITIES = (1, 2, 3)  # severity 3 is deliberately harsher than anything seen in training
AUG_STRENGTHS = ("none", "light", "strong")


def apply_glare(img: Image.Image, cx: float, cy: float, rx: float, ry: float, strength: float) -> Image.Image:
    """Blend a soft bright ellipse over the image; (cx, cy, rx, ry) are fractions of width / height."""
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = ((xx - cx * w) / (rx * w)) ** 2 + ((yy - cy * h) / (ry * h)) ** 2
    mask = strength * np.exp(-2.0 * d2)[..., None]
    arr = np.asarray(img, dtype=np.float32)
    return Image.fromarray((arr + mask * (255.0 - arr)).clip(0, 255).astype(np.uint8))


class RandomGlare:
    def __init__(self, strength=(0.4, 0.9), radius=(0.12, 0.35)):
        self.strength, self.radius = strength, radius

    def __call__(self, img: Image.Image) -> Image.Image:
        r = random.uniform(*self.radius)
        return apply_glare(img, random.uniform(0.2, 0.8), random.uniform(0.2, 0.8), r,
                           r * random.uniform(0.5, 1.5), random.uniform(*self.strength))


class RandomJPEG:
    def __init__(self, quality=(20, 90)):
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        return _jpeg(img, random.randint(*self.quality))


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _interp(spec: BackboneSpec):
    return T.InterpolationMode.BICUBIC if spec.interpolation == "bicubic" else T.InterpolationMode.BILINEAR


def _to_tensor(spec: BackboneSpec):
    return [T.ToTensor(), T.Normalize(spec.mean, spec.std)]


def eval_transform(spec: BackboneSpec) -> T.Compose:
    return T.Compose([T.Resize(spec.resize, interpolation=_interp(spec)), T.CenterCrop(spec.crop),
                      *_to_tensor(spec)])


def train_transform(spec: BackboneSpec, strength: str = "strong", hflip: bool = True) -> T.Compose:
    """strength: "none" (eval transform), "light" (crop + flip), "strong" (all phone-photo effects)."""
    if strength not in AUG_STRENGTHS:
        raise ValueError(f"unknown augmentation strength {strength!r}; choose from {AUG_STRENGTHS}")
    if strength == "none":
        return eval_transform(spec)
    ops = [T.RandomResizedCrop(spec.crop, scale=(0.5 if strength == "strong" else 0.7, 1.0),
                               interpolation=_interp(spec))]
    if hflip:
        ops.append(T.RandomHorizontalFlip())
    if strength == "strong":
        ops += [
            T.RandomApply([T.RandomRotation(25, interpolation=T.InterpolationMode.BILINEAR, fill=FILL)], p=0.4),
            T.RandomPerspective(distortion_scale=0.35, p=0.3, fill=FILL),
            # A 9-tap kernel truncates large sigmas: the effective blur tops out near 2.2 px at 224 px.
            T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 3.0))], p=0.3),
            T.RandomApply([RandomGlare()], p=0.3),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.3, 0.03)], p=0.8),
            T.RandomGrayscale(p=0.05),
            T.RandomApply([RandomJPEG()], p=0.3),
        ]
    return T.Compose(ops + _to_tensor(spec))


def corrupt(img: Image.Image, kind: str, severity: int, rng: np.random.Generator) -> Image.Image:
    """Deterministic test-time corruption (severity 1-3) of a PIL image."""
    s = severity - 1
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=(1.0, 2.0, 3.5)[s]))
    if kind == "glare":
        r = (0.2, 0.28, 0.36)[s]
        return apply_glare(img, rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7), r, r, (0.6, 0.8, 0.95)[s])
    if kind == "rotate":
        angle = (15, 30, 45)[s] * rng.choice([-1, 1])
        return img.rotate(angle, resample=Image.BILINEAR, fillcolor=FILL)
    if kind == "perspective":
        w, h = img.size
        d = (0.2, 0.35, 0.5)[s] / 2
        start = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        jitter = rng.uniform(0, d, size=(4, 2)) * [w, h]
        signs = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        end = (np.array(start) + signs * jitter).round().astype(int).tolist()
        return TF.perspective(img, start, end, interpolation=T.InterpolationMode.BILINEAR, fill=FILL)
    if kind == "jpeg":
        return _jpeg(img, (30, 15, 8)[s])
    if kind == "dark":
        dim = ImageEnhance.Brightness(img).enhance((0.6, 0.4, 0.25)[s])
        arr = np.asarray(dim, dtype=np.float32) + rng.normal(0, (4, 8, 12)[s], size=(dim.size[1], dim.size[0], 3))
        return Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    raise ValueError(f"unknown corruption {kind!r}")


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: give python's `random` (used by the augmentations) a distinct seed."""
    random.seed(torch.initial_seed() % 2**32)
