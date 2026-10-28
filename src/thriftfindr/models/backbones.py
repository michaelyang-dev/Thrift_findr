"""Pretrained image encoders, used both as frozen feature extractors and as fine-tuning backbones.

Every encoder maps a normalized batch (B, 3, 224, 224) to one feature vector per image:
  clip_vitb16    CLIP ViT-B/16 (Radford et al. 2021), post-layernorm [CLS] token, 768-d
  dinov2_vits14  DINOv2 ViT-S/14 (Oquab et al. 2024), [CLS] + mean patch token, 768-d
                 (the linear-evaluation feature used in the DINOv2 paper)
  resnet50       ImageNet ResNet-50 (He et al. 2016), global average pool, 2048-d
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import certifi
import torch
from torch import nn
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # CLIPVisionModel otherwise lists every unused text-tower weight

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    source: str  # Hugging Face model id or torchvision weights enum name
    feat_dim: int
    resize: int  # shorter side before the center crop (matches each model's own preprocessing)
    crop: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    interpolation: str


BACKBONES = {
    "clip_vitb16": BackboneSpec("clip_vitb16", "openai/clip-vit-base-patch16", 768, 224, 224,
                                CLIP_MEAN, CLIP_STD, "bicubic"),
    "dinov2_vits14": BackboneSpec("dinov2_vits14", "facebook/dinov2-small", 768, 256, 224,
                                  IMAGENET_MEAN, IMAGENET_STD, "bicubic"),
    "resnet50": BackboneSpec("resnet50", "IMAGENET1K_V2", 2048, 232, 224,
                             IMAGENET_MEAN, IMAGENET_STD, "bilinear"),
}


class CLIPEncoder(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        from transformers import CLIPVisionModel

        self.model = CLIPVisionModel.from_pretrained(model_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).pooler_output


class DINOv2Encoder(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.model(pixel_values=x).last_hidden_state  # already layer-normed
        return torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=1)


class ResNetEncoder(nn.Module):
    def __init__(self, weights: str):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        # torchvision downloads through urllib, which on python.org macOS builds has no root
        # certificates unless "Install Certificates.command" was run; use certifi's bundle instead.
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        self.model = resnet50(weights=ResNet50_Weights[weights])
        self.model.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_encoder(name: str) -> nn.Module:
    spec = BACKBONES[name]
    if name.startswith("clip"):
        return CLIPEncoder(spec.source)
    if name.startswith("dinov2"):
        return DINOv2Encoder(spec.source)
    return ResNetEncoder(spec.source)


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
