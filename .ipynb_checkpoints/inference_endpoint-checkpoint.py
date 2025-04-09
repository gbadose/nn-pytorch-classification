"""
inference_endpoint.py  –  Optimized inference script
-----------------------------------------------------
-  single global logger & device
-  transforms built once (not every request)
-  model loaded, frozen, set to eval() & (optionally) compiled once
-  inference uses torch.inference_mode() for speed
-  half‑precision on GPU if available
-  clearer error handling & type hints
"""

from __future__ import annotations
import io, json, logging, os, sys
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

JPEG = "image/jpeg"
ACCEPTED_CONTENT_TYPES = {JPEG}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_HALF = DEVICE.type == "cuda"          # fp16 speeds things up on GPU

# Re‑use the same transform object for every call
TEST_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# Modele Definition
def build_model() -> nn.Module:
    """Return a frozen ResNet‑50 with two custom FC layers (133‑class output)."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 133),
    )
    return model


# Sagemaker Hooks
def model_fn(model_dir: str) -> nn.Module:
    """Load the model once at container start‑up."""
    LOGGER.info("Loading model from %s on %s", model_dir, DEVICE)
    model = build_model()
    model_path = Path(model_dir) / "model.pth"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval().to(DEVICE)

    if USE_HALF:                       # mixed‑precision speeds up GPU inference
        model.half()

    # PyTorch ≥2.0: compile for a small extra boost
    if hasattr(torch, "compile"):
        model = torch.compile(model)

    LOGGER.info("Model loaded and ready.")
    return model


def input_fn(body: bytes, content_type: str = JPEG) -> Image.Image:
    """Deserialize incoming JPEG bytes to a PIL image."""
    if content_type not in ACCEPTED_CONTENT_TYPES:
        raise ValueError(f"Unsupported Content‑Type: {content_type!r}")

    try:
        return Image.open(io.BytesIO(body)).convert("RGB")
    except Exception as exc:
        LOGGER.exception("Failed to read image")
        raise ValueError("Could not decode image") from exc


@torch.inference_mode()
def predict_fn(img: Image.Image, model: nn.Module) -> torch.Tensor:
    """Run a forward pass and return logits."""
    tensor = TEST_TRANSFORM(img).unsqueeze(0)              # (1, 3, 224, 224)
    if USE_HALF:
        tensor = tensor.half()
    tensor = tensor.to(DEVICE, non_blocking=True)
    return model(tensor)
