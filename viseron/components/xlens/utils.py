"""Utility functions for XLens component."""
from __future__ import annotations

import numpy as np
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device(spec: str):
    """Determine the torch device to use."""
    print(f"Getting device for spec: {spec}")
    if spec == "auto":
        if torch.cuda.is_available():

            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def to_tensor(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a BGR image to a normalized CHW tensor on the given device."""
    # BGR (OpenCV) -> RGB, HWC -> CHW, scale to [0,1], normalize.
    img = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
    img = (img - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(
        IMAGENET_STD, dtype=np.float32
    )
    tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()

    return tensor.to(device)
