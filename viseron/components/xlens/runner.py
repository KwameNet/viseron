"""object detection runner using torchvision models."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .model_zoo import build_model
from .utils import get_device, to_tensor


class Detection:
    """Single object detection result."""

    def __init__(self, label: str, score: float, box: list[float]) -> None:
        self.label = label
        self.score = score
        self.box = box


class TorchvisionDetector:
    """Object detector using torchvision models."""

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        num_classes: int,
        score_threshold: float = 0.4,
        device: str = "auto",
        half_precision: bool = False,
        max_detections: int = 100,
        category_index: dict[int, str] | None = None,
    ) -> None:
        self.device = get_device(device)
        self.model = build_model(model_name, num_classes)

        state = torch.load(checkpoint_path, map_location="cpu")
        # Support both state dict directly or {"model_state_dict": ...}
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.score_threshold = float(score_threshold)
        self.max_detections = int(max_detections)
        self.category_index = category_index or {}

        self.use_amp = bool(half_precision) and self.device.type in ("cuda", "mps")

    @torch.inference_mode()
    def infer(self, frame_bgr) -> list[Detection]:
        """Run inference on a single BGR frame (numpy array)."""
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        tensor = to_tensor(image, self.device)
        inputs = [tensor]

        if self.use_amp and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model(inputs)
        else:
            outputs = self.model(inputs)

        outputs = outputs[0]  # Torchvision returns list[dict] per image
        boxes = outputs.get("boxes", [])
        scores = outputs.get("scores", [])
        labels = outputs.get("labels", [])

        detections: list[Detection] = []
        for i in range(min(len(scores), self.max_detections)):
            score = float(scores[i].item())
            if score < self.score_threshold:
                continue
            box = boxes[i].detach().float().tolist()
            cls_id = int(labels[i].item())
            label = self.category_index.get(cls_id, str(cls_id))
            detections.append(Detection(label=label, score=score, box=box))
        return detections
