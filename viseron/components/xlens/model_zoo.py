"""download models from mode zoo."""
from __future__ import annotations

from collections.abc import Callable

import torchvision
import torchvision.models.detection as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead

# Map friendly names to constructors. Extend as needed.
# NOTE: Torchvision detection models expect num_classes including background.


_ZOO: dict[str, Callable[..., object]] = {
    "fasterrcnn_resnet50_fpn": models.fasterrcnn_resnet50_fpn,
    "fasterrcnn_mobilenet_v3_large_fpn": models.fasterrcnn_mobilenet_v3_large_fpn,
    "retinanet_resnet50_fpn": models.retinanet_resnet50_fpn,
    "ssd300_vgg16": models.ssd300_vgg16,
}


def build_model(model_name: str, num_classes: int):
    """Build a model from the model zoo with a custom head."""
    if model_name not in _ZOO:
        raise ValueError(
            f"Unsupported model_name '{model_name}'. Supported: {sorted(_ZOO)}"
        )
    ctor = _ZOO[model_name]

    # Build with custom head
    if model_name.startswith("fasterrcnn"):
        # Create base model with default head; we'll swap the classifier.
        model = ctor(weights=None)
        if not hasattr(model, "roi_heads"):
            raise AttributeError(f"Model '{model_name}' does not have 'roi_heads'.")
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    if model_name.startswith("retinanet"):
        model = ctor(weights=None)

        if hasattr(model, "head") and hasattr(model.head, "classification_head"):
            # Replace classification head
            out_channels = model.head.classification_head.conv[0].in_channels
            num_anchors = model.head.classification_head.num_anchors
            model.head.classification_head = RetinaNetHead(
                out_channels, num_anchors, num_classes - 1
            )
        else:
            raise AttributeError(
                f"Model '{model_name}' does not have the expected "
                f"'head.classification_head' attributes."
            )
        return model

    if model_name == "ssd300_vgg16":
        model = ctor(weights=None)
        # SSD uses a different head replacement API; torchvision exposes a helper.
        # For custom num_classes, re-initialize with given num_classes directly.
        model = torchvision.models.detection.ssd300_vgg16(
            weights=None, num_classes=num_classes
        )
        return model

    # Fallback
    return ctor(weights=None)
