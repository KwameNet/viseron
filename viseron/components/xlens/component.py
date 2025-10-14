"""x_lens component."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from .const import (
    COMPONENT,
    CONF_CATEGORY_INDEX,
    CONF_CHECKPOINT_PATH,
    CONF_DEVICE,
    CONF_HALF_PRECISION,
    CONF_MAX_DETECTIONS,
    CONF_MODEL_NAME,
    CONF_NUM_CLASSES,
    CONF_SCORE_THRESHOLD,
    DEFAULT_DEVICE,
    DEFAULT_HALF_PRECISION,
    DEFAULT_MAX_DETECTIONS,
    DEFAULT_SCORE_THRESHOLD,
    DESC_COMPONENT,
)
from .runner import TorchvisionDetector

LOGGER = logging.getLogger(__name__)


CONFIG_SCHEMA = vol.Schema(
    {
        vol.Required("component", description=DESC_COMPONENT): COMPONENT,
        vol.Required(
            CONF_MODEL_NAME, description="Torchvision model template name"
        ): str,
        vol.Required(
            CONF_CHECKPOINT_PATH, description="Path to state_dict .pt/.pth file"
        ): str,
        vol.Required(
            CONF_NUM_CLASSES, description="Number of classes (incl. background)"
        ): vol.All(int, vol.Range(min=2)),
        vol.Optional(CONF_SCORE_THRESHOLD, default=DEFAULT_SCORE_THRESHOLD): vol.All(
            float, vol.Range(min=0, max=1)
        ),
        vol.Optional(CONF_DEVICE, default=DEFAULT_DEVICE): str,
        vol.Optional(CONF_HALF_PRECISION, default=DEFAULT_HALF_PRECISION): bool,
        vol.Optional(CONF_MAX_DETECTIONS, default=DEFAULT_MAX_DETECTIONS): vol.All(
            int, vol.Range(min=1, max=1000)
        ),
        vol.Optional(CONF_CATEGORY_INDEX, default={}): dict,
    },
    extra=vol.ALLOW_EXTRA,
)


def setup(vis: Any, config: dict[str, Any]):
    """Entry point called by Viseron to initialize the component.

    Parameters
    ----------
    vis : object
    Viseron application core (provides bus/dispatcher, camera registry, etc.)
    config : dict
    Parsed configuration for this component (must satisfy CONFIG_SCHEMA).
    """
    validated = CONFIG_SCHEMA(config)

    detector = TorchvisionDetector(
        model_name=validated[CONF_MODEL_NAME],
        checkpoint_path=validated[CONF_CHECKPOINT_PATH],
        num_classes=validated[CONF_NUM_CLASSES],
        score_threshold=validated[CONF_SCORE_THRESHOLD],
        device=validated[CONF_DEVICE],
        half_precision=validated[CONF_HALF_PRECISION],
        max_detections=validated[CONF_MAX_DETECTIONS],
        category_index=validated[CONF_CATEGORY_INDEX],
    )

    LOGGER.info(
        "[%s] Loaded %s on %s (half_precision=%s)",
        COMPONENT,
        validated[CONF_MODEL_NAME],
        detector.device,
        validated[CONF_HALF_PRECISION],
    )

    # --- Wiring into Viseron ---
    # Below is a example of bridging to a hypothetical Viseron object-detection bus.
    # Adjust to your Viseron version:
    # - Subscribe to frames from cameras (vis.frames.subscribe)
    # - Publish detections to the object detection pipeline (vis.detections.publish)

    def on_frame(camera_name: str, frame_bgr):
        try:
            results = detector.infer(frame_bgr)
            # Transform to Viseron detection objects if needed
            payload = [
                {
                    "label": d.label,
                    "score": d.score,
                    "box": d.box,  # xyxy in pixel coords
                }
                for d in results
            ]
            vis.detections.publish(
                camera_name=camera_name, detector=COMPONENT, detections=payload
            )
        except (RuntimeError, ValueError) as err:  # pragma: no cover
            LOGGER.exception("[%s] inference error: %s", COMPONENT, err)

    # Subscribe this detector to all cameras that bound to this component.
    # If your Viseron requires explicit camera binding, adapt the filter here.
    vis.frames.subscribe(callback=on_frame, detector=COMPONENT)

    return True
