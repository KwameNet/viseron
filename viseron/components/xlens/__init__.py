"""x_lens component."""
import logging
from typing import Any

import voluptuous as vol

from viseron import Viseron
from viseron.domains import OptionalDomain, RequireDomain, setup_domain
from viseron.domains.motion_detector.const import DOMAIN as MOTION_DETECTOR_DOMAIN
from viseron.domains.object_detector import BASE_CONFIG_SCHEMA
from viseron.domains.object_detector.const import CONFIG_CAMERAS

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
    CONFIG_OBJECT_DETECTOR,
    DEFAULT_DEVICE,
    DEFAULT_HALF_PRECISION,
    DEFAULT_MAX_DETECTIONS,
    DEFAULT_SCORE_THRESHOLD,
    DESC_COMPONENT,
)

LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = vol.Schema(
    {
        vol.Required(COMPONENT, description=DESC_COMPONENT): vol.Schema(
            {
                vol.Required(
                    CONFIG_OBJECT_DETECTOR,
                    description="Configuration for the xlens object detector domain",
                ): BASE_CONFIG_SCHEMA.extend(
                    {
                        vol.Required(
                            CONF_MODEL_NAME,
                            description="Torchvision model template name",
                        ): str,
                        vol.Required(
                            CONF_CHECKPOINT_PATH,
                            description="Path to state_dict .pt/.pth file",
                        ): str,
                        vol.Required(
                            CONF_NUM_CLASSES,
                            description="Number of classes (incl. background)",
                        ): vol.All(int, vol.Range(min=2)),
                        vol.Optional(
                            CONF_SCORE_THRESHOLD,
                            description="Score threshold for detections",
                            default=DEFAULT_SCORE_THRESHOLD,
                        ): vol.All(float, vol.Range(min=0, max=1)),
                        vol.Optional(
                            CONF_DEVICE,
                            description="Device to run the model on",
                            default=DEFAULT_DEVICE,
                        ): str,
                        vol.Optional(
                            CONF_HALF_PRECISION,
                            description="Use half precision (FP16) for inference",
                            default=DEFAULT_HALF_PRECISION,
                        ): bool,
                        vol.Optional(
                            CONF_MAX_DETECTIONS,
                            description="Number of max detections per inference",
                            default=DEFAULT_MAX_DETECTIONS,
                        ): vol.All(int, vol.Range(min=1, max=1000)),
                        vol.Optional(
                            CONF_CATEGORY_INDEX,
                            description="Name mapping for class IDs",
                            default={},
                        ): dict,
                        vol.Optional(
                            CONF_MAX_DETECTIONS,
                            description="Number of max detections per inference",
                            default=DEFAULT_MAX_DETECTIONS,
                        ): vol.All(int, vol.Range(min=1, max=1000)),
                    }
                ),
            }
        ),
    },
    extra=vol.ALLOW_EXTRA,
)


def setup(vis: Viseron, config: dict[str, Any]) -> bool:
    """Set up the xlens component."""

    config = config[COMPONENT]

    if config.get(CONFIG_OBJECT_DETECTOR, None):
        for camera_identifier in config[CONFIG_OBJECT_DETECTOR][CONFIG_CAMERAS].keys():
            setup_domain(
                vis,
                COMPONENT,
                CONFIG_OBJECT_DETECTOR,
                config,
                identifier=camera_identifier,
                require_domains=[
                    RequireDomain(
                        domain="camera",
                        identifier=camera_identifier,
                    )
                ],
                optional_domains=[
                    OptionalDomain(
                        domain=MOTION_DETECTOR_DOMAIN,
                        identifier=camera_identifier,
                    ),
                ],
            )
    return True
