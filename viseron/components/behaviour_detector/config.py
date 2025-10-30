"""Configuration constants and utility helpers for the behaviour detector component.

This module defines configuration keys, default values, and small helper
functions used by the behaviour detector (such as key construction for
camera-specific storage and clip output path formatting).
"""
from datetime import datetime
from typing import Final

COMPONENT: Final[str] = "behaviour_detector"
CONFIG_ROOT: Final[str] = "behaviour_detector"

CONF_CAMERAS: Final[str] = "cameras"
CONF_YOLO_MODEL: Final[str] = "yolo_model"
CONF_REQUIRE_FRAME: Final[str] = "require_frame"
CONF_SAMPLE_RATE: Final[str] = "sample_rate"
CONF_BEHAVIOURS: Final[str] = "behaviours"

CONF_LSTM_PATH: Final[str] = "lstm_path"
CONF_SEQ_LEN: Final[str] = "seq_len"
CONF_THRESHOLD: Final[str] = "threshold"
CONF_INPUT_SIZE: Final[str] = "input_size"

DEFAULT_SEQ_LEN: Final[int] = 16
DEFAULT_THRESHOLD: Final[float] = 0.85
DEFAULT_SAMPLE_RATE: Final[int] = 1
DEFAULT_INPUT_SIZE: Final[int] = 512
DEFAULT_TRACKER_MAX_AGE: Final[int] = 15

BEHAVIOUR_ALERTS_PREFIX: Final[str] = "behavior_alerts_"
DETECTIONS_PREFIX: Final[str] = "detections_"
COMPONENT_REGISTRATION_KEY: Final[
    str
] = COMPONENT  # vis.data[COMPONENT] holds the instance


def behaviour_alerts_key(camera_id: str) -> str:
    """Generate the behaviour alerts key for a specific camera."""
    return f"{BEHAVIOUR_ALERTS_PREFIX}{camera_id}"


def detections_key(camera_id: str) -> str:
    """Generate the detections key for a specific camera."""
    return f"{DETECTIONS_PREFIX}{camera_id}"


# --- Clip recording config ---
CONF_CLIP_PRE_SEC: Final[str] = "clip_pre_seconds"
CONF_CLIP_POST_SEC: Final[str] = "clip_post_seconds"
CONF_CLIP_OUTPUT_DIR: Final[str] = "clip_output_dir"

DEFAULT_CLIP_PRE_SEC: Final[float] = 3.0  # seconds before detection to include
DEFAULT_CLIP_POST_SEC: Final[float] = 5.0  # seconds after detection to include
DEFAULT_CLIP_OUTPUT_DIR: Final[str] = "/tmp/viseron_clips"


def clip_output_path(
    output_dir: str, camera_id: str, behaviour: str, timestamp: float
) -> str:
    """Generate the output path for a recorded behaviour clip."""

    t = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
    filename = f"{camera_id}_{behaviour}_{t}.mp4"
    return f"{output_dir.rstrip('/')}/{filename}"


# ...existing code...
