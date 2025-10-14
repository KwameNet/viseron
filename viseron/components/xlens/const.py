"""Constants for the xlens component."""

COMPONENT = "xlens"
DESC_COMPONENT = "Object Detection component based on torchvision"

# Config keys
CONF_MODEL_NAME = "model_name"
CONF_CHECKPOINT_PATH = "checkpoint_path"
CONF_NUM_CLASSES = "num_classes"
CONF_SCORE_THRESHOLD = "score_threshold"
CONF_DEVICE = "device"
CONF_HALF_PRECISION = "half_precision"
CONF_MAX_DETECTIONS = "max_detections"
CONF_CATEGORY_INDEX = "category_index"  # {int_id: label}
CONFIG_OBJECT_DETECTOR = "object_detector"
CONFIG_LABEL_WIDTH_MIN = "min_width"

# Defaults
DEFAULT_SCORE_THRESHOLD = 0.4
DEFAULT_DEVICE = "auto"
DEFAULT_HALF_PRECISION = False
DEFAULT_MAX_DETECTIONS = 100
