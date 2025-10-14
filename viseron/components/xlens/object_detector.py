"""xlens object detector."""

from viseron import Viseron
from viseron.domains.object_detector import AbstractObjectDetector

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
)
from .runner import TorchvisionDetector


def setup(vis: Viseron, config, identifier) -> bool:
    """Set up the xlens object_detector domain."""
    ObjectDetector(vis, config[CONFIG_OBJECT_DETECTOR], identifier)

    return True


class ObjectDetector(AbstractObjectDetector):
    """xlens object detection."""

    def __init__(self, vis: Viseron, config, camera_identifier) -> None:
        object_detector_config = config
        super().__init__(vis, COMPONENT, object_detector_config, camera_identifier)

        self._ds_config = object_detector_config

        self._detector = TorchvisionDetector(
            model_name=object_detector_config[CONF_MODEL_NAME],
            checkpoint_path=object_detector_config[CONF_CHECKPOINT_PATH],
            num_classes=object_detector_config[CONF_NUM_CLASSES],
            score_threshold=object_detector_config[CONF_SCORE_THRESHOLD],
            device=object_detector_config[CONF_DEVICE],
            half_precision=object_detector_config[CONF_HALF_PRECISION],
            max_detections=object_detector_config[CONF_MAX_DETECTIONS],
            category_index=object_detector_config[CONF_CATEGORY_INDEX],
        )

    def preprocess(self, frame):
        """Preprocess frame before detection."""
        return True

    def return_objects(self, frame):
        """Perform object detection."""
        return True
