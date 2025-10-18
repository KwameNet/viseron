"""xlens object detector."""

import logging

from viseron import Viseron
from viseron.domains.object_detector import AbstractObjectDetector
from viseron.domains.object_detector.detected_object import DetectedObject

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

LOGGER = logging.getLogger(__name__)


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
        return frame

    def return_objects(self, frame):
        """Perform object detection."""
        try:
            results = self._detector.infer(frame)
            # real frame resolution used by DetectedObject (height, width)
            frame_h, frame_w = frame.shape[:2]
            frame_res = (frame_h, frame_w)

            detections: list[DetectedObject] = []

            for d in results:
                # read raw coords (assume d.box = [x1, y1, x2, y2] or similar)
                x1_raw, y1_raw, x2_raw, y2_raw = (
                    float(d.box[0]),
                    float(d.box[1]),
                    float(d.box[2]),
                    float(d.box[3]),
                )

                # detect whether coords are absolute pixels (common heuristic)
                is_pixel_coords = any(
                    val > max(frame_w, frame_h) or val > 1.0
                    for val in (x1_raw, y1_raw, x2_raw, y2_raw)
                )

                if is_pixel_coords:
                    # convert pixels -> normalized [0,1]
                    x1 = x1_raw / frame_w
                    x2 = x2_raw / frame_w
                    y1 = y1_raw / frame_h
                    y2 = y2_raw / frame_h
                else:
                    # already normalized
                    x1, y1, x2, y2 = x1_raw, y1_raw, x2_raw, y2_raw

                # ensure values are clamped to [0,1]
                x1, x2 = max(0.0, min(x1, 1.0)), max(0.0, min(x2, 1.0))
                y1, y2 = max(0.0, min(y1, 1.0)), max(0.0, min(y2, 1.0))
                det = DetectedObject(
                    label=d.label,
                    confidence=float(d.score),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    frame_res=frame_res,
                )

                LOGGER.debug(
                    "[%s] detection label=%s score=%.3f abs=(%s,%s,%s,%s)",
                    COMPONENT,
                    det.label,
                    det.confidence,
                    getattr(det, "abs_x1", None),
                    getattr(det, "abs_x2", None),
                    getattr(det, "abs_y1", None),
                    getattr(det, "abs_y2", None),
                )

                detections.append(det)

            try:
                print(
                    ""
                    + str(detections[0].abs_x1)
                    + " "
                    + str(detections[0].abs_x2)
                    + " "
                    + str(detections[0].abs_y1)
                    + " "
                    + str(detections[0].abs_y2)
                    + " "
                    + str(detections[0].confidence)
                )
            except (AttributeError, IndexError):
                print("No absolute coordinates found.")
            return detections
        except (RuntimeError, ValueError) as err:  # pragma: no cover
            LOGGER.exception("[%s] inference error: %s", COMPONENT, err)
        return None
