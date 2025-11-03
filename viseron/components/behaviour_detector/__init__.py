"""Per-camera behaviour detection component.

This component (per configured camera):
 - obtains person detections (runs YOLO if configured or consumes
   detections published to vis.data under DETECTIONS_PREFIX + camera_id),
 - runs a per-camera DeepSORT tracker if available,
 - extracts appearance features (ResNet) for confirmed tracks,
 - buffers per-track sequences per configured behaviour,
 - runs a behaviour-specific LSTM classifier when sequence length is reached,
 - publishes alerts to vis.data under BEHAVIOUR_ALERTS_PREFIX + camera_id.
Config example:
behaviour_detector:
  cameras:
    - camera_1
  yolo_model: "/path/to/yolov8.pt"    # optional
  behaviours:
    shoplifting:
      lstm_path: "/path/to/shop_lstm.pt"
      seq_len: 16
      threshold: 0.85
      input_size: 512
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any

import numpy as np
import torch
from torchvision import models, transforms as T

from viseron import Viseron
from viseron.domains.object_detector.const import EVENT_OBJECTS_IN_FOV
from viseron.domains.object_detector.detected_object import (
    DetectedObject,
    EventDetectedObjectsData,
)

from .config import (
    COMPONENT_REGISTRATION_KEY,
    CONF_BEHAVIOURS,
    CONF_CAMERAS,
    CONF_INPUT_SIZE,
    CONF_LSTM_PATH,
    CONF_REQUIRE_FRAME,
    CONF_SAMPLE_RATE,
    CONF_SEQ_LEN,
    CONF_THRESHOLD,
    CONF_YOLO_MODEL,
    DEFAULT_INPUT_SIZE,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SEQ_LEN,
    DEFAULT_THRESHOLD,
    behaviour_alerts_key,
    detections_key,
)
from .lstm_model import LSTMClassifier

LOGGER = logging.getLogger(__name__)

# Optional imports – component will degrade if missing
try:
    from ultralytics import YOLO

    _HAS_YOLO = True
except (ImportError, ModuleNotFoundError):
    YOLO = None
    _HAS_YOLO = False

try:

    from deep_sort_realtime.deepsort_tracker import DeepSort  # pyright: ignore

    _HAS_DEEPSORT = True
except (ImportError, ModuleNotFoundError):
    DeepSort = None
    _HAS_DEEPSORT = False


def setup(vis: Viseron, config: dict) -> bool:
    """Set up the behaviour detector component."""
    comp = BehaviourDetector(vis, config or {})
    # register under configured component registration key
    vis.data[COMPONENT_REGISTRATION_KEY] = comp
    return True


class BehaviourDetector:
    """Per-camera behaviour detector (generalized)."""

    def __init__(self, vis: Viseron, config: dict) -> None:
        self.vis = vis
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        # Config (use constants from config.py)
        self.cameras: list[str] = [str(c) for c in config.get(CONF_CAMERAS, [])]
        self.yolo_model_path: str | None = config.get(CONF_YOLO_MODEL)
        self.require_frame: bool = bool(config.get(CONF_REQUIRE_FRAME, True))
        self.sample_rate: int = int(config.get(CONF_SAMPLE_RATE, DEFAULT_SAMPLE_RATE))

        # frame ring buffers: camera_id -> deque of (timestamp, frame)
        self._frame_buffers: dict[str, deque] = defaultdict(deque)

        # optional: per-camera last known fps
        self._camera_fps: dict[str, float] = {}

        # behaviours: name -> config dict (lstm_path, seq_len, threshold, input_size)
        self.behaviours: dict[str, dict[str, Any]] = {}
        raw_beh = config.get(CONF_BEHAVIOURS, {}) or {}
        for name, bc in raw_beh.items():
            if not isinstance(bc, dict):
                continue
            self.behaviours[str(name)] = {
                CONF_LSTM_PATH: bc.get(CONF_LSTM_PATH),
                CONF_SEQ_LEN: int(bc.get(CONF_SEQ_LEN, DEFAULT_SEQ_LEN)),
                CONF_THRESHOLD: float(bc.get(CONF_THRESHOLD, DEFAULT_THRESHOLD)),
                CONF_INPUT_SIZE: int(bc.get(CONF_INPUT_SIZE, DEFAULT_INPUT_SIZE)),
            }

        self._alerts_max: int = int(config.get("alerts_max", 200))  # max alerts
        self._alerts_ttl: float = float(config.get("alerts_ttl", 3600.0))  # seconds;
        self._alerts: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._alerts_max)
        )
        self._alerts_lock = threading.Lock()

        # per-camera trackers and frame counters
        self._trackers: dict[str, object | None] = {}
        self._last_processed_ts: dict[str, float] = defaultdict(lambda: 0.0)
        # sample_rate interpreted as Hz (process at most sample_rate frames/second)
        self._sample_hz: float = float(
            config.get(CONF_SAMPLE_RATE, DEFAULT_SAMPLE_RATE)
        )

        # per-camera -> behaviour -> track_id -> deque(features)
        self._sequences: dict[str, dict[str, dict[str, deque]]] = defaultdict(
            lambda: defaultdict(dict)
        )

        # feature extractor (ResNet backbone). Default output dim 512 for resnet18
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet = models.resnet18(pretrained=True)
        resnet.fc = torch.nn.Identity()
        resnet.eval().to(device)
        self._reid_model = resnet
        self._device = device
        self._transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # load behaviour LSTM models
        self._behaviour_models: dict[str, torch.nn.Module | None] = {}
        for name, bc in self.behaviours.items():
            path = bc.get(CONF_LSTM_PATH)
            if path:
                try:
                    model = self._load_lstm(
                        path, input_size=bc.get(CONF_INPUT_SIZE, DEFAULT_INPUT_SIZE)
                    )
                    model.eval().to(self._device)
                    self._behaviour_models[name] = model
                    LOGGER.info("Loaded behaviour model '%s' from %s", name, path)
                except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
                    # Catch specific errors that can occur while loading model files
                    LOGGER.exception(
                        "Failed loading behaviour model '%s' from %s: %s",
                        name,
                        path,
                        exc,
                    )
                    self._behaviour_models[name] = None
            else:
                self._behaviour_models[name] = None

        # init per-camera DeepSORT trackers if available
        for cam in self.cameras:
            if _HAS_DEEPSORT:
                try:
                    self._trackers[cam] = DeepSort(max_age=15)
                except (RuntimeError, TypeError, ValueError) as exc:
                    LOGGER.exception(
                        "Failed to init DeepSort for camera %s: %s", cam, exc
                    )
                    self._trackers[cam] = None
            else:
                self._trackers[cam] = None
                LOGGER.debug(
                    "deep_sort_realtime not available; tracking disabled for camera %s",
                    cam,
                )

        # init YOLO if available
        self._yolo = None
        if _HAS_YOLO:
            try:
                self._yolo = (
                    YOLO(self.yolo_model_path) if self.yolo_model_path else YOLO()
                )
                LOGGER.info("YOLO initialized for behaviour detector")
            except (
                RuntimeError,
                TypeError,
                FileNotFoundError,
                ValueError,
                OSError,
            ) as exc:
                LOGGER.exception("Failed to initialize YOLO model: %s", exc)
                self._yolo = None
        else:
            LOGGER.debug("ultralytics YOLO not available; detection disabled")

        # start worker thread
        self._thread.start()

    def stop(self) -> None:
        """Stop the behaviour detector."""
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _load_lstm(
        self, path: str, input_size: int = DEFAULT_INPUT_SIZE
    ) -> torch.nn.Module:
        """Load LSTM model from given path."""

        model = LSTMClassifier(input_dim=input_size)
        ckpt = torch.load(path, map_location=self._device)
        # common formats
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            else:
                try:
                    model.load_state_dict(ckpt)
                except (RuntimeError, TypeError):
                    # try to find model keys
                    model.load_state_dict(
                        {k.replace("module.", ""): v for k, v in ckpt.items()}
                    )
        else:
            model.load_state_dict(ckpt)
        return model

    def _extract_feature(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        if crop_bgr.size == 0:
            return None
        img = crop_bgr[:, :, ::-1].copy()  # BGR -> RGB
        t = self._transform(img).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            feat = self._reid_model(t).cpu().numpy().reshape(-1)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

    def _predict_behaviour(self, behaviour: str, features: list[np.ndarray]) -> float:
        """Run behaviour-specific model and return score 0..1.

        If model missing return -1.
        """
        model = self._behaviour_models.get(behaviour)
        if not model:
            return -1.0
        x = np.stack(features, axis=0).astype(np.float32)  # (T, C)
        x_t = torch.from_numpy(x).unsqueeze(0).to(self._device)  # (1, T, C)
        with torch.inference_mode():
            out = model(x_t)
        return float(out.item()) if hasattr(out, "item") else float(out)

    def _get_camera_frame(self, cam_id: str) -> np.ndarray | None:
        cam_obj = (
            getattr(self.vis, "cameras", {}).get(cam_id)
            if hasattr(self.vis, "cameras")
            else None
        )
        if cam_obj is None:
            return None
        # try to get fps attribute once
        if cam_id not in self._camera_fps:
            fps = getattr(cam_obj, "fps", None) or getattr(cam_obj, "frame_rate", None)
            try:
                self._camera_fps[cam_id] = float(fps) if fps else 30.0
            except (ValueError, TypeError):
                self._camera_fps[cam_id] = 30.0
        for attr in ("latest_frame", "frame", "shared_frame", "current_frame"):
            frame = getattr(cam_obj, attr, None)
            if frame is not None:
                return frame
        get_fn = getattr(cam_obj, "get_latest_frame", None)
        if callable(get_fn):
            try:
                return get_fn()
            except (ValueError, TypeError):
                LOGGER.debug("camera %s get_latest_frame failed", cam_id, exc_info=True)
        return None

    def _get_detections_for_tracker(
        self, cam_id: str, frame
    ) -> list[tuple[list[float], float, int]]:
        """Return detections formatted for the tracker: (bbox, score, class_id)."""
        detections: list[tuple[list[float], float, int]] = []

        # Prefer local YOLO if available
        if self._yolo is not None and frame is not None:
            try:
                results = self._yolo(frame, classes=[0], verbose=False)
                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    for box, conf in zip(boxes, confs):
                        x1, y1, x2, y2 = map(float, box)
                        detections.append(([x1, y1, x2, y2], float(conf), 0))
            except (RuntimeError, ValueError) as err:
                # Catch expected inference errors, log and continue with no detections
                LOGGER.exception(
                    "YOLO inference failure for camera %s: %s", cam_id, err
                )
            # If you see other errors from the YOLO wrapper, add them above.

        # Fall back to published detections
        if not detections:
            dets = self.vis.data.get(detections_key(cam_id))
            if dets:
                for d in dets:
                    bbox = [d["x1"], d["y1"], d["x2"], d["y2"]]
                    conf = float(d.get("confidence", 0.0))
                    detections.append((bbox, conf, 0))

        return detections

    def _should_process(self, cam_id: str) -> bool:
        now = time.time()
        min_interval = 1.0 / max(self._sample_hz, 1e-6)
        last = self._last_processed_ts.get(cam_id, 0.0)
        if now - last < min_interval:
            return False
        self._last_processed_ts[cam_id] = now
        return True

    def _get_valid_frame(self, cam_id: str):
        frame = self._get_camera_frame(cam_id)
        if frame is None and self.require_frame:
            LOGGER.debug("No frame available for camera %s", cam_id)
            return None
        return frame

    def _update_tracker(self, cam_id: str, detections, frame):
        tracker = self._trackers.get(cam_id)
        if not tracker or not detections:
            return []
        # Ensure tracker actually implements update_tracks to avoid AttributeError
        update_fn = getattr(tracker, "update_tracks", None)
        if not callable(update_fn):
            LOGGER.debug(
                "camera %s's tracker doesn't implement update_tracks; skipping update",
                cam_id,
            )
            return []
        try:
            return update_fn(detections, frame=frame)
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            LOGGER.exception("DeepSort update failed for camera %s: %s", cam_id, exc)
            return []

    def _process_tracks(self, cam_id: str, tracks: list, frame) -> list:
        alerts = []
        for tr in tracks:
            if not self._is_valid_track(tr):
                continue

            tid, bbox = str(tr.track_id), self._get_bbox(tr)
            if bbox is None or frame is None:
                continue

            crop = self._crop_frame(frame, bbox)
            if crop is None:
                continue
            feat = self._extract_feature(crop)
            if feat is None:
                continue

            alerts.extend(self._evaluate_behaviours(cam_id, tid, bbox, feat))
        return alerts

    def _is_valid_track(self, tr) -> bool:
        """Return True if track is usable.

        - has is_confirmed() and is confirmed
        - has to_ltrb() that yields a 4-value bbox with positive area
        Any exception or missing attribute -> False
        """
        try:
            # prefer explicit confirmation check if available
            is_conf = True
            if hasattr(tr, "is_confirmed"):
                is_conf = bool(tr.is_confirmed())
            if not is_conf:
                return False

            # ensure bbox is valid
            ltrb = tr.to_ltrb()  # may raise
            if not ltrb or len(ltrb) < 4:
                return False
            x1, y1, x2, y2 = map(int, ltrb[:4])
            if x2 <= x1 or y2 <= y1:
                return False

            return True
        except (AttributeError, TypeError, ValueError, IndexError):
            # any expected error means track is not valid for processing
            return False

    def _get_bbox(self, tr) -> list[int] | None:
        """Safely get bounding box as [x1,y1,x2,y3] ints from a track object.

        Returns None on failure.
        """
        try:
            ltrb = tr.to_ltrb()
            if ltrb is None:
                return None
            # ensure exactly 4 values (or at least first 4)
            if len(ltrb) < 4:
                return None
            bbox = list(map(int, ltrb[:4]))
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                return None
            return bbox
        except (AttributeError, TypeError, ValueError, IndexError):
            return None

    def _crop_frame(
        self,
        frame: np.ndarray,
        bbox: list[int] | None,
        min_size: tuple[int, int] = (8, 8),
    ) -> np.ndarray | None:
        """Safely crop `frame` with bbox = [x1,y1,x2,y2].

        - clamps bbox to frame bounds
        - returns a copy of the crop (or None if invalid / too small)
        - min_size is (min_width, min_height) to avoid tiny crops
        """

        if bbox is None or len(bbox) < 4:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # clamp coordinates to valid range
        x1c = max(0, min(w - 1, int(x1)))
        y1c = max(0, min(h - 1, int(y1)))
        x2c = max(0, min(w, int(x2)))  # allow x2 == w (slice will handle)
        y2c = max(0, min(h, int(y2)))

        # ensure positive area after clamping
        if x2c <= x1c or y2c <= y1c:
            return None

        width = x2c - x1c
        height = y2c - y1c
        if width < min_size[0] or height < min_size[1]:
            # too small to extract meaningful features
            return None

        # return a copy so downstream code can modify safely
        try:
            return frame[y1c:y2c, x1c:x2c].copy()
        except (IndexError, TypeError, ValueError) as exc:
            LOGGER.debug("Failed cropping frame with bbox %s: %s", bbox, exc)
            return None

    def _evaluate_behaviours(self, cam_id, tid, bbox, feat):
        alerts = []
        for beh_name, bc in self.behaviours.items():
            seqs = (
                self._sequences[cam_id]
                .setdefault(beh_name, {})
                .setdefault(tid, deque(maxlen=bc[CONF_SEQ_LEN]))
            )
            seqs.append(feat)

            if len(seqs) < bc[CONF_SEQ_LEN]:
                continue

            score = self._predict_behaviour(beh_name, list(seqs))
            if score >= bc[CONF_THRESHOLD]:
                alerts.append(
                    {
                        "camera_id": cam_id,
                        "track_id": tid,
                        "behaviour": beh_name,
                        "score": float(score),
                        "bbox": bbox,
                        "timestamp": time.time(),
                    }
                )
                LOGGER.warning(
                    "Behaviour '%s' detected camera=%s track=%s score=%.3f",
                    beh_name,
                    cam_id,
                    tid,
                    score,
                )
                seqs.clear()
        return alerts

    def _get_shared_frame(self, cam_id: str):
        """Return current SharedFrame from the camera if available."""
        cam = None
        try:
            domains = getattr(self.vis, "domains", None)
            if isinstance(domains, dict):
                cam_domain = domains.get("camera") or domains.get(
                    "viseron.domains.camera"
                )
                if isinstance(cam_domain, dict):
                    cam = cam_domain.get(cam_id)
        except (AttributeError, TypeError):
            cam = None
        if cam is None:
            try:
                ffmpeg_comp = self.vis.data.get("ffmpeg")
                if isinstance(ffmpeg_comp, dict):
                    cam = ffmpeg_comp.get(cam_id)
            except (AttributeError, TypeError):
                cam = None
        return getattr(cam, "current_frame", None) if cam is not None else None

    def _make_detected_object(
        self,
        behaviour: str,
        score: float | None,
        bbox: list[int] | None,
        frame_shape: tuple[int, int] | None,
    ) -> DetectedObject | None:
        """Create a real DetectedObject from absolute bbox.

        returns None if not possible.
        """

        try:
            label = str(behaviour)
            conf = float(score) if score is not None else 0.0

            # Determine frame size
            if frame_shape and len(frame_shape) >= 2:
                h, w = int(frame_shape[0]), int(frame_shape[1])
            else:
                h, w = 1, 1  # safe fallback

            # Compute absolute box (defaults to full frame if bbox missing)
            #            if bbox and len(bbox) >= 4:
            if bbox and len(bbox) >= 4:
                x1, y1, x2, y2 = (int(v) for v in bbox[:4])
                # Clamp and ensure positive area
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(x1 + 1, min(w, x2))
                y2 = max(y1 + 1, min(h, y2))
            else:
                x1, y1, x2, y2 = 0, 0, max(1, w), max(1, h)

            # Construct DetectedObject with absolute coords + frame_res (positional)
            obj = DetectedObject(label, conf, x1, y1, x2, y2, (w, h))

            # Mark flags expected by the recorder pipeline
            obj.trigger_event_recording = True
            obj.store = True
            obj.relevant = True
            return obj
        except (ValueError, TypeError, IndexError, AttributeError):
            LOGGER.debug(
                "Failed to build DetectedObject for behaviour=%s",
                behaviour,
                exc_info=True,
            )
            return None

    def _emit_objects_in_fov(self, cam_id: str, alerts: list, frame) -> None:
        """Emit behaviour detections as EVENT_OBJECTS_IN_FOV for this camera."""
        if not alerts:
            return

        frame_shape = None
        try:
            if frame is not None and hasattr(frame, "shape"):
                frame_shape = (int(frame.shape[0]), int(frame.shape[1]))
        except (AttributeError, TypeError, ValueError):
            frame_shape = None

        objects: list[DetectedObject] = []
        for a in alerts:
            obj = self._make_detected_object(
                behaviour=str(a.get("behaviour", "behaviour")),
                score=a.get("score"),
                bbox=a.get("bbox"),
                frame_shape=frame_shape,
            )
            if obj is not None:
                objects.append(obj)

        if not objects:
            return

        shared_frame = self._get_shared_frame(cam_id)
        try:
            self.vis.dispatch_event(
                EVENT_OBJECTS_IN_FOV.format(camera_identifier=cam_id),
                EventDetectedObjectsData(
                    camera_identifier=cam_id,
                    shared_frame=shared_frame,
                    objects=objects,
                ),
            )
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError):
            LOGGER.debug(
                "Failed to dispatch EVENT_OBJECTS_IN_FOV for camera %s",
                cam_id,
                exc_info=True,
            )

    def _publish_alerts(self, cam_id, alerts):
        """Store recent alerts in bounded, TTL-pruned buffer and mirror to vis.data."""
        if not alerts:
            return
        now = time.time()
        with self._alerts_lock:
            dq = self._alerts[cam_id]  # deque with maxlen=self._alerts_max
            # append new alerts (ensure timestamp present)
            for a in alerts:
                if "timestamp" not in a:
                    a = {**a, "timestamp": now}
                dq.append(a)
            # TTL prune from the left
            if self._alerts_ttl > 0:
                cutoff = now - self._alerts_ttl
                while dq and float(dq[0].get("timestamp", now)) < cutoff:
                    dq.popleft()
            # mirror snapshot to vis.data for external readers
            self.vis.data[behaviour_alerts_key(cam_id)] = list(dq)

    def _run(self) -> None:
        LOGGER.info("Behaviour detector started for cameras: %s", self.cameras)
        while not self._stop.is_set():
            try:
                for cam_id in self.cameras:
                    if not self._should_process(cam_id):
                        continue
                    frame = self._get_valid_frame(cam_id)

                    detections = self._get_detections_for_tracker(cam_id, frame)
                    tracks = self._update_tracker(cam_id, detections, frame)
                    alerts = self._process_tracks(cam_id, tracks, frame)

                    # publish alerts for this camera
                    self._publish_alerts(cam_id, alerts)
                    # emit objects_in_fov so camera recorder can react
                    self._emit_objects_in_fov(cam_id, alerts, frame)

                time.sleep(0.01)
            except (
                RuntimeError,
                ValueError,
                TypeError,
                AttributeError,
                OSError,
            ) as exc:
                LOGGER.exception("Behaviour detector main loop error: %s", exc)
                time.sleep(1.0)
