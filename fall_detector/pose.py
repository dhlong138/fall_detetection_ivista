from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
]


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class DetectionResult:
    bbox: tuple[float, float, float, float]
    confidence: float
    keypoints: dict[str, Keypoint]
    track_id: int | None = None


class UltralyticsPoseEstimator:
    keypoint_names = COCO_KEYPOINT_NAMES

    def __init__(self, model_path: str, device: str = "cuda", imgsz: int = 640,
                 conf: float = 0.25, iou: float = 0.45, tracker: str = "bytetrack.yaml") -> None:
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device = "0" if str(device) == "cuda" else device
        self.imgsz, self.conf, self.iou, self.tracker = imgsz, conf, iou, tracker

    def infer(self, frame: np.ndarray) -> list[DetectionResult]:
        results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                     device=self.device, verbose=False)
        return self._convert_result(results[0] if results else None)

    def track(self, frame: np.ndarray) -> list[DetectionResult]:
        results = self.model.track(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                   device=self.device, tracker=self.tracker, persist=True, verbose=False)
        return self._convert_result(results[0] if results else None)

    def reset_tracking(self) -> None:
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    def _convert_result(self, result: Any) -> list[DetectionResult]:
        if result is None or result.boxes is None or result.keypoints is None:
            return []
        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.zeros(len(boxes_xyxy))
        track_ids = result.boxes.id.detach().cpu().numpy().astype(int) if getattr(result.boxes, "id", None) is not None else None
        points = result.keypoints.xy.detach().cpu().numpy()
        point_scores = result.keypoints.conf.detach().cpu().numpy() if result.keypoints.conf is not None else np.ones(points.shape[:2])
        return [DetectionResult(
            bbox=tuple(float(v) for v in box), confidence=float(scores[idx]) if idx < len(scores) else 0.0,
            keypoints={name: Keypoint(float(points[idx, key_idx, 0]), float(points[idx, key_idx, 1]), float(point_scores[idx, key_idx]))
                       for key_idx, name in enumerate(self.keypoint_names) if idx < len(points) and key_idx < points.shape[1]},
            track_id=int(track_ids[idx]) if track_ids is not None and idx < len(track_ids) else None,
        ) for idx, box in enumerate(boxes_xyxy)]


@dataclass
class Tracker:
    estimator: UltralyticsPoseEstimator
    enabled: bool = True

    def update(self, frame: np.ndarray) -> list[DetectionResult]:
        detections = self.estimator.track(frame) if self.enabled else self.estimator.infer(frame)
        for index, detection in enumerate(detections, start=1):
            if detection.track_id is None:
                detection.track_id = index
        return detections

    def reset(self) -> None:
        self.estimator.reset_tracking()
