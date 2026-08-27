from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from .pose import DetectionResult, Keypoint, Tracker, UltralyticsPoseEstimator
from .capture import LatestFrameCapture
from .settings import CONFIG, DEFAULT_CAMERA_URL, DEFAULT_STORAGE_BASE_PATH

from realtime_ai_publisher import (
    DEFAULT_BOOTSTRAP_SERVERS,
    DEFAULT_META_TYPE,
    SAMPLE_DIR,
    DEFAULT_TOPIC,
    KafkaFramePublisher,
    build_realtime_message,
)

try:
    from upload_image_to_minio import managed_upload_flow
except ImportError:
    if str(SAMPLE_DIR) not in sys.path:
        sys.path.insert(0, str(SAMPLE_DIR))
    from upload_image_to_minio import managed_upload_flow


_LEGACY_DEFAULT_CAMERA_URL = "rtsp://admin:admin1234!@192.168.1.8:554/0"
_LEGACY_DEFAULT_STORAGE_BASE_PATH = "server_assets/V3SStorage"

_LEGACY_CONFIG: dict[str, Any] = {
    "pose": {
        "device": "cuda",
        "pose_model": "yolo11x-pose.pt",
        "det_input_size": 640,
        "detector_score_thr": 0.10,
        "detector_nms_thr": 0.45,
        "keypoint_confidence": 0.35,
    },
    "tracking": {
        "enabled": True,
        "tracker": "bytetrack.yaml",
        "max_missing_frames": 30,
    },
    "temporal": {
        "window_size": 30,
        "min_history": 10,
    },
    "fall_detection": {
        "angle_threshold": 60.0,
        "angle_change_threshold": 35.0,
        "aspect_ratio_threshold": 1.15,
        "aspect_ratio_drop_threshold": 0.35,
        "velocity_threshold": 1.2,
        "acceleration_threshold": 2.5,
        "candidate_threshold": 0.55,
        "fall_threshold": 0.70,
        "recovery_threshold": 0.35,
        "persistence_frames": 5,
        "cooldown_frames": 20,
        "upright_angle_threshold": 35.0,
        "upright_aspect_ratio_threshold": 1.45,
        "require_observed_upright": True,
        "weights": {
            "angle": 0.20,
            "angle_change": 0.15,
            "aspect_ratio": 0.30,
            "aspect_ratio_change": 0.20,
            "velocity": 0.10,
            "acceleration": 0.05,
        },
    },
    "features": {
        "use_angle": True,
        "use_aspect_ratio": True,
        "use_velocity": True,
        "use_acceleration": True,
    },
}

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


@dataclass(frozen=True)
class _LegacyKeypoint:
    x: float
    y: float
    confidence: float


@dataclass
class _LegacyDetectionResult:
    bbox: tuple[float, float, float, float]
    confidence: float
    keypoints: dict[str, Keypoint]
    track_id: int | None = None


class _LegacyUltralyticsPoseEstimator:
    keypoint_names = COCO_KEYPOINT_NAMES

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = "0" if str(device) == "cuda" else device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.tracker = tracker

    def infer(self, frame: np.ndarray) -> list[DetectionResult]:
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        return self._convert_result(results[0] if results else None)

    def track(self, frame: np.ndarray) -> list[DetectionResult]:
        results = self.model.track(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            tracker=self.tracker,
            persist=True,
            verbose=False,
        )
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
        box_scores = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.zeros(len(boxes_xyxy))
        track_ids = None
        if getattr(result.boxes, "id", None) is not None:
            track_ids = result.boxes.id.detach().cpu().numpy().astype(int)

        keypoints_xy = result.keypoints.xy.detach().cpu().numpy()
        keypoint_scores = result.keypoints.conf.detach().cpu().numpy() if result.keypoints.conf is not None else np.ones(keypoints_xy.shape[:2])
        detections: list[DetectionResult] = []
        for idx, box in enumerate(boxes_xyxy):
            keypoints = {
                name: Keypoint(float(keypoints_xy[idx, j, 0]), float(keypoints_xy[idx, j, 1]), float(keypoint_scores[idx, j]))
                for j, name in enumerate(self.keypoint_names)
                if idx < len(keypoints_xy) and j < keypoints_xy.shape[1]
            }
            detections.append(
                DetectionResult(
                    bbox=tuple(float(v) for v in box),
                    confidence=float(box_scores[idx]) if idx < len(box_scores) else 0.0,
                    keypoints=keypoints,
                    track_id=int(track_ids[idx]) if track_ids is not None and idx < len(track_ids) else None,
                )
            )
        return detections


@dataclass
class _LegacyTracker:
    estimator: _LegacyUltralyticsPoseEstimator
    enabled: bool = True

    def update(self, frame: np.ndarray) -> list[DetectionResult]:
        detections = self.estimator.track(frame) if self.enabled else self.estimator.infer(frame)
        for idx, detection in enumerate(detections, start=1):
            if detection.track_id is None:
                detection.track_id = idx
        return detections

    def reset(self) -> None:
        self.estimator.reset_tracking()


Point = tuple[float, float]


def distance(a: Point, b: Point) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def midpoint(points: list[Point]) -> Point:
    arr = np.asarray(points, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def safe_div(num: float, denom: float, default: float = 0.0) -> float:
    return default if abs(denom) < 1e-9 else float(num / denom)


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def angle_from_vertical(vector: Point) -> float:
    vx, vy = vector
    norm = math.hypot(vx, vy)
    if norm < 1e-9:
        return float("nan")
    return float(math.degrees(math.acos(max(-1.0, min(1.0, abs(vy) / norm)))))


@dataclass
class BodyGeometry:
    valid: bool
    reason: str = ""
    shoulder_center: Point | None = None
    hip_center: Point | None = None
    knee_center: Point | None = None
    ankle_center: Point | None = None
    head_point: Point | None = None
    body_center: Point | None = None
    body_axis_start: Point | None = None
    body_axis_end: Point | None = None
    body_angle: float | None = None
    body_core_bbox: tuple[float, float, float, float] | None = None
    body_core_width: float | None = None
    body_core_height: float | None = None
    aspect_ratio: float | None = None
    shoulder_width: float | None = None
    hip_width: float | None = None
    torso_length: float | None = None
    head_to_hip_distance: float | None = None
    hip_to_ankle_distance: float | None = None
    body_scale: float | None = None
    pose_confidence: float = 0.0
    used_keypoints: list[str] = field(default_factory=list)


def _valid_point(kp: Keypoint | None, threshold: float) -> Point | None:
    if kp is None or kp.confidence < threshold:
        return None
    return (kp.x, kp.y)


def _single_or_center(detection: DetectionResult, left_name: str, right_name: str, threshold: float, used: list[str]) -> Point | None:
    left = _valid_point(detection.keypoints.get(left_name), threshold)
    right = _valid_point(detection.keypoints.get(right_name), threshold)
    if left and right:
        used.extend([left_name, right_name])
        return midpoint([left, right])
    if left:
        used.append(left_name)
        return left
    if right:
        used.append(right_name)
        return right
    return None


def _head_point(detection: DetectionResult, threshold: float, used: list[str]) -> Point | None:
    for name in ("nose", "left_ear", "right_ear", "left_eye", "right_eye"):
        point = _valid_point(detection.keypoints.get(name), threshold)
        if point:
            used.append(name)
            return point
    return None


def _segment_width(detection: DetectionResult, left_name: str, right_name: str, threshold: float) -> float | None:
    left = _valid_point(detection.keypoints.get(left_name), threshold)
    right = _valid_point(detection.keypoints.get(right_name), threshold)
    return distance(left, right) if left and right else None


def _body_core_points(detection: DetectionResult, threshold: float) -> list[Point]:
    core_names = (
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )
    return [
        point
        for name in core_names
        if (point := _valid_point(detection.keypoints.get(name), threshold)) is not None
    ]


def compute_body_geometry(detection: DetectionResult, keypoint_threshold: float) -> BodyGeometry:
    used: list[str] = []
    shoulder_center = _single_or_center(detection, "left_shoulder", "right_shoulder", keypoint_threshold, used)
    hip_center = _single_or_center(detection, "left_hip", "right_hip", keypoint_threshold, used)
    knee_center = _single_or_center(detection, "left_knee", "right_knee", keypoint_threshold, used)
    ankle_center = _single_or_center(detection, "left_ankle", "right_ankle", keypoint_threshold, used)
    head = _head_point(detection, keypoint_threshold, used)

    if hip_center is None:
        return BodyGeometry(False, "missing hip center", pose_confidence=detection.confidence, used_keypoints=used)
    axis_top = head or shoulder_center
    if axis_top is None:
        return BodyGeometry(False, "missing head/shoulder axis point", pose_confidence=detection.confidence, used_keypoints=used)

    body_vector = (axis_top[0] - hip_center[0], axis_top[1] - hip_center[1])
    if math.hypot(*body_vector) < 1e-6:
        return BodyGeometry(False, "degenerate body axis", pose_confidence=detection.confidence, used_keypoints=used)

    core_points = _body_core_points(detection, keypoint_threshold)
    if len(core_points) < 2:
        return BodyGeometry(False, "insufficient body core points", pose_confidence=detection.confidence, used_keypoints=used)

    xs = [p[0] for p in core_points]
    ys = [p[1] for p in core_points]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    torso_length = distance(shoulder_center, hip_center) if shoulder_center else None
    head_to_hip = distance(head, hip_center) if head else None
    hip_to_ankle = distance(hip_center, ankle_center) if ankle_center else None
    scale_candidates = [v for v in (head_to_hip, torso_length, height) if v is not None and v > 1.0]
    body_scale = max(scale_candidates) if scale_candidates else height

    return BodyGeometry(
        valid=True,
        shoulder_center=shoulder_center,
        hip_center=hip_center,
        knee_center=knee_center,
        ankle_center=ankle_center,
        head_point=head,
        body_center=hip_center,
        body_axis_start=hip_center,
        body_axis_end=axis_top,
        body_angle=angle_from_vertical(body_vector),
        body_core_bbox=(x1, y1, x2, y2),
        body_core_width=width,
        body_core_height=height,
        aspect_ratio=safe_div(height, width, default=0.0),
        shoulder_width=_segment_width(detection, "left_shoulder", "right_shoulder", keypoint_threshold),
        hip_width=_segment_width(detection, "left_hip", "right_hip", keypoint_threshold),
        torso_length=torso_length,
        head_to_hip_distance=head_to_hip,
        hip_to_ankle_distance=hip_to_ankle,
        body_scale=body_scale,
        pose_confidence=detection.confidence,
        used_keypoints=used,
    )


@dataclass
class FeatureVector:
    frame: int
    timestamp: float
    track_id: int
    valid: bool
    body_angle: float | None = None
    body_angle_velocity: float | None = None
    body_angle_acceleration: float | None = None
    body_core_width: float | None = None
    body_core_height: float | None = None
    aspect_ratio: float | None = None
    aspect_ratio_velocity: float | None = None
    center_x: float | None = None
    center_y: float | None = None
    velocity_x: float | None = None
    velocity_y: float | None = None
    acceleration_x: float | None = None
    acceleration_y: float | None = None
    vertical_velocity: float | None = None
    vertical_acceleration: float | None = None
    speed: float | None = None
    acceleration: float | None = None
    shoulder_width: float | None = None
    hip_width: float | None = None
    torso_length: float | None = None
    head_to_hip_distance: float | None = None
    hip_to_ankle_distance: float | None = None
    body_scale: float | None = None
    pose_confidence: float = 0.0
    fall_score: float = 0.0
    state: str = "NORMAL"


def _diff(a: float | None, b: float | None, dt: float) -> float | None:
    if a is None or b is None or dt <= 0:
        return None
    return (a - b) / dt


def compute_motion_features(
    frame_index: int,
    timestamp: float,
    track_id: int,
    geometry: BodyGeometry,
    previous: FeatureVector | None,
    dt: float,
) -> FeatureVector:
    if not geometry.valid:
        return FeatureVector(frame_index, timestamp, track_id, valid=False, pose_confidence=geometry.pose_confidence)

    scale = geometry.body_scale or 1.0
    cx, cy = geometry.body_center or (None, None)
    vx = vy = ax = ay = None
    angle_velocity = angle_acceleration = aspect_velocity = None
    if previous and previous.valid and cx is not None and cy is not None:
        vx = _diff(cx, previous.center_x, dt)
        vy = _diff(cy, previous.center_y, dt)
        if vx is not None:
            vx /= scale
        if vy is not None:
            vy /= scale
        if previous.velocity_x is not None and vx is not None:
            ax = _diff(vx, previous.velocity_x, dt)
        if previous.velocity_y is not None and vy is not None:
            ay = _diff(vy, previous.velocity_y, dt)
        angle_velocity = _diff(geometry.body_angle, previous.body_angle, dt)
        if angle_velocity is not None and previous.body_angle_velocity is not None:
            angle_acceleration = _diff(angle_velocity, previous.body_angle_velocity, dt)
        aspect_velocity = _diff(geometry.aspect_ratio, previous.aspect_ratio, dt)

    speed = math.hypot(vx or 0.0, vy or 0.0) if vx is not None or vy is not None else None
    acceleration = math.hypot(ax or 0.0, ay or 0.0) if ax is not None or ay is not None else None
    return FeatureVector(
        frame=frame_index,
        timestamp=timestamp,
        track_id=track_id,
        valid=True,
        body_angle=geometry.body_angle,
        body_angle_velocity=angle_velocity,
        body_angle_acceleration=angle_acceleration,
        body_core_width=geometry.body_core_width,
        body_core_height=geometry.body_core_height,
        aspect_ratio=geometry.aspect_ratio,
        aspect_ratio_velocity=aspect_velocity,
        center_x=cx,
        center_y=cy,
        velocity_x=vx,
        velocity_y=vy,
        acceleration_x=ax,
        acceleration_y=ay,
        vertical_velocity=vy,
        vertical_acceleration=ay,
        speed=speed,
        acceleration=acceleration,
        shoulder_width=(geometry.shoulder_width / scale) if geometry.shoulder_width else None,
        hip_width=(geometry.hip_width / scale) if geometry.hip_width else None,
        torso_length=(geometry.torso_length / scale) if geometry.torso_length else None,
        head_to_hip_distance=(geometry.head_to_hip_distance / scale) if geometry.head_to_hip_distance else None,
        hip_to_ankle_distance=(geometry.hip_to_ankle_distance / scale) if geometry.hip_to_ankle_distance else None,
        body_scale=scale,
        pose_confidence=geometry.pose_confidence,
    )


@dataclass
class PersonState:
    track_id: int
    window_size: int
    features: deque[FeatureVector] = field(init=False)
    current_state: str = "NORMAL"
    candidate_count: int = 0
    recovery_count: int = 0
    cooldown_remaining: int = 0
    missing_frames: int = 0
    observed_upright: bool = False
    fall_hold_remaining: int = 0
    last_publish_item: tuple[DetectionResult, BodyGeometry, FeatureVector] | None = None

    def __post_init__(self) -> None:
        self.features = deque(maxlen=self.window_size)

    def add(self, feature: FeatureVector) -> None:
        self.features.append(feature)
        self.missing_frames = 0

    @property
    def previous(self) -> FeatureVector | None:
        return self.features[-1] if self.features else None

    def recent(self, n: int | None = None) -> list[FeatureVector]:
        values = list(self.features)
        return values[-n:] if n else values


class RuleBasedFallClassifier:
    def __init__(self, config: dict[str, Any]) -> None:
        self.cfg = config["fall_detection"]
        self.features_cfg = config.get("features", {})
        weights = self.cfg.get("weights", {})
        self.weights = {
            "angle": float(weights.get("angle", 0.20)),
            "angle_change": float(weights.get("angle_change", 0.15)),
            "aspect_ratio": float(weights.get("aspect_ratio", 0.30)),
            "aspect_ratio_change": float(weights.get("aspect_ratio_change", 0.20)),
            "velocity": float(weights.get("velocity", 0.10)),
            "acceleration": float(weights.get("acceleration", 0.05)),
        }

    def predict(self, person_state: PersonState) -> float:
        recent = [f for f in person_state.recent() if f.valid]
        if len(recent) < 2:
            return 0.0
        cur = recent[-1]
        if cur.body_angle is None or cur.aspect_ratio is None:
            return 0.0

        upright_angle = float(self.cfg["upright_angle_threshold"])
        upright_ratio = float(self.cfg["upright_aspect_ratio_threshold"])
        if cur.body_angle < upright_angle and cur.aspect_ratio > upright_ratio:
            person_state.observed_upright = True
        if bool(self.cfg.get("require_observed_upright", True)) and not person_state.observed_upright:
            return 0.0

        first = recent[0]
        max_angle_change = max(
            abs((f.body_angle or cur.body_angle) - (first.body_angle or cur.body_angle)) for f in recent
        )
        max_aspect_drop = max((first.aspect_ratio or cur.aspect_ratio) - (f.aspect_ratio or cur.aspect_ratio) for f in recent)
        angle_score = clamp01((cur.body_angle - upright_angle) / max(1.0, float(self.cfg["angle_threshold"]) - upright_angle))
        angle_change_score = clamp01(max_angle_change / max(1.0, float(self.cfg["angle_change_threshold"])))
        aspect_score = clamp01((float(self.cfg["aspect_ratio_threshold"]) - cur.aspect_ratio) / float(self.cfg["aspect_ratio_threshold"]))
        aspect_change_score = clamp01(max_aspect_drop / max(0.01, float(self.cfg["aspect_ratio_drop_threshold"])))
        velocity_score = clamp01(abs(cur.vertical_velocity or 0.0) / max(0.01, float(self.cfg["velocity_threshold"])))
        acceleration_score = clamp01(abs(cur.vertical_acceleration or 0.0) / max(0.01, float(self.cfg["acceleration_threshold"])))
        parts = {
            "angle": angle_score if self.features_cfg.get("use_angle", True) else 0.0,
            "angle_change": angle_change_score,
            "aspect_ratio": aspect_score if self.features_cfg.get("use_aspect_ratio", True) else 0.0,
            "aspect_ratio_change": aspect_change_score,
            "velocity": velocity_score if self.features_cfg.get("use_velocity", True) else 0.0,
            "acceleration": acceleration_score if self.features_cfg.get("use_acceleration", True) else 0.0,
        }
        enabled_parts = {
            "angle": self.features_cfg.get("use_angle", True),
            "angle_change": True,
            "aspect_ratio": self.features_cfg.get("use_aspect_ratio", True),
            "aspect_ratio_change": True,
            "velocity": self.features_cfg.get("use_velocity", True),
            "acceleration": self.features_cfg.get("use_acceleration", True),
        }
        total_weight = sum(self.weights[k] for k, enabled in enabled_parts.items() if enabled) or sum(self.weights.values())
        return clamp01(sum(parts[k] * self.weights[k] for k in parts) / total_weight)


class FallStateMachine:
    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config["fall_detection"]
        self.candidate_threshold = float(cfg["candidate_threshold"])
        self.fall_threshold = float(cfg["fall_threshold"])
        self.recovery_threshold = float(cfg["recovery_threshold"])
        self.persistence_frames = int(cfg["persistence_frames"])
        self.cooldown_frames = int(cfg["cooldown_frames"])
        self.upright_angle = float(cfg["upright_angle_threshold"])
        self.upright_ratio = float(cfg["upright_aspect_ratio_threshold"])
        self.horizontal_ratio = float(cfg["aspect_ratio_threshold"])

    def update(self, person: PersonState, feature: FeatureVector, fall_score: float) -> str:
        if person.cooldown_remaining > 0:
            person.cooldown_remaining -= 1
        is_recovered_pose = (
            feature.valid
            and feature.body_angle is not None
            and feature.aspect_ratio is not None
            and feature.body_angle < self.upright_angle
            and feature.aspect_ratio > self.upright_ratio
        )
        is_fall_like_pose = (
            feature.valid
            and feature.body_angle is not None
            and feature.aspect_ratio is not None
            and feature.body_angle >= self.upright_angle
            and feature.aspect_ratio <= self.horizontal_ratio
        )
        if person.current_state == "NORMAL":
            if fall_score >= self.candidate_threshold and is_fall_like_pose and person.cooldown_remaining == 0:
                person.current_state = "FALL_CANDIDATE"
                person.candidate_count = 1
            else:
                person.candidate_count = 0
        elif person.current_state == "FALL_CANDIDATE":
            if is_recovered_pose:
                person.current_state = "NORMAL"
                person.candidate_count = 0
            elif not is_fall_like_pose:
                person.current_state = "NORMAL"
                person.candidate_count = 0
            elif fall_score >= self.fall_threshold and is_fall_like_pose:
                person.candidate_count += 1
                if person.candidate_count >= max(1, self.persistence_frames):
                    person.current_state = "FALL"
                    person.cooldown_remaining = self.cooldown_frames
            elif fall_score < self.recovery_threshold:
                person.current_state = "NORMAL"
                person.candidate_count = 0
        elif person.current_state == "FALL":
            # Recover either when the person is visibly upright again or when
            # the complete fall evidence stays below the recovery threshold.
            # Without the score condition, a single false trigger can remain
            # FALL indefinitely for a horizontal/crouched pose.
            if is_recovered_pose or fall_score < self.recovery_threshold:
                person.recovery_count += 1
                if person.recovery_count >= self.persistence_frames:
                    person.current_state = "NORMAL"
                    person.candidate_count = 0
                    person.recovery_count = 0
                    person.cooldown_remaining = self.cooldown_frames
            else:
                person.recovery_count = 0
        feature.fall_score = fall_score
        feature.state = person.current_state
        return person.current_state


class Visualizer:
    def __init__(self, keypoint_threshold: float, debug: bool = False) -> None:
        self.keypoint_threshold = keypoint_threshold
        self.debug = debug

    def draw(self, frame: np.ndarray, detection: DetectionResult, geometry: BodyGeometry, feature: FeatureVector) -> np.ndarray:
        color = self._state_color(feature.state)
        self._draw_skeleton(frame, detection)
        display_bbox = geometry.body_core_bbox if geometry.valid and geometry.body_core_bbox else detection.bbox
        x1, y1, x2, y2 = self._safe_box(display_bbox, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        if geometry.valid and geometry.body_axis_start and geometry.body_axis_end:
            cv2.line(frame, self._pt(geometry.body_axis_start), self._pt(geometry.body_axis_end), (0, 255, 255), 3)
        if self.debug and geometry.valid:
            dx1, dy1, dx2, dy2 = self._safe_box(detection.bbox, frame)
            cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (120, 120, 120), 1)
            self._draw_debug_points(frame, geometry)
        self._draw_panel(frame, x1, max(0, y1 - 8), feature, color)
        return frame

    def _draw_skeleton(self, frame: np.ndarray, detection: DetectionResult) -> None:
        for name, kp in detection.keypoints.items():
            if kp.confidence >= self.keypoint_threshold:
                cv2.circle(frame, (int(kp.x), int(kp.y)), 3, (60, 220, 60), -1)
        for a, b in SKELETON_EDGES:
            ka = detection.keypoints.get(a)
            kb = detection.keypoints.get(b)
            if ka and kb and ka.confidence >= self.keypoint_threshold and kb.confidence >= self.keypoint_threshold:
                cv2.line(frame, (int(ka.x), int(ka.y)), (int(kb.x), int(kb.y)), (80, 180, 255), 2)

    def _draw_debug_points(self, frame: np.ndarray, geometry: BodyGeometry) -> None:
        points: Iterable[tuple[str, Point | None, tuple[int, int, int]]] = [
            ("S", geometry.shoulder_center, (255, 0, 255)),
            ("H", geometry.hip_center, (0, 255, 255)),
            ("K", geometry.knee_center, (255, 255, 0)),
            ("A", geometry.ankle_center, (255, 255, 255)),
            ("C", geometry.body_center, (0, 128, 255)),
        ]
        for label, point, color in points:
            if point:
                pt = self._pt(point)
                cv2.circle(frame, pt, 5, color, -1)
                cv2.putText(frame, label, (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def _draw_panel(self, frame: np.ndarray, x: int, y: int, feature: FeatureVector, color: tuple[int, int, int]) -> None:
        lines = [
            f"ID: {feature.track_id:02d}",
            f"State: {feature.state}",
            f"Score: {feature.fall_score:.2f}",
            f"Angle: {self._fmt(feature.body_angle, 1)} deg",
            f"Ratio: {self._fmt(feature.aspect_ratio, 2)}",
            f"Vel: {self._fmt(feature.speed, 2)}",
            f"Acc: {self._fmt(feature.acceleration, 2)}",
        ]
        panel_w, panel_h = 190, 22 * len(lines) + 8
        top = min(max(0, y - panel_h), max(0, frame.shape[0] - panel_h - 1))
        left = min(max(0, x), max(0, frame.shape[1] - panel_w - 1))
        overlay = frame.copy()
        cv2.rectangle(overlay, (left, top), (left + panel_w, top + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (left, top), (left + panel_w, top + panel_h), color, 2)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (left + 8, top + 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)

    @staticmethod
    def _pt(point: Point) -> tuple[int, int]:
        return int(point[0]), int(point[1])

    @staticmethod
    def _safe_box(bbox: tuple[float, float, float, float], frame: np.ndarray) -> tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        values = np.asarray(bbox, dtype=float).reshape(-1)[:4]
        values[0::2] = np.clip(values[0::2], 0, max(0, w - 1))
        values[1::2] = np.clip(values[1::2], 0, max(0, h - 1))
        x1, y1, x2, y2 = [int(round(float(v))) for v in values]
        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)
        return x1, y1, x2, y2

    @staticmethod
    def _fmt(value: float | None, ndigits: int) -> str:
        return "n/a" if value is None else f"{value:.{ndigits}f}"

    @staticmethod
    def _state_color(state: str) -> tuple[int, int, int]:
        if state == "FALL":
            return (0, 0, 255)
        if state == "FALL_CANDIDATE":
            return (0, 255, 255)
        return (0, 200, 0)


def mask_url(url: str) -> str:
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, f"***:***@{host}", parts.path, parts.query, parts.fragment))


def open_stream(url: str, buffer_size: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    return cap


def generate_sample_image_path(device_id: str, base_path: str, now: datetime | None = None) -> Path:
    now = datetime.now() if now is None else now
    date_path = now.strftime("%Y/%m/%d")
    filename = f"{device_id}_{now.strftime('%Y_%m_%d_%H_%M_%S_%f')}.jpg"
    return Path(base_path) / "images" / date_path / str(device_id) / filename


def materialize_camera_frame(frame: np.ndarray, device_id: str, base_path: str) -> Path:
    destination = generate_sample_image_path(device_id, base_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), frame):
        raise RuntimeError(f"Cannot write camera frame image: {destination}")
    return destination


class _LegacyLatestFrameCapture:
    def __init__(
        self,
        url: str,
        buffer_size: int,
        reconnect_delay: float,
        loop_video: bool = True,
        video_loop_count: int = 0,
    ) -> None:
        self.url = url
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        self.loop_video = loop_video
        self.video_loop_count = max(0, video_loop_count)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self._frame = None
        self._seq = 0
        self._delivered_seq = 0
        self._opened = False
        self._loop_count = 0
        self._completed_passes = 0
        self._finished = threading.Event()
        source_scheme = urlsplit(self.url).scheme.lower()
        # urlsplit treats a Windows drive letter (for example C:\\video.mp4)
        # as a URL scheme. Check the filesystem first so local videos are paced
        # at their native FPS instead of being decoded as fast as possible.
        self._is_file_source = Path(self.url).is_file() or source_scheme in {"", "file"}
        self._file_fps = 0.0
        self._next_file_frame_at = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()

    def read_latest(self, last_seq: int, timeout: float = 1.0):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._lock:
                if self._seq != last_seq and self._frame is not None:
                    frame = self._frame.copy()
                    seq = self._seq
                    self._delivered_seq = seq
                    return True, frame, seq
            time.sleep(0.002)
        return False, None, last_seq

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()

    def is_finished(self) -> bool:
        return self._finished.is_set()

    def source_frame_interval(self) -> float | None:
        if self._is_file_source and self._file_fps > 0:
            return 1.0 / self._file_fps
        return None

    def loop_generation(self) -> int:
        return self._loop_count

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                self._cap = open_stream(self.url, self.buffer_size)
                with self._lock:
                    self._opened = self._cap.isOpened()
                if not self._cap.isOpened():
                    logging.warning("Cannot open camera. Reconnecting in %.1fs...", self.reconnect_delay)
                    time.sleep(self.reconnect_delay)
                    continue
                logging.info("Camera connected.")
                if self._is_file_source:
                    self._file_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    if not 1.0 <= self._file_fps <= 120.0:
                        self._file_fps = 25.0
                    self._next_file_frame_at = time.perf_counter()

            ok, frame = self._cap.read()
            if not ok or frame is None:
                # File captures can be rewound directly. RTSP replay endpoints are
                # normally not seekable, so reconnecting asks the server to start
                # the replay again.
                if self._is_file_source:
                    self._completed_passes += 1
                    logging.info(
                        "Completed full video pass %s%s.",
                        self._completed_passes,
                        f"/{self.video_loop_count}" if self.video_loop_count else "",
                    )
                if self.video_loop_count and self._completed_passes >= self.video_loop_count:
                    logging.info("Completed requested %s full video passes.", self.video_loop_count)
                    self._finished.set()
                    break
                if self.loop_video and self._is_file_source and self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    self._loop_count += 1
                    logging.info("Video reached the end. Restarting loop %s.", self._loop_count)
                    self._next_file_frame_at = time.perf_counter()
                    continue

                action = "Restarting video replay" if self.loop_video else "Reconnecting"
                logging.warning("%s in %.1fs...", action, self.reconnect_delay)
                self._cap.release()
                self._cap = None
                with self._lock:
                    self._opened = False
                time.sleep(self.reconnect_delay)
                continue

            if self._is_file_source:
                now = time.perf_counter()
                if self._next_file_frame_at > now:
                    self._stop.wait(self._next_file_frame_at - now)
                self._next_file_frame_at = max(time.perf_counter(), self._next_file_frame_at) + (1.0 / self._file_fps)

            with self._lock:
                self._frame = frame
                self._seq += 1
                published_seq = self._seq

            # A local video must not behave like a low-latency camera buffer:
            # wait until the inference loop has taken this frame before decoding
            # the next one, otherwise slow inference silently skips most frames.
            if self._is_file_source:
                while not self._stop.is_set():
                    with self._lock:
                        if self._delivered_seq >= published_seq:
                            break
                    self._stop.wait(0.002)


def screen_size() -> tuple[int, int]:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width, height = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return int(width), int(height)
    except Exception:
        return 1280, 720


def resize_for_preview(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(max_width / max(1, width), max_height / max(1, height), 1.0)
    if scale >= 0.999:
        return frame
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def resize_for_processing(frame: np.ndarray, max_width: int | None):
    if not max_width:
        return frame, 1.0, 1.0
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0, 1.0
    scale = max_width / max(1, width)
    resized = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    return resized, width / resized.shape[1], height / resized.shape[0]


def scale_detection(detection: DetectionResult, sx: float, sy: float) -> DetectionResult:
    if sx == 1.0 and sy == 1.0:
        return detection
    x1, y1, x2, y2 = detection.bbox
    keypoints = {
        name: Keypoint(kp.x * sx, kp.y * sy, kp.confidence)
        for name, kp in detection.keypoints.items()
    }
    return DetectionResult(
        bbox=(x1 * sx, y1 * sy, x2 * sx, y2 * sy),
        confidence=detection.confidence,
        keypoints=keypoints,
        track_id=detection.track_id,
    )


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def bbox_center_distance_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx = (ax1 + ax2) * 0.5
    acy = (ay1 + ay2) * 0.5
    bcx = (bx1 + bx2) * 0.5
    bcy = (by1 + by2) * 0.5
    distance = math.hypot(acx - bcx, acy - bcy)
    scale = max(
        1.0,
        math.hypot(max(0.0, ax2 - ax1), max(0.0, ay2 - ay1)),
        math.hypot(max(0.0, bx2 - bx1), max(0.0, by2 - by1)),
    )
    return distance / scale


def _scale_point(point: Point | None, sx: float, sy: float) -> Point | None:
    if point is None:
        return None
    return point[0] * sx, point[1] * sy


def _scale_bbox(bbox: tuple[float, float, float, float] | None, sx: float, sy: float) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return x1 * sx, y1 * sy, x2 * sx, y2 * sy


def scale_geometry(geometry: BodyGeometry, sx: float, sy: float) -> BodyGeometry:
    if sx == 1.0 and sy == 1.0:
        return geometry
    return BodyGeometry(
        valid=geometry.valid,
        reason=geometry.reason,
        shoulder_center=_scale_point(geometry.shoulder_center, sx, sy),
        hip_center=_scale_point(geometry.hip_center, sx, sy),
        knee_center=_scale_point(geometry.knee_center, sx, sy),
        ankle_center=_scale_point(geometry.ankle_center, sx, sy),
        head_point=_scale_point(geometry.head_point, sx, sy),
        body_center=_scale_point(geometry.body_center, sx, sy),
        body_axis_start=_scale_point(geometry.body_axis_start, sx, sy),
        body_axis_end=_scale_point(geometry.body_axis_end, sx, sy),
        body_angle=geometry.body_angle,
        body_core_bbox=_scale_bbox(geometry.body_core_bbox, sx, sy),
        body_core_width=geometry.body_core_width * sx if geometry.body_core_width is not None else None,
        body_core_height=geometry.body_core_height * sy if geometry.body_core_height is not None else None,
        aspect_ratio=geometry.aspect_ratio,
        shoulder_width=geometry.shoulder_width * sx if geometry.shoulder_width is not None else None,
        hip_width=geometry.hip_width * sx if geometry.hip_width is not None else None,
        torso_length=geometry.torso_length * ((sx + sy) * 0.5) if geometry.torso_length is not None else None,
        head_to_hip_distance=geometry.head_to_hip_distance * ((sx + sy) * 0.5) if geometry.head_to_hip_distance is not None else None,
        hip_to_ankle_distance=geometry.hip_to_ankle_distance * ((sx + sy) * 0.5) if geometry.hip_to_ankle_distance is not None else None,
        body_scale=geometry.body_scale * ((sx + sy) * 0.5) if geometry.body_scale is not None else None,
        pose_confidence=geometry.pose_confidence,
        used_keypoints=list(geometry.used_keypoints),
    )


def log_acceleration_status() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            logging.info("Torch CUDA device: %s", torch.cuda.get_device_name(0))
        else:
            logging.warning("Torch CUDA is not available.")
    except Exception as exc:
        logging.warning("Cannot inspect Torch CUDA: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-file realtime fall detection with YOLO11x pose.")
    parser.add_argument("--url", help="Camera URL. When omitted, load it from camera_configs/<cam-id>.json, then RTSP_URL, then the default camera.")
    parser.add_argument("--video", help="Local video file. When provided, this takes precedence over --url.")
    parser.add_argument("--model", default=CONFIG["pose"]["pose_model"], help="YOLO pose model path, e.g. yolo11x-pose.pt or a TensorRT .engine file.")
    parser.add_argument("--device", default=CONFIG["pose"]["device"], help="Inference device, e.g. cuda, cuda:0, 0, or cpu.")
    parser.add_argument("--imgsz", type=int, default=CONFIG["pose"]["det_input_size"], help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=CONFIG["pose"]["detector_score_thr"], help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=CONFIG["pose"]["detector_nms_thr"], help="YOLO NMS IoU threshold.")
    parser.add_argument("--tracker", default=CONFIG["tracking"]["tracker"], help="Ultralytics tracker config, e.g. bytetrack.yaml.")
    parser.add_argument("--stabilize-track-ids", action=argparse.BooleanOptionalAction, default=True, help="Keep logical object IDs stable when the tracker briefly switches raw IDs.")
    parser.add_argument("--track-id-iou-threshold", type=float, default=0.25, help="Minimum bbox IoU for reusing a previous logical track ID.")
    parser.add_argument("--track-id-center-threshold", type=float, default=0.35, help="Maximum bbox center movement ratio for reusing a previous logical track ID when IoU is low.")
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=True, help="Show realtime preview window.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="Draw extra body geometry debug overlays.")
    parser.add_argument("--buffer-size", type=int, default=1, help="OpenCV capture buffer size.")
    parser.add_argument("--process-width", type=int, default=960, help="Resize camera frames to this width before inference. Use 0 for original size.")
    parser.add_argument("--preview-width", type=int, help="Max preview window width. Defaults to 92 percent of screen width.")
    parser.add_argument("--preview-height", type=int, help="Max preview window height. Defaults to 86 percent of screen height.")
    parser.add_argument("--log-interval", type=float, default=5.0, help="Seconds between realtime speed logs.")
    parser.add_argument("--reconnect-delay", type=float, default=2.0, help="Seconds to wait before reconnecting.")
    parser.add_argument("--loop-video", action=argparse.BooleanOptionalAction, default=True, help="Restart a finished video source; for RTSP replay sources, reconnect to begin the replay again.")
    parser.add_argument("--video-loop-count", type=int, default=0, help="Stop after this many complete video passes. Use 0 to loop forever.")
    parser.add_argument("--publish", action=argparse.BooleanOptionalAction, default=False, help="Publish AI metadata messages to Kafka.")
    parser.add_argument("--publish-dry-run", action=argparse.BooleanOptionalAction, default=False, help="Build and log messages without sending to Kafka.")
    parser.add_argument("--publish-log-json", action=argparse.BooleanOptionalAction, default=False, help="Log every metadata JSON payload.")
    parser.add_argument("--publish-log-delivery", action=argparse.BooleanOptionalAction, default=False, help="Log every successful Kafka delivery callback.")
    parser.add_argument("--publish-debug-objects", action=argparse.BooleanOptionalAction, default=False, help="Log compact published object IDs, events, and held-frame counts.")
    parser.add_argument("--publish-display-fields", action=argparse.BooleanOptionalAction, default=False, help="Add label/color display fields at top level of each AI result.")
    parser.add_argument("--publish-sample-strict", action=argparse.BooleanOptionalAction, default=False, help="Publish ai_results with exactly the sample object keys, without extra display fields.")
    parser.add_argument("--publish-name-from-state", action=argparse.BooleanOptionalAction, default=False, help="Set the sample name field to normal, fall_candidate, or fall.")
    parser.add_argument("--publish-meta-type-from-state", action=argparse.BooleanOptionalAction, default=False, help="Set meta_type to normal, fall_candidate, or fall so UIs that label from meta_type show the state.")
    parser.add_argument("--publish-stable-result-id", action=argparse.BooleanOptionalAction, default=True, help="Use a stable ai_results id per tracked object instead of a new UUID every frame.")
    parser.add_argument("--publish-result-id-mode", choices=("global", "uuid", "track"), default="global", help="Stable ai_results id format when --publish-stable-result-id is enabled.")
    parser.add_argument("--publish-object-id-offset", type=int, default=0, help="Add this offset to published tracking_object_id/object_key to avoid collisions with other publishers.")
    parser.add_argument("--publish-fast-async", action=argparse.BooleanOptionalAction, default=False, help="Use one persistent async Kafka producer instead of the sample publish_message helper.")
    parser.add_argument("--publish-every-n", type=int, default=1, help="Publish one message every N processed frames.")
    parser.add_argument("--publish-repeat-rate", type=float, default=0.0, help="Publish the latest model result at this fixed FPS, independent of inference speed. Use 0 to publish only after inference.")
    parser.add_argument("--publish-object-update-frames", type=int, default=120, help="Send object_update for the first N processed frames so late UI subscribers can create boxes.")
    parser.add_argument("--publish-always-update", action=argparse.BooleanOptionalAction, default=False, help="Send object_update on every frame to keep UI object boxes alive.")
    parser.add_argument("--publish-missing-hold-frames", type=int, default=20, help="Keep publishing the last bbox for this many missed detection frames to avoid UI flicker.")
    parser.add_argument("--suppress-empty-publish-frames", type=int, default=0, help="Do not publish empty ai_results for this many consecutive objectless frames.")
    parser.add_argument("--fall-hold-frames", type=int, default=300, help="Keep publishing a FALL alert for this many frames after the model first detects a fall.")
    parser.add_argument("--force-fall-alert", action=argparse.BooleanOptionalAction, default=False, help="Publish every detected person as a falling blacklist alert for UI integration testing.")
    parser.add_argument("--fall-probe-log-threshold", type=float, default=1.01, help="Log non-fall probe metrics only when fall_score is at or above this value. Use 0.25 for verbose debugging.")
    parser.add_argument("--publish-sample-resource", action=argparse.BooleanOptionalAction, default=True, help="Save the first camera frame using the sample storage layout and attach image_path/asset_id.")
    parser.add_argument("--sample-storage-base-path", default=os.getenv("SAMPLE_STORAGE_BASE_PATH", DEFAULT_STORAGE_BASE_PATH), help="Base path for sample-style camera frame images.")
    parser.add_argument("--sample-storage-device-id", default=os.getenv("SAMPLE_STORAGE_DEVICE_ID"), help="Device id folder used for sample-style image storage. Defaults to --cam-id.")
    parser.add_argument("--storage-is-external", action=argparse.BooleanOptionalAction, default=os.getenv("STORAGE_MANAGE_IS_EXTERNAL", "false").lower() in {"1", "true", "yes", "on"}, help="Pass the sample storage external flag to the upload helper.")
    parser.add_argument("--kafka-bootstrap-servers", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", DEFAULT_BOOTSTRAP_SERVERS), help="Kafka bootstrap servers.")
    parser.add_argument("--kafka-topic", default=os.getenv("KAFKA_TOPIC", DEFAULT_TOPIC), help="Kafka topic for AI metadata.")
    parser.add_argument("--server-id", default=os.getenv("SERVER_ID", "NODE1"), help="server_id field in metadata messages.")
    parser.add_argument("--cam-id", default=os.getenv("CAM_ID", "CAM_VP_1"), help="cam_id field in metadata messages.")
    parser.add_argument("--publish-meta-type", default=os.getenv("PUBLISH_META_TYPE", DEFAULT_META_TYPE), help="meta_type for each bbox. Defaults to the working sample value.")
    parser.add_argument("--service-instance-id", default=os.getenv("SERVICE_INSTANCE_ID"), help="Optional stable service instance id.")
    args = parser.parse_args()
    if args.video:
        args.url = args.video
    elif not args.url:
        config_name = f"{Path(str(args.cam_id)).name}.json"
        # runtime.py lives in fall_detector/, while project camera configs stay
        # beside the backward-compatible entry script at the project root.
        camera_config_path = Path(__file__).resolve().parent.parent / "camera_configs" / config_name
        if camera_config_path.is_file():
            try:
                camera_config = json.loads(camera_config_path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError) as exc:
                parser.error(f"Cannot read camera config {camera_config_path}: {exc}")
            args.url = str(camera_config.get("url") or "").strip()
            if not args.url:
                parser.error(f"Camera config has no URL: {camera_config_path}")
        else:
            args.url = os.getenv("RTSP_URL", DEFAULT_CAMERA_URL)
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logging.info("Opening camera: %s", mask_url(args.url))
    logging.info("Model: %s", args.model)
    logging.info("Device: %s", args.device)
    log_acceleration_status()

    if args.display:
        screen_w, screen_h = screen_size()
        preview_w = args.preview_width or int(screen_w * 0.92)
        preview_h = args.preview_height or int(screen_h * 0.86)
        cv2.namedWindow("Realtime Fall Detection", cv2.WINDOW_NORMAL)
    else:
        preview_w = preview_h = 0

    estimator = UltralyticsPoseEstimator(
        model_path=args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        tracker=args.tracker,
    )
    tracker = Tracker(estimator, enabled=bool(CONFIG["tracking"].get("enabled", True)))
    classifier = RuleBasedFallClassifier(CONFIG)
    state_machine = FallStateMachine(CONFIG)
    visualizer = Visualizer(float(CONFIG["pose"]["keypoint_confidence"]), debug=args.debug)
    service_instance_id = args.service_instance_id or str(uuid.uuid4())
    publisher = None
    if args.publish or args.publish_dry_run:
        publisher = KafkaFramePublisher(
            bootstrap_servers=args.kafka_bootstrap_servers,
            topic=args.kafka_topic,
            dry_run=args.publish_dry_run,
            log_json=args.publish_log_json,
            log_delivery=args.publish_log_delivery,
            sample_sync=not args.publish_fast_async,
        )
        mode = "dry-run" if args.publish_dry_run else "kafka"
        logging.info(
            "Metadata publishing enabled: mode=%s server_id=%s cam_id=%s topic=%s",
            mode,
            args.server_id,
            args.cam_id,
            args.kafka_topic,
        )

    cap = LatestFrameCapture(args.url, args.buffer_size, args.reconnect_delay, args.loop_video, args.video_loop_count)
    cap.start()
    persons: dict[int, PersonState] = {}
    raw_to_logical_track_id: dict[int, int] = {}
    logical_last_bbox: dict[int, tuple[float, float, float, float]] = {}
    next_logical_track_id = 1
    max_missing = max(
        int(CONFIG["tracking"].get("max_missing_frames", 60)),
        max(0, int(args.publish_missing_hold_frames)),
    )
    frame_index = 0
    timestamp = 0.0
    last_frame_time = time.perf_counter()
    stats_start = time.perf_counter()
    stats_processed = 0
    stats_infer_seconds = 0.0
    stats_people = 0
    last_capture_seq = 0
    last_capture_loop_generation = cap.loop_generation()
    consecutive_empty_publish_frames = 0
    publish_image_path = ""
    publish_asset_id: str | None = None
    sample_resource_ready = False
    publish_repeat_rate = max(0.0, float(args.publish_repeat_rate))
    latest_publish_lock = threading.Lock()
    latest_publish_items: list[tuple[DetectionResult, BodyGeometry, FeatureVector]] = []
    latest_publish_size: tuple[int, int] | None = None
    latest_publish_image_path = ""
    latest_publish_asset_id: str | None = None
    active_fall_publish_ids: set[int] = set()
    repeat_publish_stop = threading.Event()
    repeat_publish_thread: threading.Thread | None = None
    repeat_publish_frame_index = 0

    def publish_items_have_fall(items: list[tuple[DetectionResult, BodyGeometry, FeatureVector]]) -> bool:
        return any(
            str(getattr(item[2], "state_event", "") or "") == "falling"
            or str(getattr(item[2], "state", "") or "") == "FALL"
            for item in items
        )

    def publish_fall_object_ids(
        items: list[tuple[DetectionResult, BodyGeometry, FeatureVector]],
    ) -> set[int]:
        return {
            int(getattr(item[2], "track_id", 0) or 0)
            for item in items
            if (
                str(getattr(item[2], "state_event", "") or "") == "falling"
                or str(getattr(item[2], "state", "") or "") == "FALL"
            )
        }

    def prepare_publish_resource(frame_to_save: np.ndarray, reason: str, source_frame_index: int) -> tuple[str, str | None]:
        storage_device_id = args.sample_storage_device_id or args.cam_id
        saved_frame = materialize_camera_frame(frame_to_save, storage_device_id, args.sample_storage_base_path)
        image_path = str(saved_frame)
        asset_id = managed_upload_flow(
            image_path=image_path,
            device_id=storage_device_id,
            keep_file=True,
            is_external=args.storage_is_external,
        )
        if asset_id and str(asset_id).startswith("local:"):
            logging.warning("Ignoring non-Storage asset_id for Kafka: %s", asset_id)
            asset_id = None
        logging.info(
            "%s resource ready: frame=%s image_path=%s asset_id=%s",
            reason,
            source_frame_index,
            image_path,
            asset_id,
        )
        return image_path, asset_id

    def repeat_publish_loop() -> None:
        nonlocal repeat_publish_frame_index
        assert publisher is not None
        interval = 1.0 / max(1e-3, publish_repeat_rate)
        next_publish = time.perf_counter()
        while not repeat_publish_stop.is_set():
            now = time.perf_counter()
            if now < next_publish:
                repeat_publish_stop.wait(next_publish - now)
                continue
            next_publish += interval
            with latest_publish_lock:
                snapshot_items = list(latest_publish_items)
                snapshot_size = latest_publish_size
                snapshot_image_path = latest_publish_image_path
                snapshot_asset_id = latest_publish_asset_id
            if snapshot_size is None:
                continue
            image_width, image_height = snapshot_size
            message = build_realtime_message(
                snapshot_items,
                server_id=args.server_id,
                cam_id=args.cam_id,
                frame_num=repeat_publish_frame_index,
                image_width=image_width,
                image_height=image_height,
                service_instance_id=service_instance_id,
                image_path=snapshot_image_path,
                asset_id=snapshot_asset_id,
                meta_type=args.publish_meta_type,
                include_display_fields=args.publish_display_fields,
                sample_strict=args.publish_sample_strict,
                name_from_state=args.publish_name_from_state,
                meta_type_from_state=args.publish_meta_type_from_state,
                stable_result_id=args.publish_stable_result_id,
                object_id_offset=args.publish_object_id_offset,
                result_id_mode=args.publish_result_id_mode,
            )
            try:
                publisher.publish(message)
                if args.publish_debug_objects:
                    logging.info(
                        "Repeat publish debug: frame=%s objects=%s ids=%s",
                        repeat_publish_frame_index,
                        len(snapshot_items),
                        [int(getattr(item[2], "track_id", 0) or 0) for item in snapshot_items],
                    )
            except BufferError as exc:
                logging.warning("Kafka local queue is full, dropping repeated metadata: %s", exc)
            except Exception:
                logging.exception("Cannot publish repeated AI metadata message")
            repeat_publish_frame_index += 1

    if publisher is not None and publish_repeat_rate > 0:
        repeat_publish_thread = threading.Thread(target=repeat_publish_loop, name="metadata-repeat-publisher", daemon=True)
        repeat_publish_thread.start()
        logging.info("Metadata repeat publishing enabled: rate=%.2f fps", publish_repeat_rate)

    try:
        while True:
            ok, frame, last_capture_seq = cap.read_latest(last_capture_seq)
            if not ok:
                if cap.is_finished():
                    logging.info("Video playback finished; stopping realtime processing.")
                    break
                logging.warning("No fresh camera frame. Waiting...")
                continue

            current_loop_generation = cap.loop_generation()
            if current_loop_generation != last_capture_loop_generation:
                tracker.reset()
                persons.clear()
                raw_to_logical_track_id.clear()
                logical_last_bbox.clear()
                active_fall_publish_ids.clear()
                next_logical_track_id = 1
                last_capture_loop_generation = current_loop_generation
                logging.info("Video loop boundary: reset tracker and fall state (loop=%s).", current_loop_generation)

            now = time.perf_counter()
            source_frame_interval = cap.source_frame_interval()
            dt = source_frame_interval if source_frame_interval is not None else max(1e-3, now - last_frame_time)
            last_frame_time = now
            timestamp += dt

            inference_frame, sx, sy = resize_for_processing(frame, args.process_width)
            infer_start = time.perf_counter()
            detections = [scale_detection(detection, sx, sy) for detection in tracker.update(inference_frame)]
            stats_infer_seconds += time.perf_counter() - infer_start
            stats_people += len(detections)

            seen_ids: set[int] = set()
            display_items: list[tuple[DetectionResult, BodyGeometry, FeatureVector]] = []
            publish_items: list[tuple[DetectionResult, BodyGeometry, FeatureVector]] = []
            held_publish_ids: list[int] = []
            id_debug_pairs: list[tuple[int, int]] = []
            for detection_index, detection in enumerate(detections, start=1):
                raw_track_id = int(detection.track_id or detection_index)
                track_id = raw_track_id
                if args.stabilize_track_ids:
                    mapped_track_id = raw_to_logical_track_id.get(raw_track_id)
                    if mapped_track_id is not None and mapped_track_id not in seen_ids:
                        track_id = mapped_track_id
                    else:
                        best_track_id = None
                        best_iou = max(0.0, float(args.track_id_iou_threshold))
                        best_center_track_id = None
                        best_center_ratio = max(0.0, float(args.track_id_center_threshold))
                        for candidate_track_id, candidate_bbox in logical_last_bbox.items():
                            if candidate_track_id in seen_ids:
                                continue
                            overlap = bbox_iou(detection.bbox, candidate_bbox)
                            if overlap > best_iou:
                                best_iou = overlap
                                best_track_id = candidate_track_id
                            center_ratio = bbox_center_distance_ratio(detection.bbox, candidate_bbox)
                            if center_ratio < best_center_ratio:
                                best_center_ratio = center_ratio
                                best_center_track_id = candidate_track_id
                        if best_track_id is not None:
                            track_id = best_track_id
                        elif best_center_track_id is not None:
                            track_id = best_center_track_id
                        elif raw_track_id in persons or raw_track_id in seen_ids:
                            while next_logical_track_id in persons or next_logical_track_id in seen_ids:
                                next_logical_track_id += 1
                            track_id = next_logical_track_id
                            next_logical_track_id += 1
                        raw_to_logical_track_id[raw_track_id] = track_id
                    detection.track_id = track_id
                logical_last_bbox[track_id] = detection.bbox
                id_debug_pairs.append((raw_track_id, track_id))
                seen_ids.add(track_id)
                is_new_track = track_id not in persons
                person = persons.setdefault(track_id, PersonState(track_id, int(CONFIG["temporal"]["window_size"])))
                person.missing_frames = 0
                geometry = compute_body_geometry(detection, float(CONFIG["pose"]["keypoint_confidence"]))
                feature = compute_motion_features(frame_index, timestamp, track_id, geometry, person.previous, dt)
                person.add(feature)
                min_history = int(CONFIG["temporal"]["min_history"])
                fall_score = classifier.predict(person) if len(person.features) >= min_history else 0.0
                previous_state = person.current_state
                current_state = state_machine.update(person, feature, fall_score)
                if args.force_fall_alert:
                    feature.state = "FALL"
                    feature.fall_score = max(feature.fall_score, fall_score, 1.0)
                    current_state = "FALL"
                if current_state == "NORMAL" and previous_state != "NORMAL":
                    person.fall_hold_remaining = 0
                    setattr(feature, "state_event", "normal")
                elif current_state == "FALL":
                    person.fall_hold_remaining = max(person.fall_hold_remaining, int(args.fall_hold_frames))
                elif person.fall_hold_remaining > 0:
                    person.fall_hold_remaining -= 1
                    feature.state = "FALL"
                    feature.fall_score = max(feature.fall_score, fall_score)
                    current_state = "FALL"
                if current_state == "FALL":
                    setattr(feature, "state_event", "falling")
                setattr(feature, "fall_started", current_state == "FALL" and previous_state != "FALL")
                # The dashboard creates and moves a box only from object_update.
                # Respect the CLI controls for ordinary people as well as FALL;
                # previously these options were parsed but never applied here.
                publish_as_update = (
                    bool(args.publish_always_update)
                    or frame_index < max(0, int(args.publish_object_update_frames))
                    or current_state == "FALL"
                )
                setattr(feature, "event_type", "object_update" if publish_as_update else "object_exist")
                person.last_publish_item = (detection, geometry, feature)
                if args.display:
                    display_items.append((detection, geometry, feature))
                if publisher is not None:
                    publish_items.append((detection, geometry, feature))
                if feature.state == "FALL":
                    logging.warning("FALL detected: track=%s score=%.2f frame=%s", track_id, feature.fall_score, frame_index)
                elif feature.state != "NORMAL" or feature.fall_score >= float(args.fall_probe_log_threshold):
                    logging.info(
                        "Fall probe: track=%s state=%s score=%.2f angle=%s aspect=%s valid=%s reason=%s frame=%s",
                        track_id,
                        feature.state,
                        feature.fall_score,
                        Visualizer._fmt(feature.body_angle, 1),
                        Visualizer._fmt(feature.aspect_ratio, 2),
                        feature.valid,
                        geometry.reason,
                        frame_index,
                    )

            for track_id in list(persons):
                if track_id not in seen_ids:
                    person = persons[track_id]
                    person.missing_frames += 1
                    if person.fall_hold_remaining > 0:
                        person.fall_hold_remaining -= 1
                    if person.missing_frames > max_missing:
                        del persons[track_id]
                        logical_last_bbox.pop(track_id, None)
                        raw_to_logical_track_id = {
                            raw_id: logical_id
                            for raw_id, logical_id in raw_to_logical_track_id.items()
                            if logical_id != track_id
                        }
                        continue
                    if person.last_publish_item and person.missing_frames <= max(0, int(args.publish_missing_hold_frames)):
                        detection, geometry, feature = person.last_publish_item
                        if person.fall_hold_remaining > 0 or feature.state == "FALL":
                            feature.state = "FALL"
                            setattr(feature, "state_event", "falling")
                            setattr(feature, "fall_started", False)
                            setattr(feature, "event_type", "object_update")
                        elif bool(args.publish_always_update) or frame_index < max(0, int(args.publish_object_update_frames)):
                            setattr(feature, "state_event", "")
                            setattr(feature, "fall_started", False)
                            setattr(feature, "event_type", "object_update")
                        else:
                            setattr(feature, "state_event", "")
                            setattr(feature, "fall_started", False)
                            setattr(feature, "event_type", "object_exist")
                        if args.display:
                            display_items.append((detection, geometry, feature))
                        if publisher is not None:
                            publish_items.append((detection, geometry, feature))
                            held_publish_ids.append(track_id)

            publish_every_n = max(1, int(args.publish_every_n))
            if publisher is not None and frame_index % publish_every_n == 0:
                if not publish_items:
                    active_fall_publish_ids = set()
                    consecutive_empty_publish_frames += 1
                    if consecutive_empty_publish_frames <= max(0, int(args.suppress_empty_publish_frames)):
                        if args.publish_debug_objects:
                            logging.info(
                                "Publish debug: frame=%s suppressing empty publish count=%s active=%s",
                                frame_index,
                                consecutive_empty_publish_frames,
                                sorted(persons.keys()),
                            )
                        if args.display:
                            preview = resize_for_preview(frame, preview_w, preview_h)
                            cv2.imshow("Realtime Fall Detection", preview)
                            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                                break
                        frame_index += 1
                        stats_processed += 1
                        continue
                else:
                    consecutive_empty_publish_frames = 0
                image_height, image_width = frame.shape[:2]
                fall_object_ids = publish_fall_object_ids(publish_items)
                new_fall_object_ids = fall_object_ids - active_fall_publish_ids
                active_fall_publish_ids = fall_object_ids
                if args.publish_sample_resource and not sample_resource_ready:
                    try:
                        publish_image_path, publish_asset_id = prepare_publish_resource(frame, "Sample", frame_index)
                        sample_resource_ready = True
                    except Exception:
                        sample_resource_ready = True
                        logging.exception("Cannot prepare sample-style image resource; publishing metadata without asset_id")
                message_image_path = publish_image_path
                message_asset_id = publish_asset_id if frame_index == 0 else None
                if new_fall_object_ids and args.publish_sample_resource:
                    try:
                        message_image_path, message_asset_id = prepare_publish_resource(frame, "Fall", frame_index)
                    except Exception:
                        logging.exception("Cannot prepare fall frame resource; publishing fall metadata without new asset_id")
                if publish_repeat_rate > 0:
                    with latest_publish_lock:
                        latest_publish_items = list(publish_items)
                        latest_publish_size = (image_width, image_height)
                        latest_publish_image_path = message_image_path
                        latest_publish_asset_id = message_asset_id
                    if args.publish_debug_objects:
                        logging.info(
                            "Publish debug: frame=%s updated repeat snapshot detected=%s objects=%s held=%s raw_to_logical=%s",
                            frame_index,
                            len(detections),
                            [int(getattr(item[2], "track_id", 0) or 0) for item in publish_items],
                            held_publish_ids,
                            id_debug_pairs,
                        )
                else:
                    message = build_realtime_message(
                        publish_items,
                        server_id=args.server_id,
                        cam_id=args.cam_id,
                        frame_num=frame_index,
                        image_width=image_width,
                        image_height=image_height,
                        service_instance_id=service_instance_id,
                        image_path=message_image_path,
                        asset_id=message_asset_id,
                        meta_type=args.publish_meta_type,
                        include_display_fields=args.publish_display_fields,
                        sample_strict=args.publish_sample_strict,
                        name_from_state=args.publish_name_from_state,
                        meta_type_from_state=args.publish_meta_type_from_state,
                        stable_result_id=args.publish_stable_result_id,
                        object_id_offset=args.publish_object_id_offset,
                        result_id_mode=args.publish_result_id_mode,
                    )
                    if args.publish_debug_objects:
                        object_ids = [
                            (
                                item.get("tracking_object_id"),
                                item.get("object_key"),
                                item.get("event_type"),
                                item.get("id"),
                            )
                            for item in message.get("ai_results", [])
                        ]
                        if not object_ids:
                            logging.warning(
                                "Publish debug: frame=%s EMPTY raw_to_logical=%s active=%s held=%s",
                                frame_index,
                                id_debug_pairs,
                                sorted(persons.keys()),
                                held_publish_ids,
                            )
                        else:
                            logging.info(
                                "Publish debug: frame=%s detected=%s published=%s held=%s raw_to_logical=%s",
                                frame_index,
                                len(detections),
                                object_ids,
                                held_publish_ids,
                                id_debug_pairs,
                            )
                    try:
                        publisher.publish(message)
                        if publisher.should_log_success():
                            action = "Queued" if args.publish_fast_async else "Published"
                            logging.info("%s metadata: cam_id=%s frame=%s objects=%s", action, args.cam_id, frame_index, len(publish_items))
                    except BufferError as exc:
                        logging.warning("Kafka local queue is full, dropping frame metadata: %s", exc)
                    except Exception:
                        logging.exception("Cannot publish AI metadata message")

            if args.display:
                preview = resize_for_preview(frame, preview_w, preview_h)
                draw_sx = preview.shape[1] / max(1, frame.shape[1])
                draw_sy = preview.shape[0] / max(1, frame.shape[0])
                for detection, geometry, feature in display_items:
                    visualizer.draw(
                        preview,
                        scale_detection(detection, draw_sx, draw_sy),
                        scale_geometry(geometry, draw_sx, draw_sy),
                        feature,
                    )
                cv2.imshow("Realtime Fall Detection", preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            frame_index += 1
            stats_processed += 1
            stats_elapsed = time.perf_counter() - stats_start
            if stats_elapsed >= args.log_interval:
                proc_fps = stats_processed / stats_elapsed
                infer_ms = (stats_infer_seconds / max(1, stats_processed)) * 1000.0
                avg_people = stats_people / max(1, stats_processed)
                logging.info(
                    "Realtime speed: processed_fps=%.2f infer_ms=%.1f avg_people=%.2f frame=%s",
                    proc_fps,
                    infer_ms,
                    avg_people,
                    frame_index,
                )
                stats_start = time.perf_counter()
                stats_processed = 0
                stats_infer_seconds = 0.0
                stats_people = 0
    finally:
        repeat_publish_stop.set()
        if repeat_publish_thread is not None:
            repeat_publish_thread.join(timeout=2.0)
        if publisher is not None:
            publisher.close()
        cap.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
