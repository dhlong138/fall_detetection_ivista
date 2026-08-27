"""Default settings for the realtime fall-detection service."""

from typing import Any


DEFAULT_CAMERA_URL = "rtsp://admin:admin1234!@192.168.1.8:554/0"
DEFAULT_STORAGE_BASE_PATH = "server_assets/V3SStorage"

CONFIG: dict[str, Any] = {
    "pose": {"device": "cuda", "pose_model": "yolo11x-pose.pt", "det_input_size": 640,
             "detector_score_thr": 0.10, "detector_nms_thr": 0.45, "keypoint_confidence": 0.35},
    "tracking": {"enabled": True, "tracker": "bytetrack.yaml", "max_missing_frames": 30},
    "temporal": {"window_size": 30, "min_history": 10},
    "fall_detection": {
        "angle_threshold": 60.0, "angle_change_threshold": 35.0, "aspect_ratio_threshold": 1.15,
        "aspect_ratio_drop_threshold": 0.35, "velocity_threshold": 1.2, "acceleration_threshold": 2.5,
        "candidate_threshold": 0.55, "fall_threshold": 0.70, "recovery_threshold": 0.35,
        "persistence_frames": 5, "cooldown_frames": 20, "upright_angle_threshold": 35.0,
        "upright_aspect_ratio_threshold": 1.45, "require_observed_upright": True,
        "weights": {"angle": 0.20, "angle_change": 0.15, "aspect_ratio": 0.30,
                    "aspect_ratio_change": 0.20, "velocity": 0.10, "acceleration": 0.05},
    },
    "features": {"use_angle": True, "use_aspect_ratio": True, "use_velocity": True, "use_acceleration": True},
}
