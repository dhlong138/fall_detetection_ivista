"""Load the default fall-detection settings from a JSON file."""

import json
from pathlib import Path
from typing import Any


DEFAULT_CAMERA_URL = "rtsp://admin:admin1234!@192.168.1.8:554/0"
DEFAULT_STORAGE_BASE_PATH = "server_assets/V3SStorage"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "fall_configs" / "default.json"


def load_default_config() -> dict[str, Any]:
    try:
        config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load default settings {DEFAULT_CONFIG_PATH}: {exc}") from exc
    if not isinstance(config, dict):
        raise RuntimeError(f"Default settings must be a JSON object: {DEFAULT_CONFIG_PATH}")
    return config


CONFIG = load_default_config()
