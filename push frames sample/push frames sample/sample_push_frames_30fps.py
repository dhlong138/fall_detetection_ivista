#!/usr/bin/env python3
"""Sample producer: publish generated frame metadata at 30 FPS."""

import time
import uuid
import random
import os
from datetime import datetime
from pathlib import Path

try:
    from .construct_ai_message import construct_message, publish_message
    from .upload_image_to_minio import managed_upload_flow
except ImportError:
    from construct_ai_message import construct_message, publish_message
    from upload_image_to_minio import managed_upload_flow


# Edit these values for a test stream.
SERVER_ID = "NODE1"
CAM_ID = "CAM_VP_1"
FPS = 30.0
NUMBER_OF_FRAMES = 1  # Set to None to run forever.
SERVICE_INSTANCE_ID = str(uuid.uuid4())
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
IMAGE_PATH = os.environ.get("SAMPLE_FRAME_IMAGE_PATH", "/home/aiserver/Desktop/V3SServer/server/tests/test_image.png")
STORAGE_BASE_PATH = os.environ.get("SAMPLE_STORAGE_BASE_PATH", "server_assets/V3SStorage")
STORAGE_DEVICE_ID = os.environ.get("SAMPLE_STORAGE_DEVICE_ID", CAM_ID)
STORAGE_IS_EXTERNAL = os.environ.get("STORAGE_MANAGE_IS_EXTERNAL", "false").lower() in {
    "1", "true", "yes", "on"
}
TRACKING_IDS = [random.SystemRandom().randint(1, 2**31 - 1) for _ in range(3)]


def generate_image_path(device_id=CAM_ID, base_path=STORAGE_BASE_PATH, now=None):
    """Return a DeepStream-compatible dated JPEG path for a device image."""
    now = datetime.now() if now is None else now
    date_path = now.strftime("%Y/%m/%d")
    filename = f"{device_id}_{now.strftime('%Y_%m_%d_%H_%M_%S_%f')}.jpg"
    return Path(base_path) / "images" / date_path / str(device_id) / filename


def materialize_sample_image(source_path=IMAGE_PATH, device_id=CAM_ID,
                             base_path=STORAGE_BASE_PATH, now=None):
    """Convert the configured source image to a DeepStream-named JPEG."""
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Image file does not exist: {source}")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to materialize the sample JPEG") from exc

    destination = generate_image_path(device_id, base_path, now)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination, format="JPEG")
    return destination


def build_ai_results(frame_num):
    """Create sample tracked objects for one frame."""
    results = []
    for index in range(1):
        tracking_id = TRACKING_IDS[index]
        object_key = tracking_id
        object_event = "object_update" if frame_num == 0 else "object_exist"
        results.append(
            {
                "id": str(uuid.uuid4()),
                "meta_type": "person",
                "tracking_object_id": tracking_id,
                "bbox": {
                    "left": 200 + index * 450 + (frame_num % 30),
                    "top": 100,
                    "width": 300,
                    "height": 220,
                },
                "is_blacklist": True,
                "detected_object_ids": "person falling",
                "name": "",
                "event_type": object_event,
                "confidence": 1.0,
                "object_key": object_key,
                "global_object_id": (
                    f"{SERVICE_INSTANCE_ID}:{CAM_ID}:{object_key}"
                ),
                "object_analysis": [
                    {
                        "type": "object_analysis",
                        "object_key": object_key,
                        "last_pos": [350 + index * 450, 320],
                    }
                ],
                "placeholder": {},
            }
        )
    return results


def push_frames():
    """Publish generated metadata at the configured frame rate."""
    image_path = materialize_sample_image()
    asset_id = managed_upload_flow(
        image_path=str(image_path),
        device_id=STORAGE_DEVICE_ID,
        keep_file=True,
        is_external=STORAGE_IS_EXTERNAL,
    )
    frame_interval = 1.0 / FPS
    next_frame_time = time.monotonic()
    frame_num = 0

    while NUMBER_OF_FRAMES is None or frame_num < NUMBER_OF_FRAMES:
        now_ns = time.time_ns()
        message = construct_message(
            server_id=SERVER_ID,
            cam_id=CAM_ID,
            frame_num=frame_num,
            ntp_timestamp=now_ns,
            ai_results=build_ai_results(frame_num),
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            image_path=str(image_path),
            asset_id=asset_id if frame_num == 0 else None,
            service_instance_id=SERVICE_INSTANCE_ID,
        )
        publish_message(message)
        frame_num += 1

        next_frame_time += frame_interval
        delay = next_frame_time - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            # Reset after an overrun so latency does not grow indefinitely.
            next_frame_time = time.monotonic()


if __name__ == "__main__":
    push_frames()
