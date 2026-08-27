#!/usr/bin/env python3
"""Construct and optionally publish a frame-analysis message."""

import json
import uuid
from datetime import datetime, timezone


# Edit these values when using this module as a standalone integration helper.
SERVER_ID = "NODE1"
CAM_ID = "5008D_03"
FRAME_NUM = 0
NTP_TIMESTAMP = 0
AI_RESULTS = []
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
IMAGE_PATH = ""
ASSET_ID = None
FRAME_ANALYSIS = None
SERVICE_INSTANCE_ID = None
MESSAGE_ID = None
KAFKA_BOOTSTRAP_SERVERS = "192.168.1.199:29092"
KAFKA_TOPIC = "ai.metadata.v1"
_UNSET = object()

# Import from another service like this:
# from src.scripts.construct_ai_message import construct_message, publish_configured_message
# message = construct_message()
# publish_configured_message()  # optional Kafka publish


def construct_message(
    server_id=_UNSET, cam_id=_UNSET, frame_num=_UNSET, ntp_timestamp=_UNSET,
    ai_results=_UNSET, image_width=_UNSET, image_height=_UNSET,
    image_path=_UNSET, frame_analysis=_UNSET, service_instance_id=_UNSET,
    message_id=_UNSET, asset_id=_UNSET,
):
    """Return a message dictionary in the project's frame-analysis format."""
    server_id = SERVER_ID if server_id is _UNSET else server_id
    cam_id = CAM_ID if cam_id is _UNSET else cam_id
    frame_num = FRAME_NUM if frame_num is _UNSET else frame_num
    ntp_timestamp = NTP_TIMESTAMP if ntp_timestamp is _UNSET else ntp_timestamp
    ai_results = AI_RESULTS if ai_results is _UNSET else ai_results
    image_width = IMAGE_WIDTH if image_width is _UNSET else image_width
    image_height = IMAGE_HEIGHT if image_height is _UNSET else image_height
    image_path = IMAGE_PATH if image_path is _UNSET else image_path
    asset_id = ASSET_ID if asset_id is _UNSET else asset_id
    frame_analysis = FRAME_ANALYSIS if frame_analysis is _UNSET else frame_analysis
    service_instance_id = SERVICE_INSTANCE_ID if service_instance_id is _UNSET else service_instance_id
    message_id = MESSAGE_ID if message_id is _UNSET else message_id
    results = []
    for result in ai_results or []:
        if not isinstance(result, dict):
            raise ValueError("Every ai_results item must be a JSON object")
        item = dict(result)
        item.pop("service_instance_id", None)
        results.append(item)

    message = {
        "schema_version": 1,
        "message_id": message_id or str(uuid.uuid4()),
        "server_id": str(server_id),
        "cam_id": str(cam_id),
        "frame_num": int(frame_num),
        "ntp_timestamp": int(ntp_timestamp),
        "produced_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "image": {
            "width": int(image_width),
            "height": int(image_height),
            "image_path": image_path,
        },
        "frame_analysis": frame_analysis or {
            "counts": {"per_class": {}},
            "roi_overcrowded": {},
            "events": [],
        },
        "ai_results": results,
    }
    if asset_id is not None:
        message["image"]["asset_id"] = str(asset_id)
    if service_instance_id:
        message["service_instance_id"] = str(service_instance_id)
    return message


def publish_message(message, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, topic=KAFKA_TOPIC):
    """Publish a message to Kafka and wait for delivery."""
    try:
        from confluent_kafka import Producer
    except ImportError as exc:
        raise RuntimeError("confluent-kafka is required to publish messages") from exc

    delivery_error = []

    def delivery_callback(error, _message):
        if error is not None:
            delivery_error.append(error)

    producer = Producer({"bootstrap.servers": bootstrap_servers})
    producer.produce(
        topic,
        key=str(message["cam_id"]),
        value=json.dumps(message, separators=(",", ":")).encode("utf-8"),
        callback=delivery_callback,
    )
    remaining = producer.flush(30)
    if remaining:
        raise RuntimeError(
            f"Kafka delivery timed out with {remaining} message(s) still queued. "
            "Check bootstrap servers, advertised.listeners, firewall, and client network."
        )
    if delivery_error:
        raise RuntimeError(f"Kafka delivery failed: {delivery_error[0]}")


def publish_configured_message():
    """Construct and publish a message using the configured global values."""
    message = construct_message()
    publish_message(message)
    return message
