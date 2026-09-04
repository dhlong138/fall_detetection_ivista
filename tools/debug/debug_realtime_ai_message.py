from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from types import SimpleNamespace

from tools.realtime_ai_publisher import (
    DEFAULT_BOOTSTRAP_SERVERS,
    DEFAULT_META_TYPE,
    DEFAULT_TOPIC,
    KafkaFramePublisher,
    build_realtime_message,
)


def make_debug_items(state: str, state_event: str = "") -> list[tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]]:
    state = state.upper()
    fall_score = {
        "NORMAL": 0.08,
        "FALL_CANDIDATE": 0.62,
        "FALL": 0.91,
    }.get(state, 0.08)
    keypoints = {
        "nose": SimpleNamespace(x=530.0, y=180.0, confidence=0.88),
        "left_shoulder": SimpleNamespace(x=480.0, y=280.0, confidence=0.91),
        "right_shoulder": SimpleNamespace(x=590.0, y=285.0, confidence=0.90),
        "left_hip": SimpleNamespace(x=500.0, y=480.0, confidence=0.86),
        "right_hip": SimpleNamespace(x=585.0, y=485.0, confidence=0.84),
        "left_ankle": SimpleNamespace(x=505.0, y=770.0, confidence=0.76),
        "right_ankle": SimpleNamespace(x=585.0, y=775.0, confidence=0.78),
    }
    detection = SimpleNamespace(
        bbox=(420.0, 145.0, 680.0, 810.0),
        confidence=0.94,
        keypoints=keypoints,
        track_id=101,
    )
    geometry = SimpleNamespace(reason="")
    feature = SimpleNamespace(
        track_id=101,
        valid=True,
        state=state,
        fall_score=fall_score,
        body_angle=12.5,
        aspect_ratio=2.56,
        speed=0.14,
        acceleration=0.03,
        vertical_velocity=0.10,
        vertical_acceleration=0.02,
        body_core_width=145.0,
        body_core_height=430.0,
        pose_confidence=0.94,
        state_event=state_event,
    )
    return [(detection, geometry, feature)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or publish one realtime AI metadata debug message.")
    parser.add_argument("--publish", action=argparse.BooleanOptionalAction, default=False, help="Send the debug message to Kafka.")
    parser.add_argument("--kafka-bootstrap-servers", default=DEFAULT_BOOTSTRAP_SERVERS)
    parser.add_argument("--kafka-topic", default=DEFAULT_TOPIC)
    parser.add_argument("--server-id", default="NODE1")
    parser.add_argument("--cam-id", default="CAM_VP_1")
    parser.add_argument("--meta-type", default=DEFAULT_META_TYPE)
    parser.add_argument("--frame-num", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=1920)
    parser.add_argument("--image-height", type=int, default=1080)
    parser.add_argument("--service-instance-id", default=str(uuid.uuid4()))
    parser.add_argument("--state", choices=("normal", "fall_candidate", "fall"), default="normal")
    parser.add_argument("--state-event", choices=("", "falling"), default="")
    parser.add_argument("--display-fields", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fast-async", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--frames", type=int, default=1, help="Number of messages to send when publishing.")
    parser.add_argument("--fps", type=float, default=30.0, help="Publish rate when --frames is greater than 1.")
    return parser.parse_args()


def build_debug_message(args: argparse.Namespace, frame_num: int) -> dict:
    return build_realtime_message(
        make_debug_items(args.state, args.state_event),
        server_id=args.server_id,
        cam_id=args.cam_id,
        frame_num=frame_num,
        image_width=args.image_width,
        image_height=args.image_height,
        service_instance_id=args.service_instance_id,
        meta_type=args.meta_type,
        include_display_fields=args.display_fields,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    message = build_debug_message(args, args.frame_num)
    print(json.dumps(message, indent=2, ensure_ascii=False))
    if args.publish:
        publisher = KafkaFramePublisher(
            bootstrap_servers=args.kafka_bootstrap_servers,
            topic=args.kafka_topic,
            sample_sync=not args.fast_async,
        )
        try:
            frame_interval = 1.0 / max(0.001, args.fps)
            next_frame_time = time.monotonic()
            for offset in range(max(1, args.frames)):
                message = build_debug_message(args, args.frame_num + offset)
                publisher.publish(message)
                if args.frames > 1:
                    logging.info(
                        "Published debug metadata: cam_id=%s frame=%s state=%s event=%s",
                        args.cam_id,
                        args.frame_num + offset,
                        args.state,
                        args.state_event or "none",
                    )
                next_frame_time += frame_interval
                delay = next_frame_time - time.monotonic()
                if delay > 0 and offset < args.frames - 1:
                    time.sleep(delay)
        finally:
            publisher.close()


if __name__ == "__main__":
    main()
