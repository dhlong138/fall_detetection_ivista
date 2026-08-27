from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Any

from confluent_kafka import Consumer

from realtime_ai_publisher import DEFAULT_BOOTSTRAP_SERVERS, DEFAULT_TOPIC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor AI metadata messages for one camera.")
    parser.add_argument("--bootstrap", default=DEFAULT_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--cam-id", default="CAM_VP_1")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--warn-gap-ms", type=float, default=120.0)
    parser.add_argument("--print-every", type=int, default=1)
    return parser.parse_args()


def object_summary(message: dict[str, Any]) -> list[tuple[Any, Any, Any, Any]]:
    rows = []
    for item in message.get("ai_results") or []:
        rows.append(
            (
                item.get("tracking_object_id"),
                item.get("object_key"),
                item.get("event_type"),
                item.get("id"),
            )
        )
    return rows


def main() -> None:
    args = parse_args()
    consumer = Consumer(
        {
            "bootstrap.servers": args.bootstrap,
            "group.id": "codex-cam-monitor-" + str(uuid.uuid4()),
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([args.topic])
    deadline = time.time() + args.seconds
    last_time: float | None = None
    last_frame: int | None = None
    seen = 0
    empty = 0
    max_gap_ms = 0.0

    print(f"Monitoring topic={args.topic} cam_id={args.cam_id} seconds={args.seconds}")
    try:
        while time.time() < deadline:
            record = consumer.poll(0.5)
            if record is None:
                continue
            if record.error():
                print("CONSUME_ERROR", record.error())
                continue
            try:
                message = json.loads(record.value().decode("utf-8", errors="replace"))
            except json.JSONDecodeError as exc:
                print("BAD_JSON", record.topic(), record.partition(), record.offset(), exc)
                continue
            if str(message.get("cam_id")) != str(args.cam_id):
                continue

            now = time.perf_counter()
            frame = int(message.get("frame_num", -1))
            objects = object_summary(message)
            seen += 1
            if not objects:
                empty += 1
            gap_ms = 0.0 if last_time is None else (now - last_time) * 1000.0
            frame_gap = 0 if last_frame is None else frame - last_frame
            max_gap_ms = max(max_gap_ms, gap_ms)
            last_time = now
            last_frame = frame

            warn = " WARN_GAP" if gap_ms >= args.warn_gap_ms else ""
            if seen % max(1, args.print_every) == 0 or warn or not objects:
                print(
                    "cam=%s frame=%s frame_gap=%s gap_ms=%.1f objects=%s offset=%s%s"
                    % (args.cam_id, frame, frame_gap, gap_ms, objects, record.offset(), warn)
                )
    finally:
        consumer.close()
    print(f"SUMMARY cam={args.cam_id} seen={seen} empty={empty} max_gap_ms={max_gap_ms:.1f}")


if __name__ == "__main__":
    main()
