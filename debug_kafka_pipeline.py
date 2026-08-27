from __future__ import annotations

import argparse
import json
import time
import uuid

from confluent_kafka import Consumer, Producer, TopicPartition

from debug_realtime_ai_message import make_debug_items
from realtime_ai_publisher import DEFAULT_BOOTSTRAP_SERVERS, DEFAULT_TOPIC, build_realtime_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish one marked message and watch Kafka output topics.")
    parser.add_argument("--bootstrap", default=DEFAULT_BOOTSTRAP_SERVERS)
    parser.add_argument("--input-topic", default=DEFAULT_TOPIC)
    parser.add_argument("--watch-topic", action="append", default=["ai.metadata.objects.filtered", "ai.metadata.rois.filtered"])
    parser.add_argument("--cam-id", default="CAM_VP_2")
    parser.add_argument("--server-id", default="NODE1")
    parser.add_argument("--state", choices=("normal", "fall_candidate", "fall"), default="fall")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--display-fields", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def tail_assignments(consumer: Consumer, topic: str, tail: int = 20) -> list[TopicPartition]:
    metadata = consumer.list_topics(topic, timeout=8)
    partitions = sorted(metadata.topics[topic].partitions.keys())
    assignments = []
    for partition in partitions:
        low, high = consumer.get_watermark_offsets(TopicPartition(topic, partition), timeout=5)
        assignments.append(TopicPartition(topic, partition, max(low, high - tail)))
        print(f"{topic} partition={partition} low={low} high={high}")
    return assignments


def produce_marker(args: argparse.Namespace, marker: str) -> None:
    message = build_realtime_message(
        make_debug_items(args.state),
        server_id=args.server_id,
        cam_id=args.cam_id,
        frame_num=999999,
        image_width=1920,
        image_height=1080,
        service_instance_id=marker,
        include_display_fields=args.display_fields,
    )
    message["debug_marker"] = marker
    payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    producer = Producer({"bootstrap.servers": args.bootstrap, "message.timeout.ms": 5000})
    errors = []

    def callback(error, produced_message):
        if error:
            errors.append(str(error))
        else:
            print(
                "PRODUCED input topic=%s partition=%s offset=%s key=%s"
                % (
                    produced_message.topic(),
                    produced_message.partition(),
                    produced_message.offset(),
                    produced_message.key().decode() if produced_message.key() else None,
                )
            )

    producer.produce(args.input_topic, key=args.cam_id, value=payload, callback=callback)
    remaining = producer.flush(8)
    if errors:
        raise RuntimeError("; ".join(errors))
    if remaining:
        raise RuntimeError(f"Producer timed out with {remaining} message(s) still queued")


def main() -> None:
    args = parse_args()
    marker = "codex-pipeline-" + str(uuid.uuid4())
    print("MARKER", marker)
    consumers: list[tuple[str, Consumer]] = []
    for topic in args.watch_topic:
        consumer = Consumer(
            {
                "bootstrap.servers": args.bootstrap,
                "group.id": "codex-pipeline-watch-" + str(uuid.uuid4()),
                "auto.offset.reset": "latest",
                "enable.auto.commit": False,
            }
        )
        consumer.assign(tail_assignments(consumer, topic))
        consumers.append((topic, consumer))

    produce_marker(args, marker)
    deadline = time.time() + args.seconds
    found = False
    try:
        while time.time() < deadline:
            for topic, consumer in consumers:
                record = consumer.poll(0.25)
                if record is None:
                    continue
                if record.error():
                    print("CONSUME_ERROR", topic, record.error())
                    continue
                value = record.value().decode("utf-8", errors="replace")
                if marker in value:
                    print(
                        "FOUND marker in topic=%s partition=%s offset=%s key=%s"
                        % (
                            record.topic(),
                            record.partition(),
                            record.offset(),
                            record.key().decode(errors="replace") if record.key() else None,
                        )
                    )
                    print(value[:2000])
                    found = True
            if found:
                break
    finally:
        for _, consumer in consumers:
            consumer.close()
    if not found:
        raise SystemExit("Marker was not observed in watched output topics")


if __name__ == "__main__":
    main()
