from __future__ import annotations

import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "push frames sample" / "push frames sample"
if str(SAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(SAMPLE_DIR))

try:
    from construct_ai_message import construct_message, publish_message as sample_publish_message
except ImportError as exc:  # pragma: no cover - fails fast with a clear local path.
    raise RuntimeError(f"Cannot import construct_ai_message.py from {SAMPLE_DIR}") from exc


DEFAULT_BOOTSTRAP_SERVERS = "192.168.1.199:29092"
DEFAULT_TOPIC = "ai.metadata.v1"
DEFAULT_META_TYPE = "person"
SIGNAL_STYLES = {
    "normal": {"label": "normal", "color": "#00C853", "rgb": [0, 200, 83]},
    "fall_candidate": {"label": "fall_candidate", "color": "#FFC107", "rgb": [255, 193, 7]},
    "fall": {"label": "fall", "color": "#FF1744", "rgb": [255, 23, 68]},
}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact_metrics(feature: Any, geometry: Any) -> dict[str, Any]:
    names = (
        "body_angle",
        "aspect_ratio",
        "speed",
        "acceleration",
        "vertical_velocity",
        "vertical_acceleration",
        "body_core_width",
        "body_core_height",
        "pose_confidence",
        "fall_score",
    )
    metrics = {name: _safe_float(getattr(feature, name, None)) for name in names}
    metrics["valid_pose"] = bool(getattr(feature, "valid", False))
    metrics["geometry_reason"] = getattr(geometry, "reason", "")
    return {key: value for key, value in metrics.items() if value is not None and value != ""}


def _keypoints_to_list(detection: Any) -> list[dict[str, Any]]:
    keypoints = getattr(detection, "keypoints", {}) or {}
    values = []
    for name, keypoint in keypoints.items():
        values.append(
            {
                "name": str(name),
                "x": _safe_float(getattr(keypoint, "x", None)),
                "y": _safe_float(getattr(keypoint, "y", None)),
                "confidence": _safe_float(getattr(keypoint, "confidence", None)),
            }
        )
    return values


def _analysis_value(item: dict[str, Any], key: str, default: Any) -> Any:
    if key in item:
        return item[key]
    for analysis in item.get("object_analysis", []):
        if isinstance(analysis, dict) and key in analysis:
            return analysis[key]
    return default


def build_ai_results(
    items: Iterable[tuple[Any, Any, Any]],
    *,
    cam_id: str,
    service_instance_id: str,
    meta_type: str = DEFAULT_META_TYPE,
    include_display_fields: bool = False,
    sample_strict: bool = False,
    name_from_state: bool = False,
    meta_type_from_state: bool = False,
    stable_result_id: bool = True,
    object_id_offset: int = 0,
    result_id_mode: str = "global",
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for detection, geometry, feature in items:
        track_id = int(getattr(feature, "track_id", None) or getattr(detection, "track_id", 0) or 0)
        publish_bbox = getattr(geometry, "body_core_bbox", None) if bool(getattr(geometry, "valid", False)) else None
        x1, y1, x2, y2 = [float(value) for value in (publish_bbox or getattr(detection, "bbox"))]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        state = str(getattr(feature, "state", "NORMAL") or "NORMAL")
        signal = {
            "NORMAL": "normal",
            "FALL_CANDIDATE": "fall_candidate",
            "FALL": "fall",
        }.get(state, state.lower())
        style = SIGNAL_STYLES.get(signal, {"label": signal, "color": "#00C853", "rgb": [0, 200, 83]})
        state_event = str(getattr(feature, "state_event", "") or "")
        is_falling_event = state_event == "falling"
        event_type = str(
            getattr(
                feature,
                "event_type",
                "object_update" if int(getattr(feature, "frame", 0) or 0) == 0 else "object_exist",
            )
        )
        object_key = track_id + int(object_id_offset)
        global_object_id = f"{service_instance_id}:{cam_id}:{object_key}"
        if not stable_result_id:
            result_id = str(uuid.uuid4())
        elif result_id_mode == "uuid":
            result_id = str(uuid.uuid5(uuid.NAMESPACE_URL, global_object_id))
        elif result_id_mode == "track":
            result_id = str(object_key)
        else:
            result_id = global_object_id
        published_meta_type = signal if meta_type_from_state else ("person" if is_falling_event else str(meta_type))
        result = {
            "id": result_id,
            "meta_type": published_meta_type,
            "tracking_object_id": object_key,
            "bbox": {
                "left": int(round(x1)),
                "top": int(round(y1)),
                "width": int(round(width)),
                "height": int(round(height)),
            },
            "detected_object_ids": "person falling" if is_falling_event else "",
            "name": signal if name_from_state else "",
            "event_type": event_type,
            "confidence": _safe_float(getattr(detection, "confidence", None)) or 0.0,
            "object_key": object_key,
            "global_object_id": global_object_id,
            "object_analysis": [
                {
                    "type": "object_analysis",
                    "object_key": object_key,
                    "last_pos": [
                        int(round(x1 + width * 0.5)),
                        int(round(y1 + height * 0.5)),
                    ],
                }
            ],
            "placeholder": {},
        }
        if is_falling_event:
            result["is_blacklist"] = True
        if include_display_fields and not sample_strict:
            result.update(
                {
                    "object_type": "person",
                    "signal": signal,
                    "label": style["label"],
                    "display_label": style["label"],
                    "color": style["color"],
                    "color_rgb": style["rgb"],
                    "name": style["label"],
                    "status": signal,
                    "state_event": state_event,
                    "placeholder": {
                        "display": {
                            "label": style["label"],
                            "color": style["color"],
                            "color_rgb": style["rgb"],
                        }
                    },
                }
            )
            result["object_analysis"][0].update(
                {
                    "state": state,
                    "signal": signal,
                    "label": style["label"],
                    "color": style["color"],
                    "color_rgb": style["rgb"],
                    "metrics": _compact_metrics(feature, geometry),
                    "keypoints": _keypoints_to_list(detection),
                }
            )
        results.append(result)
    return results


def build_frame_analysis(ai_results: list[dict[str, Any]]) -> dict[str, Any]:
    events = [
        {
            "event_type": item["event_type"],
            "signal": _analysis_value(item, "signal", "normal"),
            "status": _analysis_value(item, "signal", "normal"),
            "label": _analysis_value(item, "label", "normal"),
            "color": _analysis_value(item, "color", "#00C853"),
            "color_rgb": _analysis_value(item, "color_rgb", [0, 200, 83]),
            "tracking_object_id": item["tracking_object_id"],
            "object_key": item["object_key"],
            "confidence": item["confidence"],
        }
        for item in ai_results
    ]
    per_signal: dict[str, int] = {}
    for item in ai_results:
        signal = str(_analysis_value(item, "signal", "unknown"))
        per_signal[signal] = per_signal.get(signal, 0) + 1
    return {
        "counts": {
            "per_class": {"person": len(ai_results)},
            "per_signal": per_signal,
        },
        "roi_overcrowded": {},
        "events": events,
    }


def build_realtime_message(
    items: Iterable[tuple[Any, Any, Any]],
    *,
    server_id: str,
    cam_id: str,
    frame_num: int,
    image_width: int,
    image_height: int,
    service_instance_id: str,
    image_path: str = "",
    asset_id: str | None = None,
    meta_type: str = DEFAULT_META_TYPE,
    include_display_fields: bool = False,
    sample_strict: bool = False,
    name_from_state: bool = False,
    meta_type_from_state: bool = False,
    stable_result_id: bool = True,
    object_id_offset: int = 0,
    result_id_mode: str = "global",
) -> dict[str, Any]:
    ai_results = build_ai_results(
        items,
        cam_id=cam_id,
        service_instance_id=service_instance_id,
        meta_type=meta_type,
        include_display_fields=include_display_fields,
        sample_strict=sample_strict,
        name_from_state=name_from_state,
        meta_type_from_state=meta_type_from_state,
        stable_result_id=stable_result_id,
        object_id_offset=object_id_offset,
        result_id_mode=result_id_mode,
    )
    return construct_message(
        server_id=server_id,
        cam_id=cam_id,
        frame_num=frame_num,
        ntp_timestamp=__import__("time").time_ns(),
        ai_results=ai_results,
        image_width=image_width,
        image_height=image_height,
        image_path=image_path,
        asset_id=asset_id,
        frame_analysis=build_frame_analysis(ai_results) if include_display_fields and not sample_strict else None,
        service_instance_id=service_instance_id,
    )


class KafkaFramePublisher:
    def __init__(
        self,
        *,
        bootstrap_servers: str,
        topic: str,
        dry_run: bool = False,
        log_json: bool = False,
        log_delivery: bool = False,
        message_timeout_ms: int = 30000,
        sample_sync: bool = True,
    ) -> None:
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.dry_run = dry_run
        self.log_json = log_json
        self.log_delivery = log_delivery
        self.sample_sync = sample_sync
        self._producer = None
        self._delivery_error: str | None = None
        if not dry_run and not sample_sync:
            try:
                from confluent_kafka import Producer
            except ImportError as exc:
                raise RuntimeError("confluent-kafka is required to publish messages") from exc
            self._producer = Producer(
                {
                    "bootstrap.servers": bootstrap_servers,
                    "message.timeout.ms": int(message_timeout_ms),
                    "socket.timeout.ms": min(10000, int(message_timeout_ms)),
                    "request.timeout.ms": int(message_timeout_ms),
                }
            )

    def publish(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if self.log_json or self.dry_run:
            logging.info("AI metadata message: %s", payload.decode("utf-8"))
        if self.dry_run:
            return
        if self.sample_sync:
            sample_publish_message(message, bootstrap_servers=self.bootstrap_servers, topic=self.topic)
            return
        if self._delivery_error:
            logging.warning("Previous Kafka delivery error: %s", self._delivery_error)
            self._delivery_error = None
        assert self._producer is not None
        self._producer.poll(0)
        self._producer.produce(
            self.topic,
            key=str(message["cam_id"]),
            value=payload,
            callback=self._delivery_callback,
        )

    def close(self, timeout: float = 5.0) -> None:
        if self._producer is not None:
            remaining = self._producer.flush(timeout)
            if remaining:
                raise RuntimeError(
                    f"Kafka delivery timed out with {remaining} message(s) still queued. "
                    "Check broker advertised.listeners and client DNS."
                )
        if self._delivery_error:
            raise RuntimeError(f"Kafka delivery failed: {self._delivery_error}")

    def _delivery_callback(self, error: Any, _message: Any) -> None:
        if error is not None:
            self._delivery_error = str(error)
            logging.error("Kafka delivery failed: %s", self._delivery_error)
        elif self.log_delivery:
            logging.info(
                "Kafka delivered metadata: topic=%s partition=%s offset=%s",
                _message.topic(),
                _message.partition(),
                _message.offset(),
            )

    def should_log_success(self) -> bool:
        return bool(self.log_json or self.log_delivery or self.dry_run)
