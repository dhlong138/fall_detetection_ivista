"""Asynchronous visual confirmation for newly detected fall events."""

from __future__ import annotations

import base64
import json
import logging
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import Any

import cv2
import numpy as np
import requests


@dataclass(frozen=True)
class FallVerificationResult:
    track_id: int
    frame_index: int
    confirmed: bool
    reason: str
    frame: np.ndarray
    bbox: tuple[float, float, float, float] | None = None


class FallLLMVerifier:
    """Verifies the first frame of each fall asynchronously and fail-closed."""

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_workers: int,
        image_max_width: int,
        jpeg_quality: int,
        retry_after_frames: int,
        system_prompt: str | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.image_max_width = image_max_width
        self.jpeg_quality = jpeg_quality
        self.retry_after_frames = max(1, retry_after_frames)
        self.system_prompt = str(system_prompt or "").strip() or (
            "You verify CCTV fall alerts. Respond with JSON only: "
            '{"is_fall":true|false,"reason":"short reason"}. '
            "Set is_fall true only when the person is visibly falling or lying on the ground "
            "in a way consistent with a real fall."
        )
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="fall-llm")
        self._results: SimpleQueue[FallVerificationResult] = SimpleQueue()
        self._pending: set[int] = set()
        self._decisions: dict[int, bool] = {}
        self._last_submitted_frame: dict[int, int] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        track_id: int,
        frame_index: int,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float] | None = None,
        reference_frame: np.ndarray | None = None,
        reference_frame_index: int | None = None,
        before_frame: np.ndarray | None = None,
        after_frame: np.ndarray | None = None,
    ) -> bool:
        """Queue one verification per active track. Returns True when queued."""
        with self._lock:
            # A semantic LLM decision (true *or* false) settles this raw FALL
            # episode.  Re-querying a rejected person every retry interval is
            # both noisy and can turn a standing-up sequence into a later
            # false positive.  Transport failures remain retryable below.
            if track_id in self._pending or track_id in self._decisions:
                return False
            previous_frame = self._last_submitted_frame.get(track_id)
            if previous_frame is not None and frame_index - previous_frame < self.retry_after_frames:
                return False
            self._pending.add(track_id)
            self._last_submitted_frame[track_id] = frame_index
        future = self._executor.submit(
            self._verify,
            track_id,
            frame_index,
            frame.copy(),
            bbox,
            reference_frame.copy() if reference_frame is not None else None,
            reference_frame_index,
            before_frame.copy() if before_frame is not None else None,
            after_frame.copy() if after_frame is not None else None,
        )
        future.add_done_callback(self._on_done)
        logging.info("Queued LLM fall verification: track=%s frame=%s", track_id, frame_index)
        return True

    def _on_done(self, future: Future[FallVerificationResult]) -> None:
        try:
            result = future.result()
        except Exception as exc:  # Defensive: _verify normally converts errors to a rejection.
            logging.exception("Unexpected LLM fall verification worker failure: %s", exc)
            return
        with self._lock:
            self._pending.discard(result.track_id)
            if result.reason.startswith("LLM verification failed:"):
                # Do not settle an episode on a timeout/connectivity failure;
                # submit() may retry after retry_after_frames.
                self._decisions.pop(result.track_id, None)
            else:
                self._decisions[result.track_id] = result.confirmed
        self._results.put(result)

    def drain_results(self) -> list[FallVerificationResult]:
        results: list[FallVerificationResult] = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except Empty:
                return results

    def is_approved(self, track_id: int) -> bool:
        with self._lock:
            return self._decisions.get(track_id) is True

    def forget_inactive(self, active_track_ids: set[int]) -> None:
        """Allow a fresh verification after a track leaves FALL and later re-enters it."""
        with self._lock:
            for track_id in list(self._decisions):
                if track_id not in active_track_ids:
                    self._decisions.pop(track_id, None)
                    self._last_submitted_frame.pop(track_id, None)

    def invalidate(self, track_id: int) -> None:
        """Suppress a decision when its confirmed image cannot be prepared."""
        with self._lock:
            self._decisions.pop(track_id, None)
            self._last_submitted_frame.pop(track_id, None)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _verify(
        self,
        track_id: int,
        frame_index: int,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float] | None,
        reference_frame: np.ndarray | None = None,
        reference_frame_index: int | None = None,
        before_frame: np.ndarray | None = None,
        after_frame: np.ndarray | None = None,
    ) -> FallVerificationResult:
        try:
            user_content: list[dict[str, Any]] = [
                {"type": "text", "text": "Does this CCTV image show a real human fall?"},
                {"type": "image_url", "image_url": {"url": self._encode_image(frame)}},
            ]
            if before_frame is not None and after_frame is not None:
                user_content = [
                    {"type": "text", "text": "Images 1, 2, and 3 are 0.5 seconds before, at, and 0.5 seconds after the fall candidate. Confirm is_fall=true only when this short sequence shows a real uncontrolled fall."},
                    {"type": "image_url", "image_url": {"url": self._encode_image(before_frame)}},
                    {"type": "image_url", "image_url": {"url": self._encode_image(frame)}},
                    {"type": "image_url", "image_url": {"url": self._encode_image(after_frame)}},
                ]
            if reference_frame is not None:
                user_content = [
                    {
                        "type": "text",
                        "text": (
                            f"Image 1 is an earlier LLM-confirmed fall (frame {reference_frame_index}). "
                            "Images 2, 3, and 4 are 0.5 seconds before, at, and 0.5 seconds after the current candidate. "
                            "Return is_fall=true only for a NEW fall; return is_fall=false for continuation or recovery."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": self._encode_image(reference_frame)}},
                    {"type": "image_url", "image_url": {"url": self._encode_image(before_frame if before_frame is not None else frame)}},
                    {"type": "image_url", "image_url": {"url": self._encode_image(frame)}},
                    {"type": "image_url", "image_url": {"url": self._encode_image(after_frame if after_frame is not None else frame)}},
                ]
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                "temperature": 0,
            }
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            confirmed, reason = self._parse_decision(content)
            return FallVerificationResult(track_id, frame_index, confirmed, reason, frame, bbox)
        except Exception as exc:
            return FallVerificationResult(track_id, frame_index, False, f"LLM verification failed: {exc}", frame, bbox)

    def _encode_image(self, frame: np.ndarray) -> str:
        image = frame
        height, width = image.shape[:2]
        if self.image_max_width > 0 and width > self.image_max_width:
            resized_height = max(1, round(height * self.image_max_width / width))
            image = cv2.resize(image, (self.image_max_width, resized_height), interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not success:
            raise RuntimeError("Cannot JPEG-encode fall verification frame")
        return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

    @staticmethod
    def _parse_decision(content: Any) -> tuple[bool, str]:
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
        text = str(content or "").strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        data = json.loads(match.group(0) if match else text)
        confirmed = data.get("is_fall") is True
        return confirmed, str(data.get("reason") or "No reason supplied")
