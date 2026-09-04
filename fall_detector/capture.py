from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import cv2


def open_stream(url: str, buffer_size: int) -> cv2.VideoCapture:
    # Without FFmpeg timeouts, cap.read() can block forever after an RTSP
    # interruption. A failed read lets the capture loop reconnect normally.
    params: list[int] = []
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000])
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
    try:
        capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params) if params else cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    except (TypeError, cv2.error):
        capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    return capture


class LatestFrameCapture:
    """Threaded capture that preserves every local-video frame and loops it safely."""

    def __init__(self, url: str, buffer_size: int, reconnect_delay: float,
                 loop_video: bool = True, video_loop_count: int = 0) -> None:
        self.url, self.buffer_size, self.reconnect_delay = url, buffer_size, reconnect_delay
        self.loop_video, self.video_loop_count = loop_video, max(0, video_loop_count)
        self._lock, self._stop, self._finished = threading.Lock(), threading.Event(), threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self._frame, self._seq, self._delivered_seq, self._loop_count, self._completed_passes = None, 0, 0, 0, 0
        scheme = urlsplit(url).scheme.lower()
        self._is_file_source = Path(url).is_file() or scheme in {"", "file"}
        self._file_fps, self._next_file_frame_at = 0.0, 0.0
        self._total_frames = 0
        self._consecutive_file_read_failures = 0
        self._last_successful_file_position = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()

    def read_latest(self, last_seq: int, timeout: float = 1.0):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._lock:
                if self._seq != last_seq and self._frame is not None:
                    self._delivered_seq = self._seq
                    return True, self._frame.copy(), self._seq
            time.sleep(0.002)
        return False, None, last_seq

    def release(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()

    def is_finished(self) -> bool:
        return self._finished.is_set()

    def source_frame_interval(self) -> float | None:
        return 1.0 / self._file_fps if self._is_file_source and self._file_fps > 0 else None

    def loop_generation(self) -> int:
        return self._loop_count

    def _open(self) -> bool:
        self._cap = open_stream(self.url, self.buffer_size)
        if not self._cap.isOpened():
            logging.warning("Cannot open camera. Reconnecting in %.1fs...", self.reconnect_delay)
            time.sleep(self.reconnect_delay)
            return False
        logging.info("Camera connected.")
        if self._is_file_source:
            self._file_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 25.0)
            if not 1.0 <= self._file_fps <= 120.0:
                self._file_fps = 25.0
            self._total_frames = max(0, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
            self._next_file_frame_at = time.perf_counter()
        return True

    def total_frames(self) -> int:
        """Number of frames in a local video, or zero when the source is live."""
        return self._total_frames

    def _recover_file_read(self, failed_position: int) -> bool:
        """Retry a transient local-file decoder failure without restarting at frame zero."""
        self._consecutive_file_read_failures += 1
        resume_position = max(0, min(failed_position, max(0, self._total_frames - 1)))
        if self._consecutive_file_read_failures >= 3:
            # A repeatedly corrupt frame cannot be decoded; skip exactly that
            # frame instead of losing the rest of the video on a full restart.
            resume_position = min(resume_position + 1, max(0, self._total_frames - 1))
            logging.warning("Skipping unreadable local-video frame %s after repeated retries.", failed_position)
            self._consecutive_file_read_failures = 0
        if self._cap is not None and self._cap.set(cv2.CAP_PROP_POS_FRAMES, resume_position):
            logging.warning("Local-video read interrupted at frame %s; retrying from frame %s.", failed_position, resume_position)
            return True
        if self._cap is not None:
            self._cap.release()
        self._cap = open_stream(self.url, self.buffer_size)
        if self._cap.isOpened() and self._cap.set(cv2.CAP_PROP_POS_FRAMES, resume_position):
            logging.warning("Reopened local video and resumed from frame %s.", resume_position)
            return True
        logging.warning("Cannot resume local video at frame %s.", resume_position)
        return False

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._cap is None or not self._cap.isOpened():
                if not self._open():
                    continue
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self._is_file_source:
                    # Some decoders advance POS_FRAMES to the end after a
                    # transient failure even though many frames were never
                    # returned. Trust only the last successfully delivered
                    # position when deciding that this is real EOF.
                    position = self._last_successful_file_position
                    at_end = self._total_frames > 0 and position >= self._total_frames
                    if not at_end:
                        if self._recover_file_read(position):
                            continue
                    else:
                        self._consecutive_file_read_failures = 0
                        self._completed_passes += 1
                        logging.info("Completed full video pass %s.", self._completed_passes)
                        if self.video_loop_count and self._completed_passes >= self.video_loop_count:
                            self._finished.set()
                            break
                        if self.loop_video and self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                            self._loop_count += 1
                            self._last_successful_file_position = 0
                            self._next_file_frame_at = time.perf_counter()
                            logging.info("Video reached the end. Restarting loop %s.", self._loop_count)
                            continue
                logging.warning("%s in %.1fs...", "Restarting video replay" if self.loop_video else "Reconnecting", self.reconnect_delay)
                self._cap.release()
                self._cap = None
                time.sleep(self.reconnect_delay)
                continue
            self._consecutive_file_read_failures = 0
            if self._is_file_source:
                self._last_successful_file_position = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                delay = self._next_file_frame_at - time.perf_counter()
                if delay > 0:
                    self._stop.wait(delay)
                self._next_file_frame_at = max(time.perf_counter(), self._next_file_frame_at) + 1.0 / self._file_fps
            with self._lock:
                self._frame, self._seq = frame, self._seq + 1
                published_seq = self._seq
            if self._is_file_source:
                while not self._stop.is_set():
                    with self._lock:
                        if self._delivered_seq >= published_seq:
                            break
                    self._stop.wait(0.002)
