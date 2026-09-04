from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

# Keep RTSP reads from blocking forever when the replay source reaches EOF.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")

import cv2

# This script is kept outside the root entry point, so add the project root
# before importing the backward-compatible detector runner.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import single_realtime_yolo11x_pose as realtime


FRAME_STALE_SECONDS = 4.0
REOPEN_GRACE_SECONDS = 2.0
OPEN_TIMEOUT_MSEC = 8000
READ_TIMEOUT_MSEC = 2500
CACHE_PATH = Path("CAMTESTFALLING_rtsp_loop_cache.avi")
PARTIAL_CACHE_PATH = Path("CAMTESTFALLING_rtsp_loop_cache.partial.avi")
DEFAULT_CACHE_FPS = 25.0
MIN_CACHE_FRAMES = 2000


class LoopingLatestFrameCapture:
    """Reconnect forever when an RTSP replay stream reaches the end of its video."""

    def __init__(
        self,
        url: str,
        buffer_size: int,
        reconnect_delay: float,
        loop_video: bool = True,
        video_loop_count: int = 0,
    ) -> None:
        self.url = url
        self.buffer_size = buffer_size
        self.reconnect_delay = max(0.05, reconnect_delay)
        self.loop_video = loop_video
        self.video_loop_count = max(0, video_loop_count)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self._frame = None
        self._seq = 0
        self._opened = False
        self._opening = False
        self._frames_since_open = 0
        self._loop_count = 0
        self._last_frame_at = time.perf_counter()
        self._last_reopen_at = 0.0
        self._reopen_requested = threading.Event()
        self._cache_writer: cv2.VideoWriter | None = None
        self._cache_frames = 0
        self._cache_ready = False
        self._using_cache = False
        self._cache_fps = DEFAULT_CACHE_FPS
        self._next_cache_frame_at = 0.0
        self._load_existing_cache_if_usable()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="looping-camera-capture", daemon=True)
        self._thread.start()

    def read_latest(self, last_seq: int, timeout: float = 1.0):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._lock:
                if self._seq != last_seq and self._frame is not None:
                    return True, self._frame.copy(), self._seq
                stale_seconds = time.perf_counter() - self._last_frame_at
                can_reopen = self._opened and not self._opening and self._frames_since_open > 0
            if can_reopen and stale_seconds >= FRAME_STALE_SECONDS and not self._reopen_requested.is_set():
                logging.warning("No fresh frame for %.1fs. Forcing RTSP replay reopen...", stale_seconds)
                with self._lock:
                    self._last_frame_at = time.perf_counter()
                self._reopen_requested.set()
            time.sleep(0.002)
        return False, None, last_seq

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._close_capture()
        self._close_cache_writer()
        self._delete_cache(PARTIAL_CACHE_PATH, "incomplete replay cache")

    def is_finished(self) -> bool:
        return False

    def source_frame_interval(self) -> float | None:
        return None

    def loop_generation(self) -> int:
        return self._loop_count

    def _open_capture(self) -> cv2.VideoCapture:
        if self._cache_ready and CACHE_PATH.exists():
            self._using_cache = True
            self._next_cache_frame_at = time.perf_counter()
            logging.info("Opening local replay cache: %s", CACHE_PATH)
            return cv2.VideoCapture(str(CACHE_PATH))

        self._using_cache = False
        params = []
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MSEC])
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MSEC])
        if params:
            try:
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
            except TypeError:
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        return cap

    def _ensure_cache_writer(self, frame) -> None:
        if self._cache_ready or self._cache_writer is not None:
            return
        height, width = frame.shape[:2]
        fps = DEFAULT_CACHE_FPS
        if self._cap is not None:
            detected_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if 1.0 <= detected_fps <= 120.0:
                fps = detected_fps
        self._cache_fps = fps
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._cache_writer = cv2.VideoWriter(str(PARTIAL_CACHE_PATH), fourcc, fps, (width, height))
        if not self._cache_writer.isOpened():
            logging.warning("Cannot create replay cache video: %s", PARTIAL_CACHE_PATH)
            self._cache_writer.release()
            self._cache_writer = None
            return
        logging.info("Recording first RTSP pass to incomplete cache: %s", PARTIAL_CACHE_PATH)

    def _write_cache_frame(self, frame) -> None:
        if self._using_cache or self._cache_ready:
            return
        self._ensure_cache_writer(frame)
        if self._cache_writer is not None:
            self._cache_writer.write(frame)
            self._cache_frames += 1

    def _close_cache_writer(self) -> None:
        if self._cache_writer is not None:
            self._cache_writer.release()
            self._cache_writer = None

    def _delete_cache(self, path: Path, description: str = "replay cache") -> None:
        if path.exists():
            try:
                path.unlink()
                logging.info("Deleted %s: %s", description, path)
            except OSError as exc:
                logging.warning("Cannot delete %s %s: %s", description, path, exc)

    def _load_existing_cache_if_usable(self) -> None:
        self._delete_cache(PARTIAL_CACHE_PATH, "incomplete replay cache")
        if not CACHE_PATH.exists():
            return
        cap = cv2.VideoCapture(str(CACHE_PATH))
        try:
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        finally:
            cap.release()
        if frames >= MIN_CACHE_FRAMES:
            self._cache_frames = frames
            self._cache_fps = fps if 1.0 <= fps <= 120.0 else DEFAULT_CACHE_FPS
            self._cache_ready = True
            logging.info("Using existing replay cache: %s frames at %.2f fps.", frames, self._cache_fps)
            return
        logging.warning("Existing replay cache is too short: %s frames. Deleting it.", frames)
        self._delete_cache(CACHE_PATH, "stale replay cache")

    def _finalize_cache_if_possible(self) -> bool:
        if self._cache_ready:
            return True
        self._close_cache_writer()
        if self._cache_frames <= 0 or not PARTIAL_CACHE_PATH.exists():
            return False
        if self._cache_frames < MIN_CACHE_FRAMES:
            logging.warning(
                "Discarding too-short replay cache: %s frames, need at least %s. Will retry RTSP.",
                self._cache_frames,
                MIN_CACHE_FRAMES,
            )
            self._delete_cache(PARTIAL_CACHE_PATH, "too-short replay cache")
            self._cache_frames = 0
            return False
        try:
            PARTIAL_CACHE_PATH.replace(CACHE_PATH)
        except OSError as exc:
            logging.warning("Cannot finalize replay cache %s: %s", CACHE_PATH, exc)
            self._cache_frames = 0
            return False
        self._cache_ready = True
        logging.info("Replay cache ready with %s frames. Switching to local loop playback.", self._cache_frames)
        return True

    def _pace_cache_playback(self) -> None:
        if not self._using_cache:
            return
        now = time.perf_counter()
        if self._next_cache_frame_at > now:
            time.sleep(self._next_cache_frame_at - now)
        self._next_cache_frame_at = max(time.perf_counter(), self._next_cache_frame_at) + (1.0 / max(1.0, self._cache_fps))

    def _close_capture(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._opened = False
            self._opening = False
            self._frames_since_open = 0

    def _reset_for_next_loop(self) -> None:
        self._loop_count += 1
        if self._finalize_cache_if_possible():
            logging.info("RTSP replay ended. Looping from local cache.")
            self._close_capture()
            return

        now = time.perf_counter()
        with self._lock:
            self._last_frame_at = now
        since_last_reopen = now - self._last_reopen_at
        if since_last_reopen < REOPEN_GRACE_SECONDS:
            sleep_seconds = max(self.reconnect_delay, REOPEN_GRACE_SECONDS - since_last_reopen)
        else:
            sleep_seconds = self.reconnect_delay
        self._last_reopen_at = now + sleep_seconds
        logging.info(
            "RTSP replay ended or disconnected. Reopening loop %s in %.2fs...",
            self._loop_count,
            sleep_seconds,
        )
        self._close_capture()
        time.sleep(sleep_seconds)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._reopen_requested.is_set():
                self._reopen_requested.clear()
                if not self._using_cache:
                    logging.warning("RTSP stalled before confirmed EOF; discarding incomplete cache and reconnecting.")
                    self._close_cache_writer()
                    self._delete_cache(PARTIAL_CACHE_PATH, "incomplete replay cache")
                    self._cache_frames = 0
                    self._close_capture()
                    time.sleep(self.reconnect_delay)
                else:
                    self._reset_for_next_loop()
                continue

            if self._cap is None or not self._cap.isOpened():
                with self._lock:
                    self._opening = True
                    self._opened = False
                    self._frames_since_open = 0
                    self._last_frame_at = time.perf_counter()
                self._cap = self._open_capture()
                with self._lock:
                    self._opened = self._cap.isOpened()
                    self._opening = False
                    self._last_frame_at = time.perf_counter()
                if not self._cap.isOpened():
                    logging.warning("Cannot open replay stream. Retrying in %.2fs...", self.reconnect_delay)
                    self._close_capture()
                    time.sleep(self.reconnect_delay)
                    continue
                logging.info("Replay stream connected.")

            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self._using_cache:
                    self._loop_count += 1
                    logging.info("Local replay cache loop %s.", self._loop_count)
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._next_cache_frame_at = time.perf_counter()
                    continue
                self._reset_for_next_loop()
                continue

            self._pace_cache_playback()
            self._write_cache_frame(frame)
            with self._lock:
                self._frame = frame
                self._seq += 1
                self._frames_since_open += 1
                self._last_frame_at = time.perf_counter()


def main() -> None:
    realtime.LatestFrameCapture = LoopingLatestFrameCapture
    realtime.main()


if __name__ == "__main__":
    main()
