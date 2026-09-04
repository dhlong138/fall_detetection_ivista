"""Convert an existing OpenCV MP4V video to browser-compatible H.264."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an H.264 MP4 playable in a browser.")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    source, destination = Path(args.input), Path(args.output)
    print(f"Converting {source.name} to browser-compatible H.264...", flush=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open input video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.CAP_MSMF, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit("H.264 encoder is unavailable on this machine.")
    print(f"Output: {destination}", flush=True)
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            count += 1
            if count % 300 == 0:
                print(f"Converted {count} frames")
    finally:
        capture.release()
        writer.release()
    print(f"Saved browser-compatible video: {destination} ({count} frames)")


if __name__ == "__main__":
    main()
