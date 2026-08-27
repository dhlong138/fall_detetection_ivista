"""Backward-compatible entry point for realtime pose and fall detection.

Implementation modules live under :mod:`fall_detector`; keep this file so
existing commands and automation do not need to change.
"""

from fall_detector.runtime import *  # noqa: F401,F403 - public compatibility API
from fall_detector.runtime import main


if __name__ == "__main__":
    main()
