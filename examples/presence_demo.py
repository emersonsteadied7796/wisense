"""Presence detection demo -- runs end-to-end with zero hardware.

Replays the small synthetic capture bundled in
``tests/fixtures/sample_capture.csv`` (~3s empty room, ~4s of
simulated motion, ~3s empty room again, at 20 Hz) through
FileCSISource, calibrates on the initial quiet segment, then runs
PresenceDetector over sliding windows across the whole file and prints
the result for each window.

Run it from the repository root with:

    python examples/presence_demo.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from wisense.core.calibration import calibrate
from wisense.core.connection import FileCSISource
from wisense.core.stream import CSIBuffer
from wisense.presence.detector import PresenceDetector

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_capture.csv"

WINDOW_SIZE = 20  # frames per detection window
STEP = 20  # advance this many frames between windows (non-overlapping here)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not FIXTURE_PATH.exists():
        raise SystemExit(
            f"Fixture capture not found at {FIXTURE_PATH}. Run this script from "
            "within a checkout of the wisense repository."
        )

    print(f"Replaying {FIXTURE_PATH.name} (no hardware required)\n")

    # 1. Calibrate against the first ~2 seconds of the recording,
    #    which is the quiet/empty-room portion of this fixture.
    #    realtime=True paces replay against the frames' own recorded
    #    timestamps so the calibration duration means what it says.
    with FileCSISource(FIXTURE_PATH, realtime=True) as calib_source:
        profile = calibrate(calib_source, duration_seconds=2.0)
    print(
        f"Calibration complete: {profile.n_frames} frames, "
        f"{profile.n_subcarriers} subcarriers, "
        f"variance threshold={profile.variance_threshold():.3f}\n"
    )

    # 2. Replay the whole file (fast, not real-time) into a buffer.
    with FileCSISource(FIXTURE_PATH) as source:
        buffer = CSIBuffer(capacity=1024)
        n_loaded = buffer.fill_from_source(source)
    print(f"Loaded {n_loaded} frames from the capture.\n")

    # 3. Slide a detection window across the buffer and print results.
    detector = PresenceDetector()
    all_frames = buffer.snapshot()

    print(f"{'window':>8}  {'t_start':>8}  {'t_end':>8}  {'present':>8}  {'confidence':>10}")
    for start in range(0, len(all_frames) - WINDOW_SIZE + 1, STEP):
        window = all_frames[start : start + WINDOW_SIZE]
        result = detector.detect(window, calibration=profile)
        print(
            f"{start:>8}  {window[0].timestamp:>8.2f}  {window[-1].timestamp:>8.2f}  "
            f"{str(result.present):>8}  {result.confidence:>10.3f}"
        )


if __name__ == "__main__":
    main()
