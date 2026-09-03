"""Live ESP32 presence + fall monitoring demo.

*** REQUIRES PHYSICAL HARDWARE ***

This script does NOT run without a real ESP32 flashed with
ESP32-CSI-Tool-compatible firmware
(https://github.com/StevenMHernandez/ESP32-CSI-Tool), connected over
USB serial. For a hardware-free walkthrough of the same concepts, see
``examples/presence_demo.py`` instead.

Usage
-----

    python examples/live_esp32_demo.py /dev/ttyUSB0

(On Windows, the port will look like ``COM3``; on macOS, something
like ``/dev/cu.usbserial-0001``.)

What it does
------------
1. Connects to the ESP32 over serial.
2. Calibrates for 30 seconds -- make sure the room is empty during
   this step.
3. Starts a background Monitor that polls a rolling window of recent
   frames, printing presence changes and any detected falls as they
   happen, until interrupted with Ctrl+C.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from wisense.core.calibration import calibrate
from wisense.core.connection import SerialCSISource
from wisense.core.events import Monitor
from wisense.fall.detector import FallDetector
from wisense.presence.detector import PresenceDetector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", help="Serial port the ESP32 is connected to, e.g. /dev/ttyUSB0 or COM3")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--calibration-seconds", type=float, default=30.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    source = SerialCSISource(port=args.port, baudrate=args.baudrate)

    print(f"Connecting to ESP32 on {args.port} @ {args.baudrate} baud...")
    with source:
        print(
            f"Calibrating for {args.calibration_seconds:.0f}s -- "
            "make sure the room is EMPTY right now.\n"
        )
        profile = calibrate(source, duration_seconds=args.calibration_seconds)
        print(
            f"Calibration complete: {profile.n_frames} frames, "
            f"{profile.n_subcarriers} subcarriers.\n"
        )

    # Monitor manages its own connect/disconnect of the source, so we
    # exit the `with` block above before handing it off.
    monitor = Monitor(
        source,
        calibration=profile,
        presence_detector=PresenceDetector(),
        fall_detector=FallDetector(),
        poll_interval_seconds=0.5,
    )

    def on_presence(result):
        state = "PRESENT" if result.present else "ABSENT"
        print(f"[presence] {state} (confidence={result.confidence:.2f})")

    def on_fall(event):
        print(
            f"[FALL DETECTED] severity={event.severity.value} "
            f"confidence={event.confidence:.2f} at t={event.detected_at:.2f}"
        )

    monitor.on_presence_change(on_presence)
    monitor.on_fall_detected(on_fall)

    print("Monitoring started. Press Ctrl+C to stop.\n")
    monitor.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        monitor.stop()


if __name__ == "__main__":
    sys.exit(main())
