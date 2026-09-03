import time
from pathlib import Path

import numpy as np
import pytest

from wisense.core.connection import CSIFrame, FileCSISource, write_capture_csv
from wisense.core.events import EventEmitter, Monitor, StreamWorker
from wisense.core.stream import CSIBuffer
from wisense.presence.detector import PresenceDetector


def test_event_emitter_basic():
    emitter = EventEmitter()
    received = []
    emitter.on("greet", lambda name: received.append(name))
    emitter.emit("greet", "world")
    assert received == ["world"]


def test_event_emitter_off_removes_callback():
    emitter = EventEmitter()
    received = []
    cb = lambda: received.append(1)
    emitter.on("x", cb)
    emitter.off("x", cb)
    emitter.emit("x")
    assert received == []


def test_event_emitter_bad_callback_does_not_break_others():
    emitter = EventEmitter()
    received = []

    def bad():
        raise RuntimeError("boom")

    emitter.on("x", bad)
    emitter.on("x", lambda: received.append(1))
    emitter.emit("x")  # should not raise
    assert received == [1]


def test_event_emitter_unknown_event_is_noop():
    emitter = EventEmitter()
    emitter.emit("nonexistent")  # should not raise


def test_stream_worker_pushes_frames(tmp_path: Path):
    frames = [CSIFrame(timestamp=float(i), amplitude=np.array([1.0, 2.0])) for i in range(5)]
    path = tmp_path / "capture.csv"
    write_capture_csv(path, frames)

    source = FileCSISource(path)
    source.connect()
    buf = CSIBuffer(capacity=10)
    worker = StreamWorker(source, buf, poll_interval=0.01)
    worker.start()
    time.sleep(0.3)
    worker.stop()
    source.disconnect()

    assert len(buf) == 5


def test_monitor_emits_presence_change(tmp_path: Path):
    # Build a capture: quiet frames, then loud (motion) frames, so
    # presence should flip from False to True partway through replay.
    rng = np.random.default_rng(0)
    frames = []
    t = 0.0
    for _ in range(30):
        frames.append(CSIFrame(timestamp=t, amplitude=10.0 + rng.normal(0, 0.05, size=8)))
        t += 0.02
    for _ in range(30):
        frames.append(CSIFrame(timestamp=t, amplitude=10.0 + rng.normal(0, 5.0, size=8)))
        t += 0.02

    path = tmp_path / "capture.csv"
    write_capture_csv(path, frames)

    source = FileCSISource(path, realtime=False)
    monitor = Monitor(
        source,
        window_size=10,
        poll_interval_seconds=0.05,
        presence_detector=PresenceDetector(),
    )
    events = []
    monitor.on_presence_change(lambda result: events.append(result))
    monitor.start()
    time.sleep(1.5)
    monitor.stop()

    assert len(events) >= 1
