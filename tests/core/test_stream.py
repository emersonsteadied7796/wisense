import threading
import time

import numpy as np
import pytest

from wisense.core.connection import CSIFrame
from wisense.core.stream import CSIBuffer


def _frame(i: float) -> CSIFrame:
    return CSIFrame(timestamp=i, amplitude=np.array([i, i + 1.0]))


def test_push_and_snapshot_order():
    buf = CSIBuffer(capacity=10)
    for i in range(5):
        buf.push(_frame(float(i)))
    snap = buf.snapshot()
    assert [f.timestamp for f in snap] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_capacity_eviction():
    buf = CSIBuffer(capacity=3)
    for i in range(5):
        buf.push(_frame(float(i)))
    snap = buf.snapshot()
    assert len(snap) == 3
    assert [f.timestamp for f in snap] == [2.0, 3.0, 4.0]


def test_snapshot_n_most_recent():
    buf = CSIBuffer(capacity=10)
    for i in range(5):
        buf.push(_frame(float(i)))
    snap = buf.snapshot(n=2)
    assert [f.timestamp for f in snap] == [3.0, 4.0]


def test_snapshot_n_larger_than_buffer():
    buf = CSIBuffer(capacity=10)
    buf.push(_frame(0.0))
    snap = buf.snapshot(n=100)
    assert len(snap) == 1


def test_snapshot_n_zero_or_negative():
    buf = CSIBuffer(capacity=10)
    buf.push(_frame(0.0))
    assert buf.snapshot(n=0) == []
    assert buf.snapshot(n=-1) == []


def test_clear():
    buf = CSIBuffer(capacity=10)
    buf.push(_frame(0.0))
    buf.clear()
    assert len(buf) == 0


def test_invalid_capacity_raises():
    with pytest.raises(ValueError):
        CSIBuffer(capacity=0)


def test_len():
    buf = CSIBuffer(capacity=10)
    assert len(buf) == 0
    buf.push(_frame(0.0))
    assert len(buf) == 1


def test_push_callback_invoked():
    buf = CSIBuffer(capacity=10)
    received = []
    buf.register_push_callback(lambda f: received.append(f.timestamp))
    buf.push(_frame(1.0))
    buf.push(_frame(2.0))
    assert received == [1.0, 2.0]


def test_unregister_push_callback():
    buf = CSIBuffer(capacity=10)
    received = []
    cb = lambda f: received.append(f.timestamp)
    buf.register_push_callback(cb)
    buf.push(_frame(1.0))
    buf.unregister_push_callback(cb)
    buf.push(_frame(2.0))
    assert received == [1.0]


def test_thread_safety_concurrent_pushes():
    buf = CSIBuffer(capacity=1000)

    def pusher(start):
        for i in range(100):
            buf.push(_frame(float(start + i)))

    threads = [threading.Thread(target=pusher, args=(t * 1000,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(buf) == 400


def test_wait_for_frames_returns_true_when_satisfied():
    buf = CSIBuffer(capacity=10)

    def delayed_push():
        time.sleep(0.1)
        buf.push(_frame(0.0))

    threading.Thread(target=delayed_push).start()
    assert buf.wait_for_frames(1, timeout=2.0) is True


def test_wait_for_frames_times_out():
    buf = CSIBuffer(capacity=10)
    assert buf.wait_for_frames(5, timeout=0.2) is False


def test_fill_from_source():
    from wisense.core.connection import FileCSISource, write_capture_csv
    import tempfile
    from pathlib import Path

    frames = [_frame(float(i)) for i in range(6)]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "capture.csv"
        write_capture_csv(path, frames)
        with FileCSISource(path) as source:
            buf = CSIBuffer(capacity=10)
            count = buf.fill_from_source(source)
    assert count == 6
    assert len(buf) == 6
