"""Buffering and synchronization.

:class:`CSIBuffer` maintains a rolling window of the most recent
:class:`~wisense.core.connection.CSIFrame` objects, keyed by arrival
order with timestamps preserved. It is thread-safe so a background
thread can push frames read from a :class:`~wisense.core.connection.CSISource`
while a foreground thread periodically snapshots the current window for
a detector to consume.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Deque, List, Optional

from wisense.core.connection import CSIFrame

_EventCallback = Callable[[CSIFrame], None]


class CSIBuffer:
    """A fixed-capacity, thread-safe ring buffer of :class:`CSIFrame`.

    Parameters
    ----------
    capacity:
        Maximum number of frames retained. Once full, pushing a new
        frame evicts the oldest one (standard ring-buffer / deque
        ``maxlen`` behaviour).

    Examples
    --------
    >>> buf = CSIBuffer(capacity=100)
    >>> buf.push(frame)  # doctest: +SKIP
    >>> window = buf.snapshot()  # doctest: +SKIP
    >>> len(window) <= 100
    True
    """

    def __init__(self, capacity: int = 256) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._deque: Deque[CSIFrame] = deque(maxlen=capacity)
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._on_push: List[_EventCallback] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)

    def push(self, frame: CSIFrame) -> None:
        """Append a frame, evicting the oldest one if at capacity.
        Thread-safe; wakes any thread blocked in :meth:`wait_for_frames`."""
        with self._not_empty:
            self._deque.append(frame)
            self._not_empty.notify_all()
            callbacks = list(self._on_push)
        for cb in callbacks:
            cb(frame)

    def snapshot(self, n: Optional[int] = None) -> List[CSIFrame]:
        """Return a copy of the current buffer contents, oldest first.

        Parameters
        ----------
        n:
            If given, return only the most recent ``n`` frames (or
            fewer, if the buffer holds less than ``n``). If ``None``,
            return the entire current window.
        """
        with self._lock:
            if n is None:
                return list(self._deque)
            if n <= 0:
                return []
            return list(self._deque)[-n:]

    def clear(self) -> None:
        with self._lock:
            self._deque.clear()

    def wait_for_frames(self, min_count: int, timeout: Optional[float] = None) -> bool:
        """Block until at least ``min_count`` frames are available or
        ``timeout`` seconds elapse. Returns ``True`` if the condition
        was met, ``False`` on timeout."""
        with self._not_empty:
            return self._not_empty.wait_for(
                lambda: len(self._deque) >= min_count, timeout=timeout
            )

    def register_push_callback(self, callback: _EventCallback) -> None:
        """Register a function to be called (synchronously, on the
        pushing thread) every time a frame is pushed. Used by the
        higher-level event system in :mod:`wisense.core.events` to
        implement ``on_presence_change`` / ``on_fall_detected`` style
        hooks without polling."""
        with self._lock:
            self._on_push.append(callback)

    def unregister_push_callback(self, callback: _EventCallback) -> None:
        with self._lock:
            if callback in self._on_push:
                self._on_push.remove(callback)

    def fill_from_source(self, source, max_frames: Optional[int] = None) -> int:
        """Synchronously pull frames from an already-connected
        :class:`~wisense.core.connection.CSISource` until it is
        exhausted or ``max_frames`` have been pushed. Returns the
        number of frames pushed. This is a convenience for offline /
        file-replay use; for live sources prefer running
        :meth:`push` from a background thread via
        :class:`wisense.core.events.StreamWorker`."""
        count = 0
        for frame in source.stream():
            self.push(frame)
            count += 1
            if max_frames is not None and count >= max_frames:
                break
        return count
