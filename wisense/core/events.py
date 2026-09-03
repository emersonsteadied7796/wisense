"""Pluggable event callback system.

:class:`EventEmitter` is a minimal, thread-safe observer-pattern
implementation: callers register callbacks for named events, and any
thread can dispatch an event, which synchronously invokes every
registered callback for that event name (callback exceptions are
caught and logged so one broken callback can't kill the dispatch loop
or a background streaming thread).

:class:`StreamWorker` runs a :class:`~wisense.core.connection.CSISource`
in a background thread, pushing frames into a
:class:`~wisense.core.stream.CSIBuffer` as they arrive.

:class:`Monitor` ties a source, buffer, and worker together with the
presence/fall detectors to provide the ``on_presence_change`` /
``on_fall_detected`` hooks described in the project brief, polling the
buffer at a configurable interval and emitting events only on state
transitions (not on every poll).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, DefaultDict, List, Optional
from collections import defaultdict

from wisense.core.calibration import CalibrationProfile
from wisense.core.connection import CSISource
from wisense.core.stream import CSIBuffer

logger = logging.getLogger("wisense.core.events")


class EventEmitter:
    """Thread-safe, minimal observer-pattern event dispatcher.

    Examples
    --------
    >>> emitter = EventEmitter()
    >>> received = []
    >>> emitter.on("greet", lambda name: received.append(name))
    >>> emitter.emit("greet", "world")
    >>> received
    ['world']
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._listeners: DefaultDict[str, List[Callable[..., None]]] = defaultdict(list)

    def on(self, event: str, callback: Callable[..., None]) -> None:
        """Register ``callback`` to be invoked whenever ``event`` fires."""
        with self._lock:
            self._listeners[event].append(callback)

    def off(self, event: str, callback: Callable[..., None]) -> None:
        """Unregister a previously registered callback. No-op if it
        wasn't registered."""
        with self._lock:
            if callback in self._listeners[event]:
                self._listeners[event].remove(callback)

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Synchronously invoke every callback registered for ``event``.
        A callback raising an exception is logged and does not prevent
        the remaining callbacks from running."""
        with self._lock:
            callbacks = list(self._listeners.get(event, ()))
        for cb in callbacks:
            try:
                cb(*args, **kwargs)
            except Exception:  # noqa: BLE001 - a bad callback must not crash the dispatcher
                logger.exception("Exception raised by callback for event %r", event)


class StreamWorker:
    """Runs a connected :class:`CSISource` in a background thread,
    pushing every frame it reads into a :class:`CSIBuffer`.

    The source must already be connected before calling :meth:`start`;
    :class:`StreamWorker` does not own the source's lifecycle beyond
    reading from it (call ``source.disconnect()`` yourself once
    stopped, or manage the source with its own ``with`` block around
    the worker's lifetime).
    """

    def __init__(self, source: CSISource, buffer: CSIBuffer, poll_interval: float = 0.0) -> None:
        self.source = source
        self.buffer = buffer
        self.poll_interval = poll_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="wisense-stream-worker", daemon=True)
        self._thread.start()
        logger.info("StreamWorker started")

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None
        logger.info("StreamWorker stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self.source.read_frame()
            except Exception:  # noqa: BLE001 - keep the worker alive; log and retry
                logger.exception("Error reading frame in StreamWorker; retrying")
                time.sleep(0.1)
                continue
            if frame is None:
                if self.poll_interval:
                    time.sleep(self.poll_interval)
                continue
            self.buffer.push(frame)


class Monitor:
    """High-level orchestrator wiring a live source to the presence and
    fall detectors and exposing ``on_presence_change`` /
    ``on_fall_detected`` style hooks, as described in the project
    brief's Event Callback System requirement.

    Runs a :class:`StreamWorker` in the background and polls the buffer
    at ``poll_interval_seconds``, running the configured detectors on
    each poll and emitting events only on meaningful transitions
    (presence flipping True/False; a new fall event).

    Examples
    --------
    >>> from wisense.core import FileCSISource, CSIBuffer
    >>> from wisense.presence.detector import PresenceDetector
    >>> src = FileCSISource("capture.csv")  # doctest: +SKIP
    >>> monitor = Monitor(src, presence_detector=PresenceDetector())  # doctest: +SKIP
    >>> monitor.on_presence_change(lambda result: print(result))  # doctest: +SKIP
    >>> monitor.start()  # doctest: +SKIP
    """

    def __init__(
        self,
        source: CSISource,
        buffer_capacity: int = 256,
        window_size: int = 64,
        poll_interval_seconds: float = 0.5,
        calibration: Optional[CalibrationProfile] = None,
        presence_detector: Optional[Any] = None,
        fall_detector: Optional[Any] = None,
    ) -> None:
        self.source = source
        self.buffer = CSIBuffer(capacity=buffer_capacity)
        self.window_size = window_size
        self.poll_interval_seconds = poll_interval_seconds
        self.calibration = calibration
        self.presence_detector = presence_detector
        self.fall_detector = fall_detector

        self._worker = StreamWorker(self.source, self.buffer)
        self._emitter = EventEmitter()
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_presence: Optional[bool] = None

    def on_presence_change(self, callback: Callable[[Any], None]) -> None:
        """Register ``callback(presence_result)`` to be called whenever
        presence flips from present->absent or absent->present.
        Requires a ``presence_detector`` to have been supplied."""
        self._emitter.on("presence_change", callback)

    def on_fall_detected(self, callback: Callable[[Any], None]) -> None:
        """Register ``callback(fall_event)`` to be called whenever the
        fall detector reports a new, non-``None`` :class:`FallEvent`.
        Requires a ``fall_detector`` to have been supplied."""
        self._emitter.on("fall_detected", callback)

    def start(self) -> None:
        """Connect the source (if not already connected), start the
        background stream worker, and start polling for detector
        results."""
        if not self.source.is_connected:
            self.source.connect()
        self._worker.start()
        self._stop_event.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, name="wisense-monitor-poll", daemon=True)
        self._poll_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._worker.stop()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=5.0)
        self._poll_thread = None

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            window = self.buffer.snapshot(self.window_size)
            if len(window) >= 2:
                if self.presence_detector is not None:
                    try:
                        result = self.presence_detector.detect(window, calibration=self.calibration)
                        if self._last_presence is None or result.present != self._last_presence:
                            self._last_presence = result.present
                            self._emitter.emit("presence_change", result)
                    except Exception:  # noqa: BLE001
                        logger.exception("presence_detector.detect raised during polling")
                if self.fall_detector is not None:
                    try:
                        event = self.fall_detector.detect(window, calibration=self.calibration)
                        if event is not None:
                            self._emitter.emit("fall_detected", event)
                    except Exception:  # noqa: BLE001
                        logger.exception("fall_detector.detect raised during polling")
            self._stop_event.wait(self.poll_interval_seconds)
