"""Exception hierarchy for WiSense.

All exceptions raised by the WiSense library derive from
:class:`WiSenseError`, so callers who only care about "something went
wrong in WiSense" can catch a single base class, while callers who need
finer-grained handling can catch the specific subclass.
"""

from __future__ import annotations


class WiSenseError(Exception):
    """Base class for all exceptions raised by WiSense."""


class ConnectionError(WiSenseError):
    """Raised when a :class:`~wisense.core.connection.CSISource` fails to
    connect to, or loses connection to, its underlying data source
    (serial port, network socket, or file)."""


class FrameParseError(WiSenseError):
    """Raised when a raw CSI packet/line/datagram cannot be parsed into a
    valid :class:`~wisense.core.connection.CSIFrame`."""


class CalibrationError(WiSenseError):
    """Raised when environment calibration fails or is used incorrectly
    (e.g. too few frames collected, or calibration applied to a source
    with an incompatible number of subcarriers)."""


class ModelNotFoundError(WiSenseError):
    """Raised by :mod:`wisense.models.registry` when a requested ONNX
    model cannot be located locally and cannot be downloaded (missing
    URL, network failure, or checksum mismatch)."""


class ModelChecksumError(ModelNotFoundError):
    """Raised when a downloaded model file's checksum does not match the
    expected value. Left as a subclass of :class:`ModelNotFoundError` so
    callers that only catch the parent still see it."""


class InsufficientSignalError(WiSenseError):
    """Raised (or used as a documented return-``None`` reason) when a
    frame window does not contain enough samples, or samples of
    sufficient quality, to produce a confident result (e.g. breathing
    rate estimation on too short a window)."""


class BufferError(WiSenseError):
    """Raised for invalid operations on a :class:`~wisense.core.stream.CSIBuffer`
    (e.g. requesting more frames than a non-blocking read allows)."""
