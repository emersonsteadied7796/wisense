"""Core primitives shared by every WiSense feature module: connecting to
a CSI data source, buffering frames, conditioning the signal, and
calibrating to an environment baseline."""

from wisense.core.connection import (
    CSIFrame,
    CSISource,
    FileCSISource,
    NetworkCSISource,
    SerialCSISource,
)
from wisense.core.stream import CSIBuffer
from wisense.core.calibration import CalibrationProfile, calibrate
from wisense.core.filters import (
    amplitude_phase,
    butterworth_lowpass_filter,
    hampel_filter,
    moving_average_filter,
    select_subcarriers,
)
from wisense.core.events import EventEmitter, Monitor, StreamWorker

__all__ = [
    "CSIFrame",
    "CSISource",
    "FileCSISource",
    "NetworkCSISource",
    "SerialCSISource",
    "CSIBuffer",
    "CalibrationProfile",
    "calibrate",
    "amplitude_phase",
    "butterworth_lowpass_filter",
    "hampel_filter",
    "moving_average_filter",
    "select_subcarriers",
    "EventEmitter",
    "Monitor",
    "StreamWorker",
]
