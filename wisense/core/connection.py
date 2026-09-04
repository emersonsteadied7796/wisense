"""Device connection layer.

Defines the :class:`CSIFrame` data container and the :class:`CSISource`
abstract base class, plus three concrete sources:

* :class:`SerialCSISource` -- reads CSI over a serial port from an
  ESP32 running ESP32-CSI-Tool-compatible firmware
  (https://github.com/StevenMHernandez/ESP32-CSI-Tool). The wire format
  parsed here is that project's documented CSV output line, described
  in detail in the ``_parse_esp32_csi_tool_line`` docstring below.
* :class:`NetworkCSISource` -- reads a CSI stream over UDP or TCP using
  a small newline-delimited JSON protocol that WiSense itself defines
  (there is no single industry-standard network CSI wire format, so we
  do not pretend one exists -- see the class docstring for the schema).
* :class:`FileCSISource` -- replays a previously recorded capture from
  disk, using the WiSense CSV capture format documented in the class
  docstring. Useful for tests, demos, and offline development without
  hardware.

All three implement the same ``connect`` / ``disconnect`` / `read_frame`
/ ``stream`` / context-manager interface, so downstream code (buffers,
detectors, calibration) never needs to know which one it's talking to.
"""

from __future__ import annotations

import abc
import json
import logging
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np

from wisense.exceptions import ConnectionError as WiSenseConnectionError
from wisense.exceptions import FrameParseError

logger = logging.getLogger("wisense.core.connection")


@dataclass
class CSIFrame:
    """A single CSI measurement.

    Attributes
    ----------
    timestamp:
        Unix epoch seconds (float) at which the frame was captured or
        received. For replayed files this is the timestamp stored in
        the file, not wall-clock replay time.
    amplitude:
        1-D array of per-subcarrier amplitude, shape ``(n_subcarriers,)``.
    phase:
        1-D array of per-subcarrier phase in radians, shape
        ``(n_subcarriers,)``. May be ``None`` if the source cannot
        provide phase (some cheap capture pipelines only expose
        amplitude).

        **No statistical-baseline detector in this library currently
        reads this field.** Every detection module (presence, fall,
        activity, breathing, people) operates on amplitude only. This
        is deliberate, not an oversight: commodity WiFi radios like the
        ESP32 suffer from carrier/sampling frequency offset and packet
        detection delay severe enough that raw per-packet phase rotates
        essentially randomly across ``[-pi, +pi]`` without proper
        sanitization (linear unwrapping alone, as done in
        :func:`wisense.core.filters.amplitude_phase`, is not
        sufficient -- real phase-based sensing needs per-link
        calibration or a multi-antenna reference to cancel these
        offsets, which this library does not implement). ``phase`` is
        still parsed and exposed here for advanced users who want to
        implement their own phase-based processing, or as an input
        feature to a custom-trained ONNX model, but treat it as raw,
        uncalibrated data -- not something safe to threshold on
        directly the way amplitude is.
    rssi:
        Received signal strength indicator in dBm, if the source
        reports one. ``None`` if unavailable.
    mac:
        Source MAC address string, if known. ``None`` if unavailable.
    """

    timestamp: float
    amplitude: np.ndarray
    phase: Optional[np.ndarray] = None
    rssi: Optional[float] = None
    mac: Optional[str] = None

    def __post_init__(self) -> None:
        self.amplitude = np.asarray(self.amplitude, dtype=np.float64)
        if self.phase is not None:
            self.phase = np.asarray(self.phase, dtype=np.float64)
            if self.phase.shape != self.amplitude.shape:
                raise FrameParseError(
                    f"amplitude shape {self.amplitude.shape} does not match "
                    f"phase shape {self.phase.shape}"
                )

    @property
    def n_subcarriers(self) -> int:
        return int(self.amplitude.shape[0])


class CSISource(abc.ABC):
    """Abstract base class for anything that produces a stream of
    :class:`CSIFrame` objects.

    Concrete subclasses must implement :meth:`connect`,
    :meth:`disconnect`, and :meth:`read_frame`. :meth:`stream` and the
    context-manager protocol (``with source: ...``) are provided here
    in terms of those three primitives, so subclasses get them for
    free.
    """

    def __init__(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abc.abstractmethod
    def connect(self) -> None:
        """Open the underlying connection (serial port, socket, or
        file). Must set ``self._connected = True`` on success and raise
        :class:`wisense.exceptions.ConnectionError` on failure."""
        raise NotImplementedError

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Close the underlying connection. Must be safe to call
        multiple times and safe to call even if ``connect`` was never
        called."""
        raise NotImplementedError

    @abc.abstractmethod
    def read_frame(self) -> Optional[CSIFrame]:
        """Read and return a single :class:`CSIFrame`, or ``None`` if no
        more frames are available (end of file / stream closed).

        Raises :class:`wisense.exceptions.ConnectionError` if called
        before :meth:`connect`, and
        :class:`wisense.exceptions.FrameParseError` if a malformed
        packet is encountered and cannot be recovered from.
        """
        raise NotImplementedError

    def stream(self) -> Iterator[CSIFrame]:
        """Yield frames one at a time until :meth:`read_frame` returns
        ``None``. Does not itself call :meth:`connect`/:meth:`disconnect`;
        callers should use the source as a context manager or call
        those explicitly."""
        while True:
            frame = self.read_frame()
            if frame is None:
                return
            yield frame

    def __enter__(self) -> "CSISource":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def _require_connected(self) -> None:
        if not self._connected:
            raise WiSenseConnectionError(
                f"{type(self).__name__} is not connected; call connect() "
                "first or use it as a context manager (`with source:`)."
            )


class SerialCSISource(CSISource):
    """Reads CSI frames over a serial port from an ESP32 running
    ESP32-CSI-Tool-compatible firmware.

    Wire format
    -----------
    ESP32-CSI-Tool (https://github.com/StevenMHernandez/ESP32-CSI-Tool)
    writes one CSV line per packet to the serial port, of the form::

        CSI_DATA,<type>,<mac>,<rssi>,<rate>,<sig_mode>,<mcs>,<bandwidth>,
        <smoothing>,<not_sounding>,<aggregation>,<stbc>,<fec_coding>,
        <sgi>,<noise_floor>,<ampdu_cnt>,<channel>,<secondary_channel>,
        <local_timestamp>,<ant>,<sig_len>,<rx_state>,<len>,"[<csi_int8_array>]"

    The final field is a bracketed, space-separated list of signed
    8-bit integers, alternating ``imaginary, real`` for each subcarrier,
    i.e. ``len(csi_int8_array) == 2 * n_subcarriers``. This class
    parses exactly that layout. Lines that do not start with
    ``CSI_DATA`` (e.g. firmware log/debug lines interleaved on the same
    serial port) are skipped rather than raising, since that is normal,
    expected traffic on this link.

    Multiple transmitters on the same link
    ---------------------------------------
    An ESP32 running in CSI-sniffing mode can report packets from
    *every* nearby WiFi transmitter it overhears -- your router, other
    people's routers, phones, IoT devices -- not just one. Mixing CSI
    from different transmitters into one time series is meaningless:
    each has its own distance, transmit power, and antenna geometry, so
    amplitude "variance" between two frames from two different
    transmitters reflects who's transmitting, not whether someone moved.
    Pass ``target_mac`` to keep only frames from one specific
    transmitter (e.g. your own router's BSSID) and silently discard the
    rest.

    Hardware timestamps
    --------------------
    The firmware reports ``local_timestamp`` as microseconds since the
    ESP32's own boot, not wall-clock time. USB-to-UART bridges commonly
    deliver serial data in bursts (they buffer for their own USB polling
    interval rather than delivering byte-by-byte), so using
    ``time.time()`` at Python receipt time for every frame introduces
    burst jitter that corrupts time-sensitive analysis like the FFT in
    :func:`wisense.vitals.breathing.estimate_breathing_rate`. To avoid
    this, this class establishes a wall-clock offset from the *first*
    frame it sees (``offset = time.time() - local_timestamp_seconds``)
    and applies that same offset to every subsequent frame's hardware
    timestamp, so frame-to-frame spacing reflects the firmware's own
    microsecond-accurate clock rather than USB delivery jitter.

    Requires ``pyserial`` (``pip install wisense`` pulls it in as a core
    dependency since it is required for this source).
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 921600,
        timeout: float = 2.0,
        target_mac: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.target_mac = target_mac.lower() if target_mac else None
        self._serial = None  # type: ignore[assignment]
        self._time_offset: Optional[float] = None

    def connect(self) -> None:
        try:
            import serial  # pyserial
        except ImportError as exc:  # pragma: no cover - exercised only
            raise WiSenseConnectionError(
                "pyserial is required for SerialCSISource but is not "
                "installed. Install it with `pip install pyserial`."
            ) from exc
        try:
            self._serial = serial.Serial(
                port=self.port, baudrate=self.baudrate, timeout=self.timeout
            )
        except Exception as exc:  # serial.SerialException, OSError, etc.
            raise WiSenseConnectionError(
                f"Failed to open serial port {self.port!r}: {exc}"
            ) from exc
        self._connected = True
        logger.info("Connected to ESP32 CSI source on %s @ %d baud", self.port, self.baudrate)

    def disconnect(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None
        self._connected = False

    def read_frame(self) -> Optional[CSIFrame]:
        self._require_connected()
        assert self._serial is not None
        while True:
            raw = self._serial.readline()
            if not raw:
                # Timeout with no data -- treat as "no frame right now",
                # not end of stream, since the device may still be alive.
                return None
            try:
                line = raw.decode("utf-8", errors="strict").strip()
            except UnicodeDecodeError:
                logger.debug("Discarding non-UTF-8 serial line")
                continue
            if not line.startswith("CSI_DATA"):
                # Firmware debug/log output sharing the same UART.
                continue
            try:
                frame = self._parse_esp32_csi_tool_line(line)
            except FrameParseError as exc:
                logger.warning("Discarding malformed CSI_DATA line: %s", exc)
                continue
            if self.target_mac is not None and (frame.mac is None or frame.mac.lower() != self.target_mac):
                # Frame from a different transmitter than the one we've
                # been asked to track -- see class docstring on why
                # mixing these is meaningless for sensing.
                continue
            return frame

    def _parse_esp32_csi_tool_line(self, line: str) -> CSIFrame:
        """Parse one ``CSI_DATA,...`` line as emitted by ESP32-CSI-Tool.

        See the class docstring for the field layout. Raises
        :class:`FrameParseError` if the line does not have the expected
        number of comma-separated fields or the CSI array cannot be
        parsed as an even-length list of int8 values.

        Instance method (not static) because it maintains
        ``self._time_offset`` to translate the firmware's
        boot-relative microsecond clock into wall-clock time -- see the
        "Hardware timestamps" section of the class docstring.
        """
        # The CSI array field is itself comma-free (space separated,
        # bracket delimited), but it may contain the delimiter used by
        # some firmware builds inside quotes, so split defensively on
        # the bracket rather than assuming a fixed comma count for that
        # final field.
        bracket_start = line.find("[")
        bracket_end = line.rfind("]")
        if bracket_start == -1 or bracket_end == -1 or bracket_end < bracket_start:
            raise FrameParseError(f"no CSI array found in line: {line!r}")

        header = line[:bracket_start].rstrip(', "')
        csi_blob = line[bracket_start + 1 : bracket_end]

        fields = header.split(",")
        # CSI_DATA,type,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,
        # not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,
        # ampdu_cnt,channel,secondary_channel,local_timestamp,ant,
        # sig_len,rx_state,len  == 23 fields before the CSI array
        if len(fields) < 23:
            raise FrameParseError(
                f"expected >=23 header fields before CSI array, got {len(fields)}"
            )

        try:
            mac = fields[2].strip()
            rssi = float(fields[3])
            local_timestamp_us = float(fields[18])
        except (ValueError, IndexError) as exc:
            raise FrameParseError(f"could not parse header fields: {exc}") from exc

        try:
            raw_values = [int(v) for v in csi_blob.replace(",", " ").split()]
        except ValueError as exc:
            raise FrameParseError(f"could not parse CSI int8 array: {exc}") from exc

        if len(raw_values) == 0 or len(raw_values) % 2 != 0:
            raise FrameParseError(
                f"CSI array must have an even number of int8 values "
                f"(imag,real pairs), got {len(raw_values)}"
            )

        arr = np.asarray(raw_values, dtype=np.float64).reshape(-1, 2)
        imag, real = arr[:, 0], arr[:, 1]
        amplitude = np.sqrt(real**2 + imag**2)
        phase = np.arctan2(imag, real)

        # local_timestamp from the firmware is microseconds since boot,
        # not wall-clock time. Establish a wall-clock offset from the
        # first frame seen, then apply that same offset to every
        # subsequent frame's hardware timestamp -- this preserves the
        # firmware's own microsecond-accurate inter-frame spacing
        # instead of using USB-burst-jittery Python receipt time. See
        # the "Hardware timestamps" section of the class docstring.
        local_timestamp_s = local_timestamp_us / 1_000_000.0
        if self._time_offset is None:
            self._time_offset = time.time() - local_timestamp_s
        timestamp = self._time_offset + local_timestamp_s

        return CSIFrame(
            timestamp=timestamp,
            amplitude=amplitude,
            phase=phase,
            rssi=rssi,
            mac=mac,
        )


class NetworkCSISource(CSISource):
    """Reads a CSI stream over UDP or TCP.

    Wire protocol
    --------------
    There is no universal standard for streaming CSI over a network, so
    WiSense defines a minimal one: newline-delimited JSON, one object
    per frame::

        {"t": 1717000000.123, "amp": [1.0, 2.3, ...], "phase": [0.1, ...], "rssi": -42, "mac": "aa:bb:cc:dd:ee:ff"}

    ``t`` (float, unix seconds) and ``amp`` (list of floats) are
    required. ``phase``, ``rssi``, and ``mac`` are optional. Any
    process bridging a real capture pipeline (e.g. a Linux host running
    a CSI-capable driver) to the network need only emit lines in this
    format on the chosen UDP/TCP port.

    For UDP, each datagram is expected to contain exactly one JSON
    object. For TCP, objects are newline-delimited on a persistent
    stream connection.
    """

    def __init__(
        self,
        host: str,
        port: int,
        protocol: str = "udp",
        timeout: float = 2.0,
        recv_buffer_size: int = 65536,
    ) -> None:
        super().__init__()
        protocol = protocol.lower()
        if protocol not in ("udp", "tcp"):
            raise ValueError(f"protocol must be 'udp' or 'tcp', got {protocol!r}")
        self.host = host
        self.port = port
        self.protocol = protocol
        self.timeout = timeout
        self.recv_buffer_size = recv_buffer_size
        self._sock: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._tcp_buffer = b""

    def connect(self) -> None:
        try:
            if self.protocol == "udp":
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._sock.settimeout(self.timeout)
                self._sock.bind((self.host, self.port))
            else:
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.settimeout(self.timeout)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((self.host, self.port))
                listener.listen(1)
                conn, addr = listener.accept()
                conn.settimeout(self.timeout)
                listener.close()
                self._sock = conn
                self._conn = conn
                logger.info("Accepted TCP CSI connection from %s", addr)
        except OSError as exc:
            raise WiSenseConnectionError(
                f"Failed to open {self.protocol.upper()} source on "
                f"{self.host}:{self.port}: {exc}"
            ) from exc
        self._connected = True
        logger.info(
            "Listening for CSI frames over %s on %s:%d", self.protocol.upper(), self.host, self.port
        )

    def disconnect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
                self._conn = None
        self._connected = False

    def read_frame(self) -> Optional[CSIFrame]:
        self._require_connected()
        assert self._sock is not None
        try:
            if self.protocol == "udp":
                data, _addr = self._sock.recvfrom(self.recv_buffer_size)
                line = data
            else:
                line = self._read_tcp_line()
                if line is None:
                    return None
        except socket.timeout:
            return None
        except OSError as exc:
            raise WiSenseConnectionError(f"Network read failed: {exc}") from exc

        if not line.strip():
            return None
        try:
            return self._parse_json_frame(line)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            raise FrameParseError(f"malformed CSI network frame: {exc}") from exc

    def _read_tcp_line(self) -> Optional[bytes]:
        assert self._conn is not None
        while b"\n" not in self._tcp_buffer:
            chunk = self._conn.recv(self.recv_buffer_size)
            if not chunk:
                return None  # peer closed connection
            self._tcp_buffer += chunk
        line, self._tcp_buffer = self._tcp_buffer.split(b"\n", 1)
        return line

    @staticmethod
    def _parse_json_frame(raw: bytes) -> CSIFrame:
        obj = json.loads(raw.decode("utf-8"))
        timestamp = float(obj["t"])
        amplitude = np.asarray(obj["amp"], dtype=np.float64)
        phase = np.asarray(obj["phase"], dtype=np.float64) if obj.get("phase") is not None else None
        rssi = float(obj["rssi"]) if obj.get("rssi") is not None else None
        mac = str(obj["mac"]) if obj.get("mac") is not None else None
        return CSIFrame(timestamp=timestamp, amplitude=amplitude, phase=phase, rssi=rssi, mac=mac)


class FileCSISource(CSISource):
    """Replays a recorded CSI capture from a CSV file on disk. Useful
    for tests, demos, and offline development without hardware.

    File format (WiSense CSV capture format)
    -----------------------------------------
    One header line, then one row per frame::

        timestamp,rssi,mac,amplitude,phase

    * ``timestamp`` -- float, unix seconds.
    * ``rssi`` -- float in dBm, or empty for unknown.
    * ``mac`` -- string, or empty for unknown.
    * ``amplitude`` -- ``;``-separated list of floats, one per
      subcarrier (required).
    * ``phase`` -- ``;``-separated list of floats in radians, same
      length as ``amplitude``, or empty if phase is unavailable.

    Example row::

        1717000000.100,-42.0,aa:bb:cc:dd:ee:ff,1.02;0.98;1.10,0.01;-0.02;0.03

    Use :func:`write_capture_csv` to produce files in this format from
    a list of frames (e.g. when recording a live session for later
    replay).
    """

    def __init__(self, path: Union[str, Path], loop: bool = False, realtime: bool = False) -> None:
        super().__init__()
        self.path = Path(path)
        self.loop = loop
        self.realtime = realtime
        self._fh = None
        self._header_skipped = False
        self._last_frame_time: Optional[float] = None
        self._last_replay_time: Optional[float] = None

    def connect(self) -> None:
        if not self.path.exists():
            raise WiSenseConnectionError(f"Capture file not found: {self.path}")
        try:
            self._fh = open(self.path, "r", newline="", encoding="utf-8")
        except OSError as exc:
            raise WiSenseConnectionError(f"Could not open capture file {self.path}: {exc}") from exc
        self._connected = True
        self._header_skipped = False
        logger.info("Replaying CSI capture from %s (loop=%s)", self.path, self.loop)

    def disconnect(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None
        self._connected = False

    def read_frame(self) -> Optional[CSIFrame]:
        self._require_connected()
        assert self._fh is not None
        if not self._header_skipped:
            self._fh.readline()  # discard header row
            self._header_skipped = True

        line = self._fh.readline()
        if not line:
            if self.loop:
                self._fh.seek(0)
                self._header_skipped = False
                self._last_frame_time = None
                self._last_replay_time = None
                return self.read_frame()
            return None

        line = line.rstrip("\n").rstrip("\r")
        if not line:
            return self.read_frame()

        frame = self._parse_csv_row(line)

        if self.realtime:
            self._pace_realtime(frame.timestamp)

        return frame

    def _pace_realtime(self, frame_timestamp: float) -> None:
        now = time.monotonic()
        if self._last_frame_time is not None and self._last_replay_time is not None:
            delta = frame_timestamp - self._last_frame_time
            elapsed = now - self._last_replay_time
            sleep_for = delta - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._last_frame_time = frame_timestamp
        self._last_replay_time = time.monotonic()

    @staticmethod
    def _parse_csv_row(line: str) -> CSIFrame:
        parts = line.split(",")
        if len(parts) != 5:
            raise FrameParseError(
                f"expected 5 comma-separated fields (timestamp,rssi,mac,"
                f"amplitude,phase), got {len(parts)}: {line!r}"
            )
        ts_s, rssi_s, mac_s, amp_s, phase_s = parts
        try:
            timestamp = float(ts_s)
        except ValueError as exc:
            raise FrameParseError(f"invalid timestamp {ts_s!r}: {exc}") from exc

        rssi = float(rssi_s) if rssi_s.strip() else None
        mac = mac_s.strip() or None

        if not amp_s.strip():
            raise FrameParseError("amplitude field is required and cannot be empty")
        try:
            amplitude = np.asarray([float(v) for v in amp_s.split(";")], dtype=np.float64)
        except ValueError as exc:
            raise FrameParseError(f"invalid amplitude values: {exc}") from exc

        phase = None
        if phase_s.strip():
            try:
                phase = np.asarray([float(v) for v in phase_s.split(";")], dtype=np.float64)
            except ValueError as exc:
                raise FrameParseError(f"invalid phase values: {exc}") from exc

        return CSIFrame(timestamp=timestamp, amplitude=amplitude, phase=phase, rssi=rssi, mac=mac)


def write_capture_csv(path: Union[str, Path], frames: list) -> None:
    """Write a list of :class:`CSIFrame` objects to disk in the WiSense
    CSV capture format documented on :class:`FileCSISource`, so they can
    later be replayed with ``FileCSISource(path)``."""
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write("timestamp,rssi,mac,amplitude,phase\n")
        for frame in frames:
            amp_str = ";".join(repr(float(v)) for v in frame.amplitude)
            phase_str = ";".join(repr(float(v)) for v in frame.phase) if frame.phase is not None else ""
            rssi_str = "" if frame.rssi is None else repr(float(frame.rssi))
            mac_str = frame.mac or ""
            fh.write(f"{frame.timestamp!r},{rssi_str},{mac_str},{amp_str},{phase_str}\n")
