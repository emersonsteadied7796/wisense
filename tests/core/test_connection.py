import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from wisense.core.connection import (
    CSIFrame,
    FileCSISource,
    NetworkCSISource,
    SerialCSISource,
    write_capture_csv,
)
from wisense.exceptions import ConnectionError as WiSenseConnectionError
from wisense.exceptions import FrameParseError


def test_csiframe_computes_n_subcarriers():
    frame = CSIFrame(timestamp=1.0, amplitude=np.array([1.0, 2.0, 3.0]))
    assert frame.n_subcarriers == 3


def test_csiframe_rejects_mismatched_phase_shape():
    with pytest.raises(FrameParseError):
        CSIFrame(timestamp=1.0, amplitude=np.array([1.0, 2.0]), phase=np.array([0.1, 0.2, 0.3]))


def test_file_source_roundtrip(tmp_path: Path):
    frames = [
        CSIFrame(timestamp=0.0, amplitude=np.array([1.0, 2.0, 3.0]), phase=np.array([0.1, 0.2, 0.3]), rssi=-40.0, mac="aa:bb:cc:dd:ee:ff"),
        CSIFrame(timestamp=0.1, amplitude=np.array([1.1, 2.1, 3.1])),
    ]
    path = tmp_path / "capture.csv"
    write_capture_csv(path, frames)

    with FileCSISource(path) as source:
        f1 = source.read_frame()
        f2 = source.read_frame()
        f3 = source.read_frame()

    assert f1 is not None and f2 is not None
    assert f3 is None
    np.testing.assert_allclose(f1.amplitude, frames[0].amplitude)
    np.testing.assert_allclose(f1.phase, frames[0].phase)
    assert f1.rssi == -40.0
    assert f1.mac == "aa:bb:cc:dd:ee:ff"
    assert f2.phase is None


def test_file_source_missing_file_raises_connection_error(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    source = FileCSISource(missing)
    with pytest.raises(WiSenseConnectionError):
        source.connect()


def test_file_source_read_before_connect_raises(tmp_path: Path):
    path = tmp_path / "capture.csv"
    write_capture_csv(path, [CSIFrame(timestamp=0.0, amplitude=np.array([1.0]))])
    source = FileCSISource(path)
    with pytest.raises(WiSenseConnectionError):
        source.read_frame()


def test_file_source_loop(tmp_path: Path):
    frames = [CSIFrame(timestamp=float(i), amplitude=np.array([1.0, 2.0])) for i in range(3)]
    path = tmp_path / "capture.csv"
    write_capture_csv(path, frames)

    with FileCSISource(path, loop=True) as source:
        read = [source.read_frame() for _ in range(7)]  # more than 3, should wrap around
    assert all(f is not None for f in read)
    assert len(read) == 7


def test_file_source_rejects_malformed_row(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("timestamp,rssi,mac,amplitude,phase\n0.0,,,,\n")
    with FileCSISource(path) as source:
        with pytest.raises(FrameParseError):
            source.read_frame()


def test_file_source_stream_generator(tmp_path: Path):
    frames = [CSIFrame(timestamp=float(i), amplitude=np.array([1.0, 2.0])) for i in range(5)]
    path = tmp_path / "capture.csv"
    write_capture_csv(path, frames)
    with FileCSISource(path) as source:
        collected = list(source.stream())
    assert len(collected) == 5


def test_serial_parse_esp32_csi_tool_line():
    # A synthetic line matching the documented ESP32-CSI-Tool CSV layout:
    # 23 header fields, then a bracketed [imag,real, imag,real, ...] array.
    header_fields = ["CSI_DATA", "0", "aa:bb:cc:dd:ee:ff", "-45", "1", "1", "0", "1", "0", "0", "0", "0", "0", "0", "-90", "0", "6", "0", "123456", "0", "128", "0", "10"]
    csi_values = "[1 2 -1 3 0 4]"  # 3 subcarriers: (imag,real) pairs
    line = ",".join(header_fields) + "," + csi_values
    source = SerialCSISource(port="/dev/null")
    frame = source._parse_esp32_csi_tool_line(line)
    assert frame.n_subcarriers == 3
    assert frame.rssi == -45.0
    assert frame.mac == "aa:bb:cc:dd:ee:ff"
    # imag=1,real=2 -> amplitude sqrt(5)
    assert frame.amplitude[0] == pytest.approx((1**2 + 2**2) ** 0.5)


def test_serial_parse_establishes_wall_clock_offset_from_first_frame():
    header_fields = ["CSI_DATA", "0", "aa:bb:cc:dd:ee:ff", "-45", "1", "1", "0", "1", "0", "0", "0", "0", "0", "0", "-90", "0", "6", "0", "1000000", "0", "128", "0", "10"]
    line1 = ",".join(header_fields) + ",[1 2]"
    header_fields2 = list(header_fields)
    header_fields2[18] = "1500000"  # +0.5s of hardware time later
    line2 = ",".join(header_fields2) + ",[1 2]"

    source = SerialCSISource(port="/dev/null")
    f1 = source._parse_esp32_csi_tool_line(line1)
    f2 = source._parse_esp32_csi_tool_line(line2)
    # Frame-to-frame spacing should track the *hardware* clock delta
    # (0.5s) regardless of how much wall-clock time actually elapsed
    # between the two Python calls.
    assert f2.timestamp - f1.timestamp == pytest.approx(0.5, abs=1e-6)


def test_serial_target_mac_filters_other_transmitters():
    header_fields = ["CSI_DATA", "0", "aa:bb:cc:dd:ee:ff", "-45", "1", "1", "0", "1", "0", "0", "0", "0", "0", "0", "-90", "0", "6", "0", "123456", "0", "128", "0", "10"]
    other_mac_fields = list(header_fields)
    other_mac_fields[2] = "11:22:33:44:55:66"
    line_other = ",".join(other_mac_fields) + ",[1 2]"
    line_target = ",".join(header_fields) + ",[1 2]"

    source = SerialCSISource(port="/dev/null", target_mac="AA:BB:CC:DD:EE:FF")
    # Parsing itself doesn't filter -- filtering happens in read_frame().
    # Directly verify the filtering condition matches what read_frame() uses:
    frame_other = source._parse_esp32_csi_tool_line(line_other)
    frame_target = source._parse_esp32_csi_tool_line(line_target)
    assert frame_other.mac.lower() != source.target_mac
    assert frame_target.mac.lower() == source.target_mac


def test_serial_parse_rejects_too_few_header_fields():
    source = SerialCSISource(port="/dev/null")
    with pytest.raises(FrameParseError):
        source._parse_esp32_csi_tool_line("CSI_DATA,0,1 [1 2]")


def test_serial_parse_rejects_odd_length_csi_array():
    header_fields = ["CSI_DATA"] + ["0"] * 22
    line = ",".join(header_fields) + ",[1 2 3]"
    source = SerialCSISource(port="/dev/null")
    with pytest.raises(FrameParseError):
        source._parse_esp32_csi_tool_line(line)


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_network_udp_source_receives_frame():
    port = _free_udp_port()
    source = NetworkCSISource("127.0.0.1", port, protocol="udp", timeout=2.0)
    source.connect()
    try:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        payload = b'{"t": 1.5, "amp": [1.0, 2.0, 3.0], "rssi": -50, "mac": "aa:bb:cc:dd:ee:ff"}'
        sender.sendto(payload, ("127.0.0.1", port))
        sender.close()
        frame = source.read_frame()
        assert frame is not None
        assert frame.timestamp == 1.5
        np.testing.assert_allclose(frame.amplitude, [1.0, 2.0, 3.0])
        assert frame.rssi == -50.0
    finally:
        source.disconnect()


def test_network_udp_source_timeout_returns_none():
    port = _free_udp_port()
    source = NetworkCSISource("127.0.0.1", port, protocol="udp", timeout=0.2)
    source.connect()
    try:
        frame = source.read_frame()
        assert frame is None
    finally:
        source.disconnect()


def test_network_udp_source_malformed_json_raises():
    port = _free_udp_port()
    source = NetworkCSISource("127.0.0.1", port, protocol="udp", timeout=2.0)
    source.connect()
    try:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.sendto(b"not json", ("127.0.0.1", port))
        sender.close()
        with pytest.raises(FrameParseError):
            source.read_frame()
    finally:
        source.disconnect()


def test_network_tcp_source_roundtrip():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    source = NetworkCSISource("127.0.0.1", port, protocol="tcp", timeout=3.0)

    result = {}

    def server():
        result["frame"] = None
        source.connect()
        result["frame"] = source.read_frame()

    thread = threading.Thread(target=server)
    thread.start()
    time.sleep(0.3)  # let the server start listening

    client = socket.create_connection(("127.0.0.1", port), timeout=3.0)
    client.sendall(b'{"t": 2.0, "amp": [4.0, 5.0]}\n')
    thread.join(timeout=5.0)
    client.close()
    source.disconnect()

    assert result["frame"] is not None
    assert result["frame"].timestamp == 2.0
