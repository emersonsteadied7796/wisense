import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import make_motion_frames, make_still_frames

from wisense.core.calibration import calibrate
from wisense.core.connection import CSIFrame, FileCSISource, write_capture_csv
from wisense.exceptions import InsufficientSignalError
from wisense.presence.detector import PresenceDetector, PresenceResult


def test_still_frames_detected_as_absent():
    frames = make_still_frames(n_frames=30)
    result = PresenceDetector().detect(frames)
    assert isinstance(result, PresenceResult)
    assert result.present is False
    assert 0.0 <= result.confidence <= 1.0


def test_motion_frames_detected_as_present():
    frames = make_motion_frames(n_frames=30)
    result = PresenceDetector().detect(frames)
    assert result.present is True


def test_too_few_frames_raises():
    frames = make_still_frames(n_frames=2)
    with pytest.raises(InsufficientSignalError):
        PresenceDetector().detect(frames)


def test_uses_calibration_profile(tmp_path: Path):
    quiet_frames = make_still_frames(n_frames=60)
    path = tmp_path / "quiet.csv"
    write_capture_csv(path, quiet_frames)
    with FileCSISource(path) as source:
        profile = calibrate(source, duration_seconds=10.0)

    motion_frames = make_motion_frames(n_frames=30)
    result = PresenceDetector().detect(motion_frames, calibration=profile)
    assert result.present is True

    still_frames = make_still_frames(n_frames=30)
    result2 = PresenceDetector().detect(still_frames, calibration=profile)
    assert result2.present is False


def test_result_timestamp_matches_last_frame():
    frames = make_still_frames(n_frames=10)
    result = PresenceDetector().detect(frames)
    assert result.timestamp == frames[-1].timestamp


def test_model_path_and_registry_model_mutually_exclusive():
    with pytest.raises(ValueError):
        PresenceDetector(model_path="a.onnx", use_registry_model="b.onnx")
