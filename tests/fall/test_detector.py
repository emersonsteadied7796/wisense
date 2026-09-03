import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import make_fall_frames, make_still_frames, make_walking_frames

from wisense.exceptions import InsufficientSignalError
from wisense.fall.detector import FallDetector, FallEvent, FallSeverity


def test_still_frames_produce_no_fall():
    frames = make_still_frames(n_frames=20)
    event = FallDetector().detect(frames)
    assert event is None


def test_fall_signature_detected():
    frames = make_fall_frames(n_pre=10, n_post=20, drop=15.0)
    event = FallDetector().detect(frames)
    assert event is not None
    assert isinstance(event, FallEvent)
    assert event.severity in (FallSeverity.LIKELY, FallSeverity.CONFIRMED)
    assert 0.0 <= event.confidence <= 1.0


def test_fall_with_short_post_window_is_possible_severity():
    frames = make_fall_frames(n_pre=10, n_post=1, drop=15.0)
    event = FallDetector().detect(frames)
    assert event is not None
    assert event.severity == FallSeverity.POSSIBLE


def test_walking_does_not_trigger_confirmed_fall():
    frames = make_walking_frames(n_frames=60)
    event = FallDetector().detect(frames)
    # Walking may or may not cross the spike threshold, but it should
    # never look like a CONFIRMED fall since motion continues throughout.
    if event is not None:
        assert event.severity != FallSeverity.CONFIRMED


def test_too_few_frames_raises():
    with pytest.raises(InsufficientSignalError):
        FallDetector().detect(make_still_frames(n_frames=3))


def test_model_path_and_registry_model_mutually_exclusive():
    with pytest.raises(ValueError):
        FallDetector(model_path="a.onnx", use_registry_model="b.onnx")


def test_detected_at_is_within_window_timestamps():
    frames = make_fall_frames(n_pre=10, n_post=20, drop=15.0)
    event = FallDetector().detect(frames)
    assert event is not None
    timestamps = [f.timestamp for f in frames]
    assert min(timestamps) <= event.detected_at <= max(timestamps)
