import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import make_fall_frames, make_still_frames, make_walking_frames

from wisense.activity.classifier import ActivityClassifier, ActivityLabel, ActivityResult
from wisense.exceptions import InsufficientSignalError


def test_still_frames_classified_as_still():
    frames = make_still_frames(n_frames=20)
    result = ActivityClassifier().classify(frames)
    assert isinstance(result, ActivityResult)
    assert result.label == ActivityLabel.STILL


def test_walking_frames_classified_as_walking_or_unknown():
    frames = make_walking_frames(n_frames=60)
    result = ActivityClassifier().classify(frames)
    # A periodic gait-like signal should be recognized as WALKING by
    # the zero-crossing heuristic; assert it is not confused with a
    # single-transient posture change.
    assert result.label in (ActivityLabel.WALKING, ActivityLabel.UNKNOWN)
    assert result.label not in (ActivityLabel.SITTING_DOWN, ActivityLabel.STANDING_UP)


def test_posture_change_signature_not_still():
    frames = make_fall_frames(n_pre=10, n_post=20, drop=15.0)
    result = ActivityClassifier().classify(frames)
    assert result.label != ActivityLabel.STILL


def test_too_few_frames_raises():
    with pytest.raises(InsufficientSignalError):
        ActivityClassifier().classify(make_still_frames(n_frames=3))


def test_confidence_in_valid_range():
    frames = make_still_frames(n_frames=20)
    result = ActivityClassifier().classify(frames)
    assert 0.0 <= result.confidence <= 1.0


def test_model_path_and_registry_model_mutually_exclusive():
    with pytest.raises(ValueError):
        ActivityClassifier(model_path="a.onnx", use_registry_model="b.onnx")
