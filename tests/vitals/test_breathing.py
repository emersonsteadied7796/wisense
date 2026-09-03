import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import make_breathing_frames, make_still_frames

from wisense.vitals.breathing import estimate_breathing_rate


def test_recovers_known_breathing_frequency():
    sample_rate = 20.0
    breathing_hz = 0.3  # 18 breaths/min
    frames = make_breathing_frames(n_frames=int(sample_rate * 40), sample_rate_hz=sample_rate, breathing_hz=breathing_hz, amplitude_scale=3.0, noise_std=0.02)
    rate = estimate_breathing_rate(frames, sample_rate_hz=sample_rate)
    assert rate is not None
    expected_bpm = breathing_hz * 60.0
    assert rate == pytest.approx(expected_bpm, abs=1.0)


def test_recovers_different_known_frequency():
    sample_rate = 15.0
    breathing_hz = 0.2  # 12 breaths/min
    frames = make_breathing_frames(n_frames=int(sample_rate * 45), sample_rate_hz=sample_rate, breathing_hz=breathing_hz, amplitude_scale=2.5, noise_std=0.02)
    rate = estimate_breathing_rate(frames, sample_rate_hz=sample_rate)
    assert rate is not None
    assert rate == pytest.approx(breathing_hz * 60.0, abs=1.0)


def test_too_short_window_returns_none():
    sample_rate = 20.0
    frames = make_breathing_frames(n_frames=int(sample_rate * 2), sample_rate_hz=sample_rate, breathing_hz=0.3)
    rate = estimate_breathing_rate(frames, sample_rate_hz=sample_rate)
    assert rate is None


def test_pure_noise_returns_none_or_low_confidence_result():
    sample_rate = 20.0
    frames = make_still_frames(n_frames=int(sample_rate * 40), noise_std=0.5)
    rate = estimate_breathing_rate(frames, sample_rate_hz=sample_rate)
    # Pure noise has no reliable spectral peak in the respiration band,
    # so this should not return a confident number.
    assert rate is None


def test_rejects_non_positive_sample_rate():
    frames = make_still_frames(n_frames=20)
    with pytest.raises(ValueError):
        estimate_breathing_rate(frames, sample_rate_hz=0)
