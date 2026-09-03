from pathlib import Path

import numpy as np
import pytest

from wisense.core.calibration import calibrate
from wisense.core.connection import FileCSISource, write_capture_csv, CSIFrame
from wisense.exceptions import CalibrationError


def _write_frames(path: Path, n: int, n_sub: int = 8, noise_std: float = 0.1):
    rng = np.random.default_rng(0)
    frames = [
        CSIFrame(timestamp=i * 0.05, amplitude=10.0 + rng.normal(0, noise_std, size=n_sub))
        for i in range(n)
    ]
    write_capture_csv(path, frames)


def test_calibrate_produces_reasonable_profile(tmp_path: Path):
    path = tmp_path / "quiet.csv"
    _write_frames(path, n=60)
    with FileCSISource(path) as source:
        profile = calibrate(source, duration_seconds=5.0)
    assert profile.n_frames == 60
    assert profile.n_subcarriers == 8
    assert profile.baseline_variance_p95 >= 0
    assert profile.baseline_mean.shape == (8,)
    assert profile.baseline_std.shape == (8,)


def test_calibrate_too_few_frames_raises(tmp_path: Path):
    path = tmp_path / "tiny.csv"
    _write_frames(path, n=3)
    with FileCSISource(path) as source:
        with pytest.raises(CalibrationError):
            calibrate(source, duration_seconds=5.0)


def test_calibrate_rejects_non_positive_duration(tmp_path: Path):
    path = tmp_path / "quiet.csv"
    _write_frames(path, n=20)
    with FileCSISource(path) as source:
        with pytest.raises(ValueError):
            calibrate(source, duration_seconds=0)


def test_variance_threshold_scales_with_spread():
    from wisense.core.calibration import CalibrationProfile
    import time

    low_spread = CalibrationProfile(
        baseline_mean=np.array([10.0]),
        baseline_std=np.array([0.1]),
        baseline_variance_p95=0.05,
        n_frames=50,
        n_subcarriers=1,
        duration_seconds=5.0,
        created_at=time.time(),
    )
    high_spread = CalibrationProfile(
        baseline_mean=np.array([10.0]),
        baseline_std=np.array([2.0]),
        baseline_variance_p95=0.05,
        n_frames=50,
        n_subcarriers=1,
        duration_seconds=5.0,
        created_at=time.time(),
    )
    assert high_spread.variance_threshold() > low_spread.variance_threshold()
